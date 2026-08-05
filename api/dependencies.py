"""
Application state and dependency injection.

Holds the WorkflowExecutor (core SDK) initialized at startup and
available to all request handlers via get_app_state().
"""

import hmac
from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, Request

from common.config import Config, load_config
from common.logging import configure_logging, get_logger
from core.executor import WorkflowExecutor
from core.place_to_book import PlaceToBookResolver
from core.types import ExecutorConfig
from api.ratelimit import InFlightLimiter, SlidingWindowRateLimiter


@dataclass
class AppState:
    """Shared application state initialized at startup."""

    config: Config
    executor: WorkflowExecutor
    rate_limiter: SlidingWindowRateLimiter
    inflight_limiter: InFlightLimiter


_app_state: Optional[AppState] = None
_place_to_book_resolver: Optional[PlaceToBookResolver] = None


async def initialize() -> AppState:
    """Initialize application state with WorkflowExecutor."""
    global _app_state

    config = load_config()
    configure_logging(
        level=config.log_level, enable_adk_debug=config.enable_adk_debug
    )
    logger = get_logger("storyland.api")

    executor_config = ExecutorConfig.from_config(config)
    executor = WorkflowExecutor(executor_config)

    rate_limiter = SlidingWindowRateLimiter(
        max_requests=config.rate_limit_requests,
        window_seconds=config.rate_limit_window_seconds,
    )
    inflight_limiter = InFlightLimiter(max_in_flight=config.max_inflight_requests)

    _app_state = AppState(
        config=config,
        executor=executor,
        rate_limiter=rate_limiter,
        inflight_limiter=inflight_limiter,
    )

    # Boot-time visibility of the EFFECTIVE cache config so a dropped/
    # misconfigured setting is immediately obvious in prod logs (rather than
    # silently serving with caching off). Warn loudly if a cache expected-on
    # resolved to disabled.
    logger.info(
        "cache_config_effective",
        enabled=config.cache_enabled,
        ttl_seconds=config.cache_ttl_seconds,
        max_entries=config.cache_max_entries,
        backend=config.cache_backend,
        cache_dir=config.cache_dir if config.cache_backend == "disk" else None,
    )
    if not config.cache_enabled:
        logger.warning(
            "cache_disabled",
            detail=(
                "Discovery result cache is DISABLED (CACHE_ENABLED=false): "
                "every discover re-pays live Gemini cost. "
                "Unset CACHE_ENABLED to fall back to on."
            ),
        )

    # Boot-time visibility of the EFFECTIVE gateway auth config. An empty
    # INTERNAL_API_SECRET disables service-to-service auth entirely (every
    # caller is accepted, and X-User-ID is then a forgeable identity), and until
    # now that state was completely SILENT — nothing at boot or per-request said
    # so, unlike the cache, which has warned loudly since day one. A
    # misconfiguration you cannot see in the logs is a misconfiguration you
    # cannot fix. This is that missing signal.
    gateway_auth_enforced = bool(config.internal_api_secret)
    logger.info(
        "gateway_auth_effective",
        enforced=gateway_auth_enforced,
        required=config.require_gateway_secret,
        environment=config.environment,
    )
    if not gateway_auth_enforced:
        if config.require_gateway_secret:
            # Fail closed: refuse to start rather than serve unauthenticated.
            logger.critical(
                "gateway_auth_misconfigured",
                detail=(
                    "INTERNAL_API_SECRET is empty but REQUIRE_GATEWAY_SECRET=true "
                    "(the default): refusing to start. Set INTERNAL_API_SECRET "
                    "(the backend gateway reads the same variable) or explicitly "
                    "set REQUIRE_GATEWAY_SECRET=false to run standalone (dev only) "
                    "without service-to-service auth."
                ),
                environment=config.environment,
            )
            raise RuntimeError(
                "INTERNAL_API_SECRET is empty but REQUIRE_GATEWAY_SECRET=true. "
                "Refusing to start without service-to-service authentication."
            )
        logger.warning(
            "gateway_auth_disabled",
            detail=(
                "Service-to-service auth is DISABLED (INTERNAL_API_SECRET is empty): "
                "the X-Internal-Secret check accepts EVERY caller, and X-User-ID is "
                "then an unverified, forgeable identity — any caller able to reach "
                "this service can drive Gemini spend and read or mutate any user's "
                "sessions and itineraries. Set INTERNAL_API_SECRET (the backend "
                "gateway reads the same variable), then remove "
                "REQUIRE_GATEWAY_SECRET=false (the default is true) so this state "
                "is fatal instead of merely loud."
            ),
            environment=config.environment,
        )

    logger.info("api_initialized", model=config.model_name)
    return _app_state


async def shutdown() -> None:
    """Cleanup on application shutdown."""
    global _app_state, _place_to_book_resolver
    if _app_state:
        await _app_state.executor.close()
    _app_state = None
    _place_to_book_resolver = None


def get_app_state() -> AppState:
    """Get the initialized application state."""
    if _app_state is None:
        raise RuntimeError("Application not initialized. Call initialize() first.")
    return _app_state


def get_place_to_book_resolver() -> PlaceToBookResolver:
    """Return the process-wide place→book resolver (lazy singleton).

    Reuses the executor's already-constructed model so we don't spin up a second
    Gemini client, and keeps the resolver's in-process cache warm across
    requests. The resolver keeps its own (in-memory) session service, isolated
    from the discovery/compose session lifecycle.
    """
    global _place_to_book_resolver
    if _place_to_book_resolver is None:
        app_state = get_app_state()
        _place_to_book_resolver = PlaceToBookResolver(model=app_state.executor.model)
    return _place_to_book_resolver


def verify_gateway_secret(request: Request) -> None:
    """Require a matching X-Internal-Secret header when the gateway secret is set.

    Behaviour is deliberately explicit about the empty-secret case, which used to
    be an accident of ``if secret and ...``:

    * secret configured -> the header MUST match, else HTTP 403 (fail closed).
    * secret empty       -> every caller is accepted. This is a misconfiguration,
      not a feature. It is no longer silent: boot logs ``gateway_auth_disabled``
      (WARNING), and with ``REQUIRE_GATEWAY_SECRET=true`` — the default — the
      service refuses to start at all (see ``initialize``), so this permissive
      state is only reachable behind an explicit ``REQUIRE_GATEWAY_SECRET=false``
      opt-out. The check is left permissive HERE rather
      than 403-ing at request time so that the failure mode of a misconfigured
      deploy is a loud log, not a silent total outage of the discovery chain.

    The comparison is constant-time: ``==`` on a secret leaks its prefix length
    through timing, and this header is attacker-supplied.

    It compares BYTES, not ``str``. ``hmac.compare_digest`` accepts ``str`` only
    when BOTH operands are ASCII-only, but Starlette decodes header values as
    latin-1 -- so one attacker-supplied byte in 0x80-0xFF arrives as a non-ASCII
    ``str`` and makes ``compare_digest`` raise ``TypeError``, turning a clean 403
    into an unhandled 500 on the one input an attacker fully controls. Encoding
    the presented value back to latin-1 recovers the exact bytes that came off
    the wire (``errors="replace"`` cannot itself raise, even on a hand-built
    Request carrying codepoints > 255), and the configured secret is encoded as
    UTF-8 -- which is what the backend gateway puts on the wire.
    """
    secret = get_app_state().config.internal_api_secret
    if not secret:
        return
    presented = request.headers.get("X-Internal-Secret") or ""
    if not hmac.compare_digest(
        presented.encode("latin-1", "replace"), secret.encode("utf-8")
    ):
        raise HTTPException(status_code=403, detail="Forbidden")


def get_gateway_user_id(
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
) -> str:
    """
    Resolve the authenticated user_id from the trusted X-User-ID header.

    The backend gateway sets X-User-ID only AFTER it has validated the caller's
    JWT, so this service trusts X-User-ID and nothing else. It deliberately does
    NOT inspect or trust the raw Authorization JWT: this service cannot verify
    the token's signature, and trusting an unverified claim would let a forged
    token impersonate any user (sessions/itineraries are scoped by user_id).

    Resolution:
    1. X-User-ID present                          -> that user_id.
    2. Absent + ALLOW_DEV_USER=true (local dev)   -> shared 'dev_user'.
    3. Absent otherwise                           -> fail closed, HTTP 403.

    Failing closed removes the previous foot-gun: whenever INTERNAL_API_SECRET
    was unset, identity silently fell back to an unverified JWT claim (or a
    shared 'dev_user'), so a single misconfigured deploy could open cross-tenant
    access. Identity now requires the trusted header regardless of that secret.
    """
    if x_user_id:
        return x_user_id
    if get_app_state().config.allow_dev_user:
        return "dev_user"
    raise HTTPException(status_code=403, detail="X-User-ID header is required")


def _rate_limit_key(request: Request, x_user_id: str | None) -> str:
    """Identify the caller for rate limiting: user id when known, else client IP."""
    if x_user_id:
        return f"user:{x_user_id}"
    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


def enforce_rate_limit(
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
) -> None:
    """Per-identity request-rate cap on the expensive endpoints.

    No-op unless RATE_LIMIT_REQUESTS is configured (> 0). Raises HTTP 429 with a
    Retry-After hint when a user/IP exceeds its window budget.
    """
    limiter = get_app_state().rate_limiter
    if not limiter.enabled:
        return
    if not limiter.allow(_rate_limit_key(request, x_user_id)):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down and retry shortly.",
            headers={"Retry-After": str(get_app_state().config.rate_limit_window_seconds)},
        )


async def limit_inflight():
    """Bounded concurrency guard for one expensive request.

    No-op unless MAX_INFLIGHT_REQUESTS is configured (> 0). Acquires a slot for
    the duration of the request (held across SSE streaming via the yield) and
    releases it when the response completes. Sheds load with HTTP 503 when the
    box is already at capacity rather than queueing work onto the single loop.
    """
    limiter = get_app_state().inflight_limiter
    if not limiter.try_acquire():
        raise HTTPException(
            status_code=503,
            detail="Server is busy processing other requests. Please retry shortly.",
            headers={"Retry-After": "5"},
        )
    try:
        yield
    finally:
        limiter.release()
