"""Fail-closed identity guards for ``get_gateway_user_id``.

Regression coverage for the unverified-signature-JWT / shared-``dev_user``
foot-gun: ``api/dependencies.get_gateway_user_id`` previously fell back to an
*unverified* JWT claim (or a shared ``dev_user``) whenever ``INTERNAL_API_SECRET``
was unset, so a forged token — or a single misconfigured deploy — could read or
write another user's sessions/itineraries (everything is scoped by ``user_id``).

The hardened contract these tests pin:
  * trust ONLY the gateway-set ``X-User-ID`` header,
  * never parse/trust the raw Authorization JWT,
  * fall back to ``dev_user`` ONLY behind the explicit local ``ALLOW_DEV_USER`` flag,
  * otherwise fail closed with HTTP 403 (never a shared id).

The second half of this module covers ``verify_gateway_secret`` — the
service-to-service check that gates the same endpoints. It had NO negative
coverage at all (MYS-403 gap 1): every existing test asserted identity, none
asserted that a wrong or missing X-Internal-Secret is rejected, and none pinned
what happens when the secret is empty. An empty INTERNAL_API_SECRET skips the
check entirely and accepts every caller, and until now nothing — no test, no log
line — said so out loud.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import api.dependencies as deps


@pytest.fixture
def install_state(monkeypatch):
    """Install a stub AppState whose config exposes the auth-relevant fields."""

    def _install(internal_api_secret: str = "", allow_dev_user: bool = False):
        config = SimpleNamespace(
            internal_api_secret=internal_api_secret,
            allow_dev_user=allow_dev_user,
        )
        monkeypatch.setattr(deps, "get_app_state", lambda: SimpleNamespace(config=config))

    return _install


def test_returns_trusted_x_user_id(install_state):
    """Present X-User-ID is the identity (the only trusted source)."""
    install_state(internal_api_secret="s3cret")
    assert deps.get_gateway_user_id(x_user_id="alice") == "alice"


def test_missing_identity_fails_closed_when_secret_set(install_state):
    """No X-User-ID with the secret set -> 403 (unchanged prod behaviour)."""
    install_state(internal_api_secret="s3cret")
    with pytest.raises(HTTPException) as exc:
        deps.get_gateway_user_id(x_user_id=None)
    assert exc.value.status_code == 403


def test_missing_identity_fails_closed_when_secret_unset(install_state):
    """The core fix: secret unset must NOT collapse to a shared id — fail closed."""
    install_state(internal_api_secret="", allow_dev_user=False)
    with pytest.raises(HTTPException) as exc:
        deps.get_gateway_user_id(x_user_id=None)
    assert exc.value.status_code == 403


def test_dev_user_only_behind_explicit_flag(install_state):
    """``dev_user`` is reachable only when ALLOW_DEV_USER is explicitly true."""
    install_state(internal_api_secret="", allow_dev_user=True)
    assert deps.get_gateway_user_id(x_user_id=None) == "dev_user"


def test_x_user_id_wins_even_with_dev_flag(install_state):
    """A real trusted id always beats the dev fallback."""
    install_state(internal_api_secret="", allow_dev_user=True)
    assert deps.get_gateway_user_id(x_user_id="bob") == "bob"


def test_unverified_jwt_helper_is_removed():
    """The unverified-signature JWT path must be gone entirely."""
    assert not hasattr(deps, "_user_from_jwt")


# --------------------------------------------------------------------------
# verify_gateway_secret — the service-to-service check
# --------------------------------------------------------------------------


def _request(secret_header: str | None = None) -> SimpleNamespace:
    """A stand-in Request exposing only the header map the check reads."""
    headers = {} if secret_header is None else {"X-Internal-Secret": secret_header}
    return SimpleNamespace(headers=headers)


def test_matching_secret_is_accepted(install_state):
    """The happy path: the gateway presents the configured secret."""
    install_state(internal_api_secret="s3cret")
    assert deps.verify_gateway_secret(_request("s3cret")) is None


def test_wrong_secret_is_rejected(install_state):
    """A wrong X-Internal-Secret must 403 (the check's whole purpose)."""
    install_state(internal_api_secret="s3cret")
    with pytest.raises(HTTPException) as exc:
        deps.verify_gateway_secret(_request("not-the-secret"))
    assert exc.value.status_code == 403


def test_missing_secret_header_is_rejected(install_state):
    """No X-Internal-Secret at all, with the secret configured -> 403."""
    install_state(internal_api_secret="s3cret")
    with pytest.raises(HTTPException) as exc:
        deps.verify_gateway_secret(_request(None))
    assert exc.value.status_code == 403


def test_empty_secret_header_is_rejected(install_state):
    """An empty header value must not satisfy a configured secret."""
    install_state(internal_api_secret="s3cret")
    with pytest.raises(HTTPException) as exc:
        deps.verify_gateway_secret(_request(""))
    assert exc.value.status_code == 403


def test_secret_prefix_is_not_enough(install_state):
    """A prefix of the real secret is not a match (guards a sloppy startswith)."""
    install_state(internal_api_secret="s3cret-long-value")
    with pytest.raises(HTTPException) as exc:
        deps.verify_gateway_secret(_request("s3cret"))
    assert exc.value.status_code == 403


def test_empty_config_accepts_every_caller_and_that_is_a_misconfiguration(install_state):
    """PINNED, not endorsed: an empty INTERNAL_API_SECRET accepts ANY caller.

    This is the fail-open. The request path stays permissive on purpose — 403-ing
    here would turn a misconfigured deploy into a silent total outage of the
    discovery chain. What changed is that the state is no longer invisible: boot
    logs ``gateway_auth_disabled`` (see the initialize tests below), and
    ``REQUIRE_GATEWAY_SECRET=true`` makes it fatal at startup.

    If this test ever fails, someone made the empty case reject at request time.
    That may well be right — but it is an outage-shaped change, and it must be a
    deliberate one, made with the prod value of INTERNAL_API_SECRET known.
    """
    install_state(internal_api_secret="")
    assert deps.verify_gateway_secret(_request(None)) is None
    assert deps.verify_gateway_secret(_request("anything-at-all")) is None


# --------------------------------------------------------------------------
# initialize() — boot-time visibility and the fail-closed switch
# --------------------------------------------------------------------------


class _FakeLogger:
    """Records structlog-style calls so the boot signal can be asserted."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def _record(self, level):
        def _log(event, **kwargs):
            self.calls.append((level, event))

        return _log

    def __getattr__(self, level):
        return self._record(level)

    def events(self, level: str) -> list[str]:
        return [event for lvl, event in self.calls if lvl == level]


@pytest.fixture
def boot(monkeypatch):
    """Run initialize() with every heavy collaborator stubbed out."""
    logger = _FakeLogger()

    def _boot(internal_api_secret: str = "", require_gateway_secret: bool = False):
        config = SimpleNamespace(
            internal_api_secret=internal_api_secret,
            require_gateway_secret=require_gateway_secret,
            environment="production",
            log_level="INFO",
            enable_adk_debug=False,
            model_name="test-model",
            cache_enabled=True,
            cache_ttl_seconds=1,
            cache_max_entries=1,
            cache_backend="memory",
            cache_dir="/tmp/none",
            rate_limit_requests=0,
            rate_limit_window_seconds=60,
            max_inflight_requests=0,
        )
        monkeypatch.setattr(deps, "load_config", lambda: config)
        monkeypatch.setattr(deps, "configure_logging", lambda **kw: None)
        monkeypatch.setattr(deps, "get_logger", lambda *a, **kw: logger)
        monkeypatch.setattr(
            deps.ExecutorConfig, "from_config", staticmethod(lambda c: SimpleNamespace())
        )
        monkeypatch.setattr(deps, "WorkflowExecutor", lambda cfg: SimpleNamespace())
        monkeypatch.setattr(deps, "SlidingWindowRateLimiter", lambda **kw: SimpleNamespace())
        monkeypatch.setattr(deps, "InFlightLimiter", lambda **kw: SimpleNamespace())
        return logger

    yield _boot
    deps._app_state = None


@pytest.mark.asyncio
async def test_boot_warns_loudly_when_gateway_secret_is_empty(boot):
    """The core fix: an empty secret is no longer silent at boot."""
    logger = boot(internal_api_secret="")
    await deps.initialize()
    assert "gateway_auth_disabled" in logger.events("warning")
    assert "gateway_auth_effective" in logger.events("info")


@pytest.mark.asyncio
async def test_boot_is_quiet_when_gateway_secret_is_set(boot):
    """A correctly configured deploy states the fact and raises no warning."""
    logger = boot(internal_api_secret="s3cret")
    await deps.initialize()
    assert "gateway_auth_effective" in logger.events("info")
    assert "gateway_auth_disabled" not in logger.events("warning")


@pytest.mark.asyncio
async def test_require_gateway_secret_refuses_to_start_when_empty(boot):
    """The fail-closed switch: REQUIRE_GATEWAY_SECRET=true + empty secret -> no boot."""
    logger = boot(internal_api_secret="", require_gateway_secret=True)
    with pytest.raises(RuntimeError, match="REQUIRE_GATEWAY_SECRET"):
        await deps.initialize()
    assert "gateway_auth_misconfigured" in logger.events("critical")


@pytest.mark.asyncio
async def test_require_gateway_secret_boots_normally_when_secret_present(boot):
    """Enforcement on + secret set is the target prod state: boots, no warning."""
    logger = boot(internal_api_secret="s3cret", require_gateway_secret=True)
    state = await deps.initialize()
    assert state.config.require_gateway_secret is True
    assert logger.events("critical") == []
    assert "gateway_auth_disabled" not in logger.events("warning")
