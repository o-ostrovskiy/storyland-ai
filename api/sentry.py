"""
Env-gated Sentry initialization for the API service.

Error tracking only by default: unhandled exceptions become Sentry events via
the FastAPI integration, and handled workflow failures reach Sentry through
the structlog processor in common.logging (structlog prints straight to
stdout here, so the SDK's stdlib logging integration alone would never see
them; stdlib loggers — ADK, google libs — still feed breadcrumbs/events the
normal way). Request bodies are never attached (max_request_body_size:
send_default_pii=False does not cover them, and ours carry book titles,
taste context, and lat/lng). Performance tracing is OFF by default —
agent runs are already traced in Langfuse, so Sentry transactions would
duplicate that spend; opt in per-environment via SENTRY_TRACES_SAMPLE_RATE.

The DSN is deliberately NOT in this (public) repo or in config defaults:
set SENTRY_DSN in the environment (.env.prod on the box, plus the service's
`environment:` block in the deploy compose file — a var absent from that
block never reaches the container). No DSN means Sentry stays fully off,
so local dev and CI never report.

Environment variables:
    SENTRY_DSN                  enable switch; empty/absent = disabled
    SENTRY_ENABLE_LOGS          "true"/"false", default true — ship INFO+
                                logs to Sentry Logs (searchable, alertable)
    SENTRY_TRACES_SAMPLE_RATE   0.0-1.0, default 0.0 (errors only)
    ENVIRONMENT                 reused as the Sentry environment tag
    SENTRY_RELEASE              optional release tag (read by the SDK itself)

Logging examples (what reaches Sentry, and how):

    from common.logging import get_logger
    logger = get_logger(__name__)

    # -> Sentry Logs entry (level info), with job_id/phase as attributes
    logger.info("discovery_started", job_id=job_id, phase="discovery")

    # -> Sentry Logs entry (level warn)
    logger.warning("book_search_thin", results=1, floor=3)

    # -> Sentry EVENT (alerting): inside an except block the live
    #    exception + traceback is captured, structlog kvs attached
    logger.error("workflow_failed", job_id=job_id, phase=phase)

    # debug stays local-only by design (LLM prompts can appear there)

Stdlib loggers (google-adk, urllib3, ...) flow via the SDK's logging
integration automatically; the structlog bridge in common.logging covers
our own loggers, which bypass stdlib entirely (PrintLoggerFactory).
"""

import os

from common.logging import get_logger

logger = get_logger(__name__)


def init_sentry() -> bool:
    """
    Initialize the Sentry SDK if SENTRY_DSN is set.

    Returns:
        True if Sentry was initialized, False if disabled (no DSN).

    Raises:
        ValueError: on a malformed SENTRY_TRACES_SAMPLE_RATE — fail loudly
            at boot like the rest of config loading, never silently sample.
    """
    dsn = (os.getenv("SENTRY_DSN") or "").strip()
    if not dsn:
        logger.info("sentry_disabled", reason="SENTRY_DSN not set")
        return False

    traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE") or "0.0")
    environment = os.getenv("ENVIRONMENT", "local")
    # Default ON when Sentry itself is on: log volume at current traffic is
    # tiny and searchable prod logs are the point. Kill switch, not opt-in.
    enable_logs = (os.getenv("SENTRY_ENABLE_LOGS") or "true").lower() == "true"

    # Imported lazily so the module can be imported (and the disabled path
    # unit-tested) even if the SDK were ever absent from a local env.
    import sentry_sdk

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        # No request headers/IPs/cookies on events. User prompts already live
        # in Langfuse traces under access control; Sentry only needs the error.
        send_default_pii=False,
        # send_default_pii=False does NOT cover request bodies — the SDK
        # default ("medium") uploads small failing-request bodies, which here
        # carry book titles, taste context, and local-atmosphere lat/lng.
        max_request_body_size="never",
        # Sentry Logs: stdlib INFO+ records ship automatically; our structlog
        # events ship via the bridge in common.logging (structlog bypasses
        # stdlib, so without the bridge nothing of ours would appear).
        enable_logs=enable_logs,
    )
    logger.info(
        "sentry_enabled",
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        logs_enabled=enable_logs,
    )
    return True
