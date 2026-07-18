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
    SENTRY_ENABLE_METRICS       "true"/"false", default true — emit metrics;
                                every structlog event is auto-counted as
                                `log.events` {event, level}, so timeouts,
                                cache hits, and workflow failures are
                                chartable without per-site instrumentation
    SENTRY_TRACES_SAMPLE_RATE   0.0-1.0, default 0.0 (errors only)
    ENVIRONMENT                 reused as the Sentry environment tag
    SENTRY_RELEASE              release tag (read by the SDK itself); the
                                deploy tooling sets it to the deployed git
                                SHA so Sentry can pin "first seen in
                                release X" and auto-resolve on new deploys

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

Metrics examples (for NEW call sites; existing log events are counted
automatically as `log.events` by the structlog bridge):

    from sentry_sdk import metrics

    # counter — how often something happens
    metrics.count("discovery.cache.hit", 1, attributes={"backend": "disk"})

    # distribution — spread of a measured value (p50/p95 in Sentry)
    metrics.distribution("workflow.discover.duration", elapsed_s, unit="second")

    # gauge — current level of something
    metrics.gauge("sessions.active", n)

Stdlib loggers (google-adk, urllib3, ...) flow via the SDK's logging
integration automatically; the structlog bridge in common.logging covers
our own loggers, which bypass stdlib entirely (PrintLoggerFactory).
"""

import os
import re

from common.logging import get_logger

logger = get_logger(__name__)


# Everything from '?' up to whitespace or a closing quote — URL query strings
# in access lines and any other URL-bearing stdlib log reaching Sentry Logs.
_URL_QUERY_RE = re.compile(r"\?[^\s\"]*")


def _drop_health_probe_logs(log, hint):
    """
    before_send_log filter, two jobs (parity with the backend gateway —
    Codex on storyland-services#92):

    1. DROP the docker healthcheck's uvicorn access-log lines
       ('GET /api/v1/health ... 200') — every 30s, pure noise; the
       healthcheck's FAILURE signal is the container going unhealthy.
    2. SCRUB URL query strings from bodies and string attributes: stdlib
       records (uvicorn.access etc.) bypass the structlog allowlist and can
       carry request paths WITH query. Fail-safe: any '?' tail is stripped.
    """
    body = log.get("body") or ""
    if "/api/v1/health" in body:
        return None
    if "?" in body:
        log["body"] = _URL_QUERY_RE.sub("", body)
    attributes = log.get("attributes") or {}
    for key, value in list(attributes.items()):
        if isinstance(value, str) and "?" in value:
            attributes[key] = _URL_QUERY_RE.sub("", value)
    return log


def _scrub_event(event, hint):
    """
    before_send: strip URL query strings from error EVENTS (request.url,
    stdlib logentry message/params) — these bypass the log-path scrub.
    Parity with the backend gateway (storyland-services#92).
    """
    request = event.get("request")
    if request:
        url = request.get("url")
        if isinstance(url, str) and "?" in url:
            request["url"] = _URL_QUERY_RE.sub("", url)
        request.pop("query_string", None)
    logentry = event.get("logentry")
    if logentry:
        for key in ("message", "formatted"):
            value = logentry.get(key)
            if isinstance(value, str) and "?" in value:
                logentry[key] = _URL_QUERY_RE.sub("", value)
        params = logentry.get("params")
        if isinstance(params, (list, tuple)):
            logentry["params"] = [
                _URL_QUERY_RE.sub("", p) if isinstance(p, str) and "?" in p else p
                for p in params
            ]
        elif isinstance(params, dict):
            # Mapping-style formatting: logger.error("%(url)s", {"url": …})
            for key, value in list(params.items()):
                if isinstance(value, str) and "?" in value:
                    params[key] = _URL_QUERY_RE.sub("", value)
    # stdlib records with extra={"url": …} land on event["extra"].
    extra = event.get("extra")
    if isinstance(extra, dict):
        for key, value in list(extra.items()):
            if isinstance(value, str) and "?" in value:
                extra[key] = _URL_QUERY_RE.sub("", value)
    # Transaction spans (before_send_transaction path): HTTP-client spans
    # record outbound URLs with query strings in description/data.
    spans = event.get("spans")
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, dict):
                continue
            description = span.get("description")
            if isinstance(description, str) and "?" in description:
                span["description"] = _URL_QUERY_RE.sub("", description)
            data = span.get("data")
            if isinstance(data, dict):
                for key, value in list(data.items()):
                    if isinstance(value, str) and "?" in value:
                        data[key] = _URL_QUERY_RE.sub("", value)
    return event


def _scrub_breadcrumb(crumb, hint):
    """
    before_breadcrumb: scrub stdlib-record crumbs (message + string data);
    DROP health-probe lines entirely — at one every 30s they can fill the
    100-crumb buffer in a quiet container and evict real pre-error context.
    """
    message = crumb.get("message")
    if isinstance(message, str):
        if "/api/v1/health" in message:
            return None
        if "?" in message:
            crumb["message"] = _URL_QUERY_RE.sub("", message)
    data = crumb.get("data") or {}
    for key, value in list(data.items()):
        if isinstance(value, str) and "?" in value:
            data[key] = _URL_QUERY_RE.sub("", value)
    return crumb


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
    # tiny and searchable prod logs are the point. Kill switches, not opt-ins.
    enable_logs = (os.getenv("SENTRY_ENABLE_LOGS") or "true").lower() == "true"
    enable_metrics = (os.getenv("SENTRY_ENABLE_METRICS") or "true").lower() == "true"

    # Imported lazily so the module can be imported (and the disabled path
    # unit-tested) even if the SDK were ever absent from a local env.
    import sentry_sdk

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        # None (not 0.0) when tracing is off: with 0.0 the SDK still honors
        # inbound sentry-trace headers and can emit transactions, which skip
        # every error/log scrubber. before_send_transaction covers opt-in.
        traces_sample_rate=traces_sample_rate if traces_sample_rate > 0 else None,
        before_send_transaction=_scrub_event,
        # No request headers/IPs/cookies on events. User prompts already live
        # in Langfuse traces under access control; Sentry only needs the error.
        send_default_pii=False,
        # send_default_pii=False does NOT cover request bodies — the SDK
        # default ("medium") uploads small failing-request bodies, which here
        # carry book titles, taste context, and local-atmosphere lat/lng.
        max_request_body_size="never",
        # Nor does it cover failing-frame LOCALS (SDK default True) — which
        # here hold user prompts, taste context, and request models. Same
        # hardening as the backend gateway (Codex on storyland-services#92).
        include_local_variables=False,
        # Sentry Logs: stdlib INFO+ records ship automatically; our structlog
        # events ship via the bridge in common.logging (structlog bypasses
        # stdlib, so without the bridge nothing of ours would appear).
        enable_logs=enable_logs,
        before_send_log=_drop_health_probe_logs,
        before_send=_scrub_event,
        before_breadcrumb=_scrub_breadcrumb,
        # Metrics: every structlog event is auto-counted (`log.events`) by the
        # bridge in common.logging; new call sites can use sentry_sdk.metrics
        # directly (examples in the module docstring).
        enable_metrics=enable_metrics,
    )
    logger.info(
        "sentry_enabled",
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        logs_enabled=enable_logs,
        metrics_enabled=enable_metrics,
    )
    return True
