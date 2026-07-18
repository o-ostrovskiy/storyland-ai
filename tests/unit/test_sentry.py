"""Unit tests for env-gated Sentry initialization (api/sentry.py)."""

from unittest.mock import patch

import pytest

from api.sentry import (
    _drop_health_probe_logs,
    _scrub_breadcrumb,
    _scrub_event,
    init_sentry,
)


@pytest.fixture(autouse=True)
def clean_sentry_env(monkeypatch):
    """Start every test with no Sentry-related environment."""
    for var in (
        "SENTRY_DSN",
        "SENTRY_TRACES_SAMPLE_RATE",
        "SENTRY_ENABLE_LOGS",
        "SENTRY_ENABLE_METRICS",
        "ENVIRONMENT",
    ):
        monkeypatch.delenv(var, raising=False)


class TestInitSentry:
    def test_disabled_without_dsn(self):
        with patch("sentry_sdk.init") as mock_init:
            assert init_sentry() is False
        mock_init.assert_not_called()

    def test_disabled_on_blank_dsn(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "   ")
        with patch("sentry_sdk.init") as mock_init:
            assert init_sentry() is False
        mock_init.assert_not_called()

    def test_enabled_with_dsn_defaults(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
        with patch("sentry_sdk.init") as mock_init:
            assert init_sentry() is True
        mock_init.assert_called_once_with(
            dsn="https://key@example.ingest.sentry.io/1",
            environment="local",
            traces_sample_rate=None,
            before_send_transaction=_scrub_event,
            send_default_pii=False,
            max_request_body_size="never",
            include_local_variables=False,
            enable_logs=True,
            before_send_log=_drop_health_probe_logs,
            before_send=_scrub_event,
            before_breadcrumb=_scrub_breadcrumb,
            enable_metrics=True,
        )

    def test_logs_and_metrics_kill_switches(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
        monkeypatch.setenv("SENTRY_ENABLE_LOGS", "false")
        monkeypatch.setenv("SENTRY_ENABLE_METRICS", "false")
        with patch("sentry_sdk.init") as mock_init:
            assert init_sentry() is True
        assert mock_init.call_args.kwargs["enable_logs"] is False
        assert mock_init.call_args.kwargs["enable_metrics"] is False

    def test_environment_and_sample_rate_from_env(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.25")
        with patch("sentry_sdk.init") as mock_init:
            assert init_sentry() is True
        kwargs = mock_init.call_args.kwargs
        assert kwargs["environment"] == "production"
        assert kwargs["traces_sample_rate"] == 0.25

    def test_malformed_sample_rate_fails_loudly(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
        monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "not-a-number")
        with patch("sentry_sdk.init") as mock_init:
            with pytest.raises(ValueError):
                init_sentry()
        mock_init.assert_not_called()


class TestStructlogSentryProcessor:
    """The structlog->Sentry bridge (common.logging._sentry_error_processor).

    structlog uses PrintLoggerFactory (stdout, not stdlib logging), so this
    processor is the ONLY path by which handled workflow failures reach
    Sentry — see Codex review on PR #203.
    """

    def test_error_event_without_exception_captures_message(self):
        from common.logging import _sentry_error_processor

        event_dict = {"event": "workflow_failed", "job_id": "abc"}
        with patch("sentry_sdk.capture_message") as mock_msg, patch(
            "sentry_sdk.capture_exception"
        ) as mock_exc:
            result = _sentry_error_processor(None, "error", dict(event_dict))
        mock_msg.assert_called_once_with("workflow_failed", level="error")
        mock_exc.assert_not_called()
        assert result["event"] == "workflow_failed"

    def test_error_event_inside_except_captures_exception(self):
        from common.logging import _sentry_error_processor

        with patch("sentry_sdk.capture_exception") as mock_exc, patch(
            "sentry_sdk.capture_message"
        ) as mock_msg:
            try:
                raise RuntimeError("boom")
            except RuntimeError:
                _sentry_error_processor(None, "error", {"event": "workflow_failed"})
        assert isinstance(mock_exc.call_args.args[0], RuntimeError)
        mock_msg.assert_not_called()

    def test_info_event_is_not_captured_as_event(self):
        from common.logging import _sentry_error_processor

        with patch("sentry_sdk.capture_message") as mock_msg, patch(
            "sentry_sdk.capture_exception"
        ) as mock_exc:
            _sentry_error_processor(None, "info", {"event": "sentry_enabled"})
        mock_msg.assert_not_called()
        mock_exc.assert_not_called()

    def test_info_event_forwarded_to_sentry_logs(self):
        from common.logging import _sentry_error_processor

        with patch("sentry_sdk.logger.info") as mock_log:
            _sentry_error_processor(
                None, "info", {"event": "discovery_started", "job_id": "j1"}
            )
        mock_log.assert_called_once_with(
            "discovery_started", attributes={"job_id": "j1"}
        )

    def test_user_content_attributes_redacted_from_logs(self):
        """Non-allowlisted keys (user content, secrets) never ship — the key
        NAMES are recorded in redacted_keys, the values dropped (Codex P1 on
        PR #204: book titles / locations / place queries / connection strings
        must not be exported by default)."""
        from common.logging import _sentry_error_processor

        with patch("sentry_sdk.logger.info") as mock_log:
            _sentry_error_processor(
                None,
                "info",
                {
                    "event": "search_started",
                    "job_id": "j1",
                    "book_title": "1984",
                    "place": "Paris",
                    "connection_string": "postgres://secret",
                },
            )
        attrs = mock_log.call_args.kwargs["attributes"]
        assert attrs["job_id"] == "j1"
        assert "book_title" not in attrs
        assert "place" not in attrs
        assert "connection_string" not in attrs
        assert attrs["redacted_keys"] == "book_title,connection_string,place"
        assert "1984" not in str(mock_log.call_args)
        assert "postgres://secret" not in str(mock_log.call_args)

    def test_warning_event_forwarded_to_sentry_logs(self):
        from common.logging import _sentry_error_processor

        with patch("sentry_sdk.logger.warning") as mock_log:
            _sentry_error_processor(None, "warning", {"event": "book_search_thin"})
        mock_log.assert_called_once_with("book_search_thin", attributes={})

    def test_debug_event_stays_local(self):
        from common.logging import _sentry_error_processor

        with patch("sentry_sdk.logger.debug") as mock_debug, patch(
            "sentry_sdk.logger.info"
        ) as mock_info, patch("sentry_sdk.metrics.count") as mock_count:
            _sentry_error_processor(None, "debug", {"event": "llm_prompt"})
        mock_debug.assert_not_called()
        mock_info.assert_not_called()
        mock_count.assert_not_called()

    @pytest.mark.parametrize("level", ["info", "warning", "error", "critical"])
    def test_non_debug_events_counted_as_metric(self, level):
        from common.logging import _sentry_error_processor

        with patch("sentry_sdk.metrics.count") as mock_count:
            _sentry_error_processor(None, level, {"event": "discover_timeout"})
        mock_count.assert_called_once_with(
            "log.events",
            1,
            attributes={"event": "discover_timeout", "level": level},
        )

    def test_processor_registered_in_structlog_config(self):
        import structlog

        from common.logging import _sentry_error_processor, configure_logging

        configure_logging(level="INFO")
        assert _sentry_error_processor in structlog.get_config()["processors"]


class TestHealthProbeLogFilter:
    def test_health_probe_access_log_dropped(self):
        log = {"body": '172.18.0.4:0 - "GET /api/v1/health HTTP/1.1" 200'}
        assert _drop_health_probe_logs(log, {}) is None

    def test_regular_log_passes_through(self):
        log = {"body": "discovery_started"}
        assert _drop_health_probe_logs(log, {}) is log

    def test_event_and_breadcrumb_query_scrub(self):
        """Parity with storyland-services#92 4th finding: events and
        breadcrumbs carry URLs the log scrub never sees."""
        event = {
            "request": {"url": "http://x/api?q=user+text", "query_string": "q=user+text"},
            "logentry": {"message": "m", "params": ["GET /y?place=Paris"], "formatted": "f"},
        }
        out = _scrub_event(event, {})
        assert out["request"]["url"] == "http://x/api"
        assert "query_string" not in out["request"]
        assert out["logentry"]["params"] == ["GET /y"]
        crumb = {"message": "GET /z?token=s", "data": {"url": "http://x/w?a=b"}}
        out2 = _scrub_breadcrumb(crumb, {})
        assert out2["message"] == "GET /z"
        assert out2["data"]["url"] == "http://x/w"
        spans_event = {
            "spans": [{"description": "GET http://u/v?book_title=secret", "data": {"url": "http://u/v?book_title=secret"}}]
        }
        out3 = _scrub_event(spans_event, {})
        assert out3["spans"][0]["description"] == "GET http://u/v"
        assert out3["spans"][0]["data"]["url"] == "http://u/v"

    def test_round5_parity(self, monkeypatch):
        """storyland-services#92 round 5: dict params + extras scrubbed,
        health crumbs dropped, tracing None when off / kept when opted in."""
        event = {
            "logentry": {"message": "%(url)s", "params": {"url": "/x?token=s"}},
            "extra": {"url": "/api?place=Paris", "count": 3},
        }
        out = _scrub_event(event, {})
        assert out["logentry"]["params"] == {"url": "/x"}
        assert out["extra"] == {"url": "/api", "count": 3}
        assert _scrub_breadcrumb(
            {"message": '"GET /api/v1/health HTTP/1.1" 200'}, {}
        ) is None
        monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
        monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.5")
        with patch("sentry_sdk.init") as mock_init:
            init_sentry()
        assert mock_init.call_args.kwargs["traces_sample_rate"] == 0.5

    def test_url_query_strings_scrubbed(self):
        """Parity with the backend (Codex on storyland-services#92): stdlib
        access records bypass the structlog allowlist and can carry request
        paths WITH query strings."""
        log = {
            "body": '10.0.0.2:0 - "GET /api/v1/thing?q=user+text HTTP/1.1" 200',
            "attributes": {"sentry.message.parameter.request_line": "GET /x?place=Paris HTTP/1.1"},
        }
        out = _drop_health_probe_logs(log, {})
        assert out is not None
        assert "q=" not in out["body"]
        assert out["attributes"]["sentry.message.parameter.request_line"] == "GET /x HTTP/1.1"

    def test_missing_body_passes_through(self):
        log = {"attributes": {}}
        assert _drop_health_probe_logs(log, {}) is log


class TestCreateAppWiring:
    def test_create_app_initializes_sentry(self):
        from api.app import create_app

        with patch("api.app.init_sentry") as mock_init_sentry:
            create_app()
        mock_init_sentry.assert_called_once()
