"""Unit tests for env-gated Sentry initialization (api/sentry.py)."""

from unittest.mock import patch

import pytest

from api.sentry import init_sentry


@pytest.fixture(autouse=True)
def clean_sentry_env(monkeypatch):
    """Start every test with no Sentry-related environment."""
    for var in ("SENTRY_DSN", "SENTRY_TRACES_SAMPLE_RATE", "ENVIRONMENT"):
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
            traces_sample_rate=0.0,
            send_default_pii=False,
        )

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


class TestCreateAppWiring:
    def test_create_app_initializes_sentry(self):
        from api.app import create_app

        with patch("api.app.init_sentry") as mock_init_sentry:
            create_app()
        mock_init_sentry.assert_called_once()
