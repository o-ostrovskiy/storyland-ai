"""ai-private#78: the AI service security tune-up.

1. CORS is off unless CORS_ORIGINS names origins (no browser calls this service).
2. Every URL field the model or a tool can emit keeps only https:// URLs.
3. Logs render as JSON in production, as console text elsewhere.
(4, no model name on /health, is in test_api.py::TestHealthEndpoint.)
"""
from __future__ import annotations

import json

import pytest
import structlog
from httpx import ASGITransport, AsyncClient

# --- 1. CORS deny by default ------------------------------------------------------


async def _preflight(monkeypatch, cors_origins):
    from api.app import create_app

    if cors_origins is None:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("CORS_ORIGINS", cors_origins)
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.options(
            "/api/v1/health",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("unset", [None, "", "  "])
async def test_no_cors_origins_means_no_cors_headers(monkeypatch, unset):
    r = await _preflight(monkeypatch, unset)
    assert "access-control-allow-origin" not in r.headers


@pytest.mark.asyncio
async def test_configured_origins_keep_todays_behaviour(monkeypatch):
    r = await _preflight(monkeypatch, "https://mystoryland.ai")
    assert "access-control-allow-origin" not in r.headers  # evil.example is not listed

    from api.app import create_app

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        ok = await client.options(
            "/api/v1/health",
            headers={"Origin": "https://mystoryland.ai", "Access-Control-Request-Method": "GET"},
        )
    assert ok.headers["access-control-allow-origin"] == "https://mystoryland.ai"
    assert ok.headers.get("access-control-allow-credentials") == "true"


# --- 2. https-only model URLs -------------------------------------------------------

BAD_URLS = ["javascript:alert(1)", "data:text/html,x", "http://x", "  javascript:alert(1)", "HTTP://x", "//evil.example/x.png"]
GOOD = "https://books.google.com/cover.jpg"


def _url_models():
    from api.models import SSEMetadataEvent
    from models.book import BookInfo, BookMetadata, BookRecommendation

    return [
        (BookMetadata, {"book_title": "T", "author": "A"}),
        (BookRecommendation, {"title": "T", "author": "A", "reason": "r", "recommendation_basis": "themes"}),
        (BookInfo, {"title": "T", "authors": ["A"]}),
        (SSEMetadataEvent, {"book_title": "T", "author": "A"}),
    ]


def _build(model, base, url):
    return model(**base, image_url=url)


@pytest.mark.parametrize("bad", BAD_URLS)
def test_every_url_field_drops_a_non_https_url(bad):
    for model, base in _url_models():
        assert _build(model, base, bad).image_url is None, f"{model.__name__} kept {bad!r}"


def test_every_url_field_keeps_an_https_url_and_none():
    for model, base in _url_models():
        assert _build(model, base, GOOD).image_url == GOOD, model.__name__
        assert _build(model, base, None).image_url is None, model.__name__


def test_the_url_field_list_is_complete():
    """A new *_url field must join the https-only list (repo-wide grep, ai-private#78)."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    found = set()
    for path in [*root.glob("models/*.py"), root / "api" / "models.py"]:
        for m in re.finditer(r"^\s+([a-z_]*_url)\s*:", path.read_text(), re.M):
            found.add((path.relative_to(root).as_posix(), m.group(1)))
    assert found == {("models/book.py", "image_url"), ("api/models.py", "image_url")}


# --- 3. JSON logs in production -----------------------------------------------------


def _render(monkeypatch, capsys, environment):
    from common.logging import configure_logging

    if environment is None:
        monkeypatch.delenv("ENVIRONMENT", raising=False)
    else:
        monkeypatch.setenv("ENVIRONMENT", environment)
    configure_logging("INFO")
    capsys.readouterr()
    structlog.get_logger("t").info("probe_event", answer=42)
    return capsys.readouterr().out.strip().splitlines()[-1]


def test_production_logs_are_json(monkeypatch, capsys):
    try:
        line = _render(monkeypatch, capsys, "production")
        record = json.loads(line)
        assert record["event"] == "probe_event"
        assert record["answer"] == 42
        assert record["level"] == "info"
    finally:
        structlog.reset_defaults()


@pytest.mark.parametrize("environment", [None, "local", "staging"])
def test_other_environments_log_console_text(monkeypatch, capsys, environment):
    try:
        line = _render(monkeypatch, capsys, environment)
        assert "probe_event" in line
        with pytest.raises(json.JSONDecodeError):
            json.loads(line)
    finally:
        structlog.reset_defaults()
