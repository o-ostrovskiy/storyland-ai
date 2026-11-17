"""Backstop: no query string survives the Sentry scrubbers (ai-private#74).

The per-key rules in api/sentry.py each name a place a URL query can hide. Four
times, a place nobody named leaked (storyland-team-os#18). This test asserts the
PROPERTY instead of the list: after the scrubbers run, no string value
ANYWHERE in the serialized event looks like a query. It sits behind the per-key
tests in tests/unit/test_sentry.py and replaces none of them.

- The walk is recursive over every dict and list, so a field a future
  sentry-sdk adds is checked because it exists, not because someone listed it.
- The events are CAPTURED from the real SDK, the real app and the real
  google-genai client by scripts/capture_sentry_fixtures.py (a synthetic local
  run, Gemini stubbed in-process, no reader data), never hand-built: a
  hand-built dict only holds fields we already knew about.
- The pipeline is the one init_sentry() wires: before_breadcrumb on every
  breadcrumb (the SDK runs it when the crumb is recorded), then before_send
  for an error event or before_send_transaction for a transaction.
"""

import copy
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from api.sentry import _scrub_breadcrumb, _scrub_event, init_sentry

FIXTURES = Path(__file__).parent.parent / "fixtures" / "sentry"

# "Looks like a query": a '?' or '#' tail (the mark followed directly by
# something, so a code comment's "# note" or a sentence's "why?" is not one),
# or a bare k=v pair shape at the start of the value.
_QUERY_TAIL = re.compile(r"[?#][^\s?#]")
_BARE_PAIR = re.compile(r"^[\w.\-\[\]]+=")


def _looks_like_query(value: str) -> bool:
    return bool(_QUERY_TAIL.search(value) or _BARE_PAIR.search(value))


def _query_like_strings(node, path="$"):
    """Every (path, value) whose string value looks like a query, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _query_like_strings(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            yield from _query_like_strings(value, f"{path}[{i}]")
    elif isinstance(node, str) and _looks_like_query(node):
        yield path, node


def _through_the_pipeline(raw: dict) -> dict:
    """Run a raw captured event through the hooks init_sentry() installs."""
    event = copy.deepcopy(raw)
    crumbs = (event.get("breadcrumbs") or {}).get("values")
    if isinstance(crumbs, list):
        kept = [_scrub_breadcrumb(crumb, {}) for crumb in crumbs]
        event["breadcrumbs"]["values"] = [c for c in kept if c is not None]
    return _scrub_event(event, {})


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


FIXTURE_NAMES = [
    "transaction_status_request.json",
    "transaction_gemini_fetch.json",
    "error_after_gemini_call.json",
]


def test_the_pipeline_here_is_the_one_init_sentry_wires(monkeypatch):
    """If init_sentry() swaps a hook, _through_the_pipeline must follow."""
    monkeypatch.setenv("SENTRY_DSN", "https://public@o0.ingest.sentry.io/0")
    with patch("sentry_sdk.init") as sdk_init:
        init_sentry()
    kwargs = sdk_init.call_args.kwargs
    assert kwargs["before_breadcrumb"] is _scrub_breadcrumb
    assert kwargs["before_send"] is _scrub_event
    assert kwargs["before_send_transaction"] is _scrub_event


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_captured_event_really_carries_queries(name):
    """Control: a fixture with nothing to scrub would make the backstop vacuous."""
    assert list(_query_like_strings(_fixture(name))), f"{name} has no query-like value"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_no_query_survives_the_scrubbers(name):
    scrubbed = _through_the_pipeline(_fixture(name))
    leaks = list(_query_like_strings(scrubbed))
    assert not leaks, "query-like values survived the Sentry scrubbers:\n" + "\n".join(
        f"  {path} = {value[:100]!r}" for path, value in leaks
    )


@pytest.mark.parametrize(
    "value, expected",
    [
        ("book_title=Synthetic+Novel", True),
        ("alt=sse", True),
        ("http.query", False),
        ("POST /v1beta/models/x:streamGenerateContent?alt=sse", True),
        ("https://example.com/page#section", True),
        ("        # a code comment", False),
        ("is that a question?", False),
        ("/api/v1/itinerary/{job_id}/status", False),
        ("x == y", False),
    ],
)
def test_query_predicate(value, expected):
    assert _looks_like_query(value) is expected
