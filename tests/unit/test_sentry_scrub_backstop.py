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


# ── Second property (ai-private#77): no sensitive header name survives ──────
# send_default_pii=False drops only the SDK's fixed header list, and the SDK's
# EventScrubber matches whole key names, so x-internal-secret / x-user-id rode
# ai transactions in clear. Same shape as the query property: walk the WHOLE
# event, every "headers" container and every OTel-style header attribute.
_SENSITIVE_HEADER = re.compile(r"secret|token|key|auth|session|cookie", re.IGNORECASE)
_ALWAYS_DROPPED = {"x-user-id"}
_HEADER_ATTR = re.compile(r"^http\.(request|response)\.header\.(.+)$", re.IGNORECASE)


def _is_sensitive_header(name: str) -> bool:
    return name.lower() in _ALWAYS_DROPPED or bool(_SENSITIVE_HEADER.search(name))


def _header_names(container):
    if isinstance(container, dict):
        yield from (k for k in container if isinstance(k, str))
    elif isinstance(container, (list, tuple)):
        for pair in container:
            if isinstance(pair, (list, tuple)) and pair and isinstance(pair[0], str):
                yield pair[0]


def _sensitive_headers(node, path="$"):
    """Every (path, header name) with a sensitive name, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "headers":
                for name in _header_names(value):
                    if _is_sensitive_header(name):
                        yield f"{path}.headers", name
            if isinstance(key, str):
                attr = _HEADER_ATTR.match(key)
                if attr and _is_sensitive_header(attr.group(2)):
                    yield path, key
            yield from _sensitive_headers(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            yield from _sensitive_headers(value, f"{path}[{i}]")


def test_the_captured_status_request_really_carries_the_internal_headers():
    found = {name for _, name in _sensitive_headers(_fixture("transaction_status_request.json"))}
    assert {"x-internal-secret", "x-user-id"} <= found


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_no_sensitive_header_survives_the_scrubbers(name):
    scrubbed = _through_the_pipeline(_fixture(name))
    leaks = list(_sensitive_headers(scrubbed))
    assert not leaks, f"sensitive headers survived the Sentry scrubbers: {leaks}"


def test_the_scrub_keeps_harmless_headers():
    scrubbed = _through_the_pipeline(_fixture("transaction_status_request.json"))
    assert {"accept", "host", "user-agent"} <= set(scrubbed["request"]["headers"])


@pytest.mark.parametrize(
    "headers",
    [
        {"X-Internal-Secret": "s", "Authorization": "b", "X-Api-Key": "k", "Cookie": "c", "Accept": "a"},
        [["X-Internal-Secret", "s"], ["x-session-id", "i"], ["X-User-Id", "u"], ["accept", "a"]],
    ],
)
def test_header_shapes_dict_and_pair_list_are_both_scrubbed(headers):
    event = {"request": {"url": "http://x/api", "headers": headers}}
    scrubbed = _scrub_event(event, {})
    assert not list(_sensitive_headers(scrubbed))
    assert [n.lower() for n in _header_names(scrubbed["request"]["headers"])] == ["accept"]


def test_header_attributes_on_spans_and_the_root_trace_are_scrubbed():
    event = {
        "spans": [{"data": {"http.request.header.x-internal-secret": "s", "http.method": "GET"}}],
        "contexts": {"trace": {"data": {"http.request.header.authorization": "b"}}},
    }
    scrubbed = _scrub_event(event, {})
    assert not list(_sensitive_headers(scrubbed))
    assert scrubbed["spans"][0]["data"] == {"http.method": "GET"}


@pytest.mark.parametrize(
    "name,expected",
    [
        ("x-internal-secret", True),
        ("X-User-Id", True),
        ("authorization", True),
        ("x-api-key", True),
        ("cookie", True),
        ("x-session-id", True),
        ("accept", False),
        ("user-agent", False),
        ("host", False),
    ],
)
def test_sensitive_header_predicate(name, expected):
    assert _is_sensitive_header(name) is expected
