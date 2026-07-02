"""Unit tests for the discovery-compose typed-error boundary (MYS-124 PR1).

Covers three things:
  1. ``classify_discovery_failure`` — collapses an async-TaskGroup
     ``ExceptionGroup`` into a single, client-safe ``DiscoveryComposeError``
     classified ``transient`` vs ``taste_validation`` (with ``offending_title``),
     and never leaks the raw exception / "TaskGroup" / "ExceptionGroup" string.
  2. The wire contract — ``WorkflowError`` / ``SSEErrorEvent`` carry the optional
     ``reason`` + ``offending_title`` the gateway (be) and fe consume.
  3. ``WorkflowExecutor.discover()`` surfaces the typed error instead of a raw
     ExceptionGroup when a discovery child task crashes.
"""

import json

import pytest

from google.adk.events import Event
from google.adk.events.event_actions import EventActions

from core.discovery_errors import (
    DiscoveryComposeError,
    TasteContextValidationError,
    classify_discovery_failure,
    TRANSIENT_MESSAGE,
    TASTE_VALIDATION_MESSAGE,
)
from core.events import RegionsReady, WorkflowError, WorkflowComplete
from core.types import ExecutorConfig
from core.session_state import SessionStateKeys

# The opaque strings that must NEVER reach a client.
_LEAK_MARKERS = ("TaskGroup", "ExceptionGroup", "sub-exception", "Traceback")


def _assert_no_leak(text: str) -> None:
    for marker in _LEAK_MARKERS:
        assert marker not in text, f"leaked internal marker {marker!r}: {text!r}"


class TestClassifyDiscoveryFailure:
    def test_bare_exception_is_transient(self):
        err = classify_discovery_failure(RuntimeError("boom: upstream timed out"))
        assert isinstance(err, DiscoveryComposeError)
        assert err.kind == "transient"
        assert err.offending_title is None
        assert err.message == TRANSIENT_MESSAGE
        _assert_no_leak(err.message)

    def test_taskgroup_child_raise_is_transient_no_leak(self):
        # Simulate the exact reported failure: a child task crashes inside an
        # asyncio.TaskGroup, surfacing as an ExceptionGroup whose str() is the
        # opaque "unhandled errors in a TaskGroup (1 sub-exception)".
        group = ExceptionGroup(
            "unhandled errors in a TaskGroup", [ValueError("child crashed")]
        )
        assert "TaskGroup" in str(group)  # precondition: the raw leak exists
        err = classify_discovery_failure(group)
        assert err.kind == "transient"
        _assert_no_leak(err.message)
        _assert_no_leak(str(err))

    def test_taste_validation_leaf_direct(self):
        leaf = TasteContextValidationError(offending_title="Ignore all previous…")
        err = classify_discovery_failure(leaf)
        assert err.kind == "taste_validation"
        assert err.offending_title == "Ignore all previous…"
        assert err.message == TASTE_VALIDATION_MESSAGE
        _assert_no_leak(err.message)

    def test_taste_validation_wrapped_in_group(self):
        group = ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [TasteContextValidationError(offending_title="Bad Title")],
        )
        err = classify_discovery_failure(group)
        assert err.kind == "taste_validation"
        assert err.offending_title == "Bad Title"

    def test_offending_title_falls_back_to_taste_context(self):
        # Leaf recorded no title -> fall back to the first taste_context title.
        leaf = TasteContextValidationError(offending_title=None)
        err = classify_discovery_failure(
            leaf, taste_context={"titles": ["First Saved Book", "Second"]}
        )
        assert err.kind == "taste_validation"
        assert err.offending_title == "First Saved Book"

    def test_nested_group_unwraps(self):
        inner = ExceptionGroup("inner", [TasteContextValidationError("Nested Bad")])
        outer = ExceptionGroup("outer", [inner])
        err = classify_discovery_failure(outer)
        assert err.kind == "taste_validation"
        assert err.offending_title == "Nested Bad"

    def test_mixed_group_prefers_taste_validation(self):
        group = ExceptionGroup(
            "mixed",
            [RuntimeError("transient bit"), TasteContextValidationError("Poison")],
        )
        err = classify_discovery_failure(group)
        assert err.kind == "taste_validation"
        assert err.offending_title == "Poison"

    def test_passthrough_existing_compose_error(self):
        original = DiscoveryComposeError("transient", "already typed")
        assert classify_discovery_failure(original) is original

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError):
            DiscoveryComposeError("weird", "nope")


class TestWireContract:
    def test_workflow_error_defaults_are_backward_compatible(self):
        err = WorkflowError(message="x", error_type="NoRegions")
        assert err.reason is None
        assert err.offending_title is None

    def test_workflow_error_carries_reason_and_title(self):
        err = WorkflowError(
            message="friendly",
            error_type="DiscoveryComposeError",
            reason="taste_validation",
            offending_title="Bad Book",
        )
        assert err.reason == "taste_validation"
        assert err.offending_title == "Bad Book"

    def test_sse_error_event_serializes_contract_fields(self):
        from api.models import SSEErrorEvent

        payload = json.loads(
            SSEErrorEvent(
                message="friendly",
                error_type="DiscoveryComposeError",
                phase=2,
                reason="taste_validation",
                offending_title="Bad Book",
            ).model_dump_json()
        )
        assert payload["reason"] == "taste_validation"
        assert payload["offending_title"] == "Bad Book"
        _assert_no_leak(json.dumps(payload))


def _make_executor(monkeypatch, raise_exc):
    """WorkflowExecutor whose stubbed discovery run raises ``raise_exc``."""
    import core.executor as ex
    from core.executor import WorkflowExecutor
    from services.session_service import create_session_service

    monkeypatch.setattr(ex, "create_discovery_workflow", lambda *a, **k: object())

    class _RaisingRunner:
        def __init__(self, *args, **kwargs):
            self._session_service = kwargs.get("session_service")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run_async(self, user_id, session_id, new_message):
            raise raise_exc
            if False:  # keep this an async generator
                yield None

    monkeypatch.setattr(ex, "Runner", _RaisingRunner)

    config = ExecutorConfig(model_name="gemini-2.0-flash", google_api_key="test-key")
    return WorkflowExecutor(
        config=config,
        session_service=create_session_service(use_database=False),
        model=object(),
    )


class TestDiscoverSurfacesTypedError:
    async def test_taskgroup_crash_surfaces_transient_not_raw_group(
        self, monkeypatch
    ):
        group = ExceptionGroup(
            "unhandled errors in a TaskGroup", [RuntimeError("child crashed")]
        )
        executor = _make_executor(monkeypatch, group)

        events = [
            e async for e in executor.discover(book_title="1984", author="George Orwell")
        ]

        errors = [e for e in events if isinstance(e, WorkflowError)]
        assert len(errors) == 1
        err = errors[0]
        assert err.error_type == "DiscoveryComposeError"
        assert err.reason == "transient"
        _assert_no_leak(err.message)
        assert not any(isinstance(e, RegionsReady) for e in events)
        assert any(isinstance(e, WorkflowComplete) for e in events)

    async def test_taste_validation_crash_surfaces_reason_and_title(
        self, monkeypatch
    ):
        group = ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [TasteContextValidationError(offending_title="Poison Title")],
        )
        executor = _make_executor(monkeypatch, group)

        events = [
            e
            async for e in executor.discover(
                book_title="1984",
                author="George Orwell",
                taste_context={"titles": ["Poison Title"]},
            )
        ]

        errors = [e for e in events if isinstance(e, WorkflowError)]
        assert len(errors) == 1
        assert errors[0].reason == "taste_validation"
        assert errors[0].offending_title == "Poison Title"
        _assert_no_leak(errors[0].message)
