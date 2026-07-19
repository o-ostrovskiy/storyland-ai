"""Unit tests for core/run_harness.py — the shared runner scaffolding.

These pin the contract the executor flows (and the ADK 2 migration) rely on:
pump_events' progress/capture semantics and run_guarded's per-flow
timeout/cancel/exception policy, including cleanup ordering.
"""

import asyncio
from types import SimpleNamespace

import pytest

from core.events import (
    Phase,
    ProgressEvent,
    WorkflowComplete,
    WorkflowError,
)
from core.run_harness import (
    GuardSpec,
    RunCapture,
    collect_token_usage,
    error_events,
    pump_events,
    run_guarded,
)


def _adk_event(author, texts=(), final=False):
    parts = [SimpleNamespace(text=t) for t in texts]
    content = SimpleNamespace(parts=parts) if parts else None
    return SimpleNamespace(
        author=author,
        content=content,
        is_final_response=lambda: final,
    )


class _FakeRunner:
    """Minimal async-context runner yielding a fixed event sequence."""

    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run_async(self, *, user_id, session_id, new_message):
        for ev in self._events:
            yield ev


def _message():
    return SimpleNamespace(role="user")


class TestPumpEvents:
    async def test_one_progress_event_per_agent(self):
        runner = _FakeRunner(
            [
                _adk_event("city_researcher"),
                _adk_event("city_researcher"),  # repeat: must not re-report
                _adk_event("unknown_agent"),  # unmapped: no progress event
                _adk_event("city_formatter"),
            ]
        )
        steps = {"city_researcher": "Finding cities", "city_formatter": "Formatting"}
        out = [
            ev
            async for ev in pump_events(
                runner,
                user_id="u",
                session_id="s",
                message=_message(),
                phase=Phase.DISCOVERY,
                agent_steps=steps,
            )
        ]
        assert out == [
            ProgressEvent(phase=Phase.DISCOVERY, step="Finding cities", detail="city_researcher"),
            ProgressEvent(phase=Phase.DISCOVERY, step="Formatting", detail="city_formatter"),
        ]

    async def test_captures_text_only_for_requested_authors(self):
        runner = _FakeRunner(
            [
                _adk_event("researcher", texts=["alpha", "beta"]),
                _adk_event("other", texts=["noise"]),
                _adk_event("researcher", texts=["gamma"]),
            ]
        )
        capture = RunCapture()
        async for _ in pump_events(
            runner,
            user_id="u",
            session_id="s",
            message=_message(),
            phase=Phase.COMPOSITION,
            agent_steps={},
            capture=capture,
            capture_authors=("researcher",),
        ):
            pass
        assert capture.text_for("researcher") == "alpha\nbeta\ngamma"
        assert capture.text_for("other") == ""

    async def test_tracks_last_final_response(self):
        first_final = _adk_event("composer", final=True)
        last_final = _adk_event("composer", final=True)
        runner = _FakeRunner([_adk_event("composer"), first_final, last_final])
        capture = RunCapture()
        async for _ in pump_events(
            runner,
            user_id="u",
            session_id="s",
            message=_message(),
            phase=Phase.COMPOSITION,
            agent_steps={},
            capture=capture,
            track_final_response=True,
        ):
            pass
        assert capture.final_response is last_final

    async def test_no_capture_object_means_no_tracking(self):
        runner = _FakeRunner([_adk_event("a", texts=["t"], final=True)])
        # Must not raise despite capture_authors/track flags being unset.
        out = [
            ev
            async for ev in pump_events(
                runner,
                user_id="u",
                session_id="s",
                message=_message(),
                phase=Phase.DISCOVERY,
                agent_steps={},
            )
        ]
        assert out == []


class TestRunGuarded:
    def _spec(self, **overrides):
        defaults = dict(
            flow_name="testflow",
            job_id="job-1",
            timeout_seconds=5,
            timeout_message="Timed out after 5s",
            phase=Phase.COMPOSITION,
        )
        defaults.update(overrides)
        return GuardSpec(**defaults)

    async def test_passthrough_on_success(self):
        async def body():
            yield ProgressEvent(phase=Phase.COMPOSITION, step="one")
            yield WorkflowComplete(job_id="job-1")

        out = [ev async for ev in run_guarded(body(), self._spec())]
        assert [type(e).__name__ for e in out] == ["ProgressEvent", "WorkflowComplete"]

    async def test_timeout_runs_cleanup_then_error_pair(self):
        calls = []

        async def cleanup():
            calls.append("cleanup")

        async def body():
            await asyncio.sleep(60)
            yield ProgressEvent(phase=Phase.COMPOSITION, step="never")

        spec = self._spec(timeout_seconds=0.01, on_timeout=cleanup)
        out = [ev async for ev in run_guarded(body(), spec)]
        assert calls == ["cleanup"]
        assert isinstance(out[-2], WorkflowError)
        assert out[-2].error_type == "WorkflowTimeoutError"
        assert out[-2].message == "Timed out after 5s"
        assert isinstance(out[-1], WorkflowComplete)

    async def test_cancel_runs_cleanup_and_reraises(self):
        calls = []

        async def cleanup():
            calls.append("cancel-cleanup")

        async def body():
            raise asyncio.CancelledError()
            yield  # pragma: no cover

        spec = self._spec(on_cancel=cleanup)
        with pytest.raises(asyncio.CancelledError):
            async for _ in run_guarded(body(), spec):
                pass
        assert calls == ["cancel-cleanup"]

    async def test_generic_exception_default_mapping(self):
        async def body():
            raise ValueError("boom")
            yield  # pragma: no cover

        out = [ev async for ev in run_guarded(body(), self._spec())]
        assert isinstance(out[0], WorkflowError)
        assert out[0].message == "boom"
        assert out[0].error_type == "ValueError"
        assert out[0].phase == Phase.COMPOSITION
        assert isinstance(out[1], WorkflowComplete)

    async def test_generic_exception_custom_mapper_and_cleanup(self):
        calls = []

        async def cleanup():
            calls.append("error-cleanup")

        def mapper(e):
            calls.append(f"mapped:{type(e).__name__}")
            return WorkflowError(
                message="client-safe",
                error_type="DiscoveryComposeError",
                phase=Phase.DISCOVERY,
                reason="transient",
            )

        async def body():
            raise RuntimeError("internal detail")
            yield  # pragma: no cover

        spec = self._spec(on_error=cleanup, map_exception=mapper)
        out = [ev async for ev in run_guarded(body(), spec)]
        # Mapper (which owns logging/classification) runs before cleanup,
        # mirroring the historical classify -> log -> mark_failed order.
        assert calls == ["mapped:RuntimeError", "error-cleanup"]
        assert out[0].message == "client-safe"
        assert out[0].reason == "transient"

    async def test_exception_mid_stream_after_events(self):
        async def body():
            yield ProgressEvent(phase=Phase.COMPOSITION, step="one")
            raise ValueError("late failure")

        out = [ev async for ev in run_guarded(body(), self._spec())]
        assert [type(e).__name__ for e in out] == [
            "ProgressEvent",
            "WorkflowError",
            "WorkflowComplete",
        ]


class TestHelpers:
    async def test_collect_token_usage_enabled(self):
        flushed = []

        class Plugin:
            enabled = True

            def get_session_stats(self):
                return {"total_tokens": 42}

            async def flush(self):
                flushed.append(True)

        assert await collect_token_usage(Plugin()) == {"total_tokens": 42}
        assert flushed == [True]

    async def test_collect_token_usage_disabled(self):
        class Plugin:
            enabled = False

        assert await collect_token_usage(Plugin()) is None

    def test_error_events_pair(self):
        err, done = error_events("j1", "msg", "SomeError", phase=Phase.DISCOVERY)
        assert isinstance(err, WorkflowError)
        assert err.message == "msg"
        assert err.error_type == "SomeError"
        assert err.phase == Phase.DISCOVERY
        assert isinstance(done, WorkflowComplete)
        assert done.job_id == "j1"

    def test_error_events_extra_fields(self):
        err, _ = error_events(
            "j1", "msg", "DiscoveryComposeError", reason="taste_validation",
            offending_title="Some Book",
        )
        assert err.reason == "taste_validation"
        assert err.offending_title == "Some Book"
