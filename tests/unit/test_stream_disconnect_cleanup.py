"""
A reader leaving mid-stream: the cleanup finishes, nothing is an error, and
the departure is counted with where it happened.

sse_starlette cancels its anyio task group when the client disconnects, and
anyio re-delivers that cancel at EVERY later ``await`` in the scope. ai#305
made each session DB call survive that (its own shielded task), so the pooled
connection comes back. But ``run_guarded``'s ``on_cancel`` cleanup still ran
inside the cancelled scope and died at its first ``await``: a cancelled
discover was never marked failed, and a dropped expand left
EXPANSION_IN_PROGRESS set, so every later expand on that job was refused.

The discover test drives the real chain: ``EventSourceResponse`` ->
``discover_stream`` -> ``WorkflowExecutor.discover`` -> the real
``CancellationSafeDatabaseSessionService`` on a temp SQLite file, with a fake
runner that blocks mid-discovery (no model call).
"""

import asyncio
import logging

import anyio
import pytest
import structlog
from google.adk.workflow import Workflow
from sse_starlette.sse import EventSourceResponse

import core.run_harness as harness
from api.streaming import discover_stream, track_abandonment
from core.events import (
    JobStarted,
    Phase,
    ProgressEvent,
    RegionsReady,
    WorkflowComplete,
)
from core.executor import APP_NAME
from core.run_harness import GuardSpec, run_guarded
from core.session_state import SessionStateKeys
from core.types import ExecutorConfig

USER = "u1"


def _spec(on_cancel=None, timeout=60.0):
    return GuardSpec(
        flow_name="discover",
        job_id="job-1",
        timeout_seconds=timeout,
        timeout_message="timed out",
        phase=Phase.DISCOVERY,
        on_cancel=on_cancel,
    )


async def _cancel_in_task_group(consume) -> None:
    """Run ``consume`` in an anyio task group, then cancel the group's scope:
    the same level-triggered cancel sse_starlette delivers on a disconnect."""
    started = anyio.Event()

    async with anyio.create_task_group() as tg:
        tg.start_soon(consume, started)
        await started.wait()
        await asyncio.sleep(0.01)
        tg.cancel_scope.cancel()


# --- run_guarded: on_cancel survives the anyio cancel ----------------------


async def test_on_cancel_completes_inside_a_cancelled_anyio_scope():
    """The cleanup awaits several times; each await would be re-cancelled
    without the shield. It must still run to the end."""
    done = []

    async def cleanup():
        for _ in range(3):
            await asyncio.sleep(0)
        done.append(True)

    async def body():
        yield JobStarted(job_id="job-1")
        await asyncio.Event().wait()  # mid-discovery, waiting on the model

    async def consume(started):
        async for _ in run_guarded(body(), _spec(on_cancel=cleanup)):
            started.set()

    await _cancel_in_task_group(consume)

    assert done == [True]


async def test_cancel_still_propagates_after_the_cleanup():
    """The shield protects the cleanup, never the caller."""
    async def body():
        yield JobStarted(job_id="job-1")
        await asyncio.Event().wait()

    async def noop():
        return None

    async def consume():
        async for _ in run_guarded(body(), _spec(on_cancel=noop)):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_on_cancel_skipped_once_the_result_went_out():
    """The MCP server closes discover right after `regions`: the job is
    delivered and must not be marked failed by the cut-off tail."""
    calls = []

    async def cleanup():
        calls.append(True)

    async def body():
        yield RegionsReady(job_id="job-1", regions=[], analysis_note="")
        await asyncio.Event().wait()  # the Langfuse flush before `done`

    async def consume(started):
        async for _ in run_guarded(body(), _spec(on_cancel=cleanup)):
            started.set()

    await _cancel_in_task_group(consume)

    assert calls == []


async def test_on_cancel_skipped_when_closed_at_the_result():
    """Codex on ai#321: receive the result with anext(), then aclose() --
    the generator never resumes past that yield. The job was delivered, so
    on_cancel must not run (the test above asks for the next event first)."""
    calls = []

    async def cleanup():
        calls.append(True)

    async def body():
        yield JobStarted(job_id="job-1")
        yield RegionsReady(job_id="job-1", regions=[], analysis_note="")
        await asyncio.Event().wait()

    gen = run_guarded(body(), _spec(on_cancel=cleanup))
    assert isinstance(await anext(gen), JobStarted)
    assert isinstance(await anext(gen), RegionsReady)
    await gen.aclose()

    assert calls == []


async def test_on_cancel_runs_when_closed_before_the_result():
    """Control for the test above: a close before the result still cleans up."""
    calls = []

    async def cleanup():
        calls.append(True)

    async def body():
        yield JobStarted(job_id="job-1")
        yield RegionsReady(job_id="job-1", regions=[], analysis_note="")

    gen = run_guarded(body(), _spec(on_cancel=cleanup))
    assert isinstance(await anext(gen), JobStarted)
    await gen.aclose()

    assert calls == [True]


async def test_on_cancel_is_bounded(monkeypatch):
    """A stuck cleanup gives up at the deadline instead of hanging the request."""
    monkeypatch.setattr(harness, "CANCEL_CLEANUP_TIMEOUT_S", 0.05)

    async def stuck():
        await asyncio.Event().wait()

    async def body():
        yield JobStarted(job_id="job-1")
        await asyncio.Event().wait()

    async def consume(started):
        async for _ in run_guarded(body(), _spec(on_cancel=stuck)):
            started.set()

    with structlog.testing.capture_logs() as logs:
        with anyio.fail_after(2):
            await _cancel_in_task_group(consume)

    assert any(e["event"] == "discover_cancel_cleanup_timeout" for e in logs)


# --- track_abandonment: the countable departure ----------------------------


async def test_abandonment_logged_with_where_the_reader_left():
    async def events():
        yield JobStarted(job_id="abcdef123456")
        yield ProgressEvent(phase=Phase.DISCOVERY, step="Starting location discovery")
        await asyncio.Event().wait()

    async def consume(started):
        async for _ in track_abandonment("discover", events()):
            started.set()

    with structlog.testing.capture_logs() as logs:
        await _cancel_in_task_group(consume)

    [entry] = [e for e in logs if e["event"] == "sse_stream_abandoned"]
    assert entry["log_level"] == "info"
    assert entry["flow"] == "discover"
    assert entry["reason"] == "client_closed"
    assert entry["last_event"] == "progress"
    assert entry["phase"] == int(Phase.DISCOVERY)
    assert entry["result_sent"] is False
    assert entry["events_sent"] == 2
    assert entry["job_id"] == "abcdef12"


async def test_abandonment_after_the_result_says_so():
    async def events():
        yield RegionsReady(job_id="job-1", regions=[], analysis_note="")
        await asyncio.Event().wait()

    async def consume(started):
        async for _ in track_abandonment("discover", events()):
            started.set()

    with structlog.testing.capture_logs() as logs:
        await _cancel_in_task_group(consume)

    [entry] = [e for e in logs if e["event"] == "sse_stream_abandoned"]
    assert entry["result_sent"] is True
    assert entry["last_event"] == "regions"


async def test_completed_stream_logs_no_abandonment():
    async def events():
        yield JobStarted(job_id="job-1")
        yield WorkflowComplete(job_id="job-1")

    with structlog.testing.capture_logs() as logs:
        out = [sse async for sse in track_abandonment("discover", events())]

    assert [s["event"] for s in out] == ["started", "done"]
    assert not [e for e in logs if e["event"] == "sse_stream_abandoned"]


# --- the real discover chain, dropped mid-discovery -------------------------


def _make_executor(monkeypatch, tmp_path, runner_behavior):
    import core.executor as ex
    from core.executor import WorkflowExecutor
    from services.session_service import create_session_service

    monkeypatch.setattr(
        ex, "create_book_to_place_discovery_workflow",
        lambda *a, **k: Workflow(name="stub_workflow"),
    )

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run_async(self, user_id, session_id, new_message):
            await runner_behavior()
            if False:  # pragma: no cover - makes this an async generator
                yield None

    monkeypatch.setattr(ex, "Runner", _FakeRunner)
    svc = create_session_service(
        connection_string=f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}",
        use_database=True,
    )
    executor = WorkflowExecutor(
        config=ExecutorConfig(model_name="gemini-2.0-flash", google_api_key="test-key"),
        session_service=svc,
        model=object(),
    )
    return executor, svc


def _scope():
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/api/v1/itinerary/discover",
        "headers": [],
    }


async def _serve_and_drop(executor, drop_after: bytes) -> list[bytes]:
    """Serve discover over a real EventSourceResponse; the client disconnects
    once a chunk containing ``drop_after`` has been sent."""
    disconnected = asyncio.Event()
    sent: list[bytes] = []

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        body = message.get("body") or b""
        sent.append(body)
        if drop_after in body:
            disconnected.set()

    stream = discover_stream(
        book_title="A Book", author="An Author", preferences=None,
        user_id=USER, executor=executor,
    )
    await asyncio.wait_for(EventSourceResponse(stream)(_scope(), receive, send), timeout=5)
    return sent


async def _settle(svc, timeout: float = 2.0) -> int:
    pool = svc.db_engine.sync_engine.pool
    deadline = asyncio.get_running_loop().time() + timeout
    while pool.checkedout() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    return pool.checkedout()


def _job_id(sent: list[bytes]) -> str:
    import json
    import re

    for chunk in sent:
        m = re.search(rb"event: started\r?\ndata: (.*?)\r?\n", chunk)
        if m:
            return json.loads(m.group(1))["job_id"]
    raise AssertionError("no started event was sent")


async def test_dropped_discover_cleans_up_and_logs_no_error(monkeypatch, tmp_path, caplog):
    async def blocks_mid_discovery():
        await asyncio.Event().wait()

    executor, svc = _make_executor(monkeypatch, tmp_path, blocks_mid_discovery)
    caplog.set_level(logging.WARNING)
    try:
        with structlog.testing.capture_logs() as logs:
            sent = await _serve_and_drop(executor, b"Starting location discovery")
            leftover = await _settle(svc)

        # 1. The connection is back in the pool (ai#305 still holds).
        assert leftover == 0
        # 2. Nothing reached the error level: no Sentry event for a reader leaving.
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not [e for e in logs if e["log_level"] in ("error", "critical")]
        # 3. The cleanup survived the cancel: /status will say FAILED, not
        #    "discovering" forever.
        session = await svc.get_session(app_name=APP_NAME, user_id=USER, session_id=_job_id(sent))
        assert session.state.get(SessionStateKeys.JOB_FAILED) is True
        # 4. The departure is counted once, with where it happened.
        [entry] = [e for e in logs if e["event"] == "sse_stream_abandoned"]
        assert entry["flow"] == "discover"
        assert entry["last_event"] == "progress"
        assert entry["result_sent"] is False
    finally:
        await svc.close()


async def test_real_discovery_failure_is_still_an_error(monkeypatch, tmp_path):
    """Converse: the quiet disconnect path must not swallow a real failure."""
    async def model_blows_up():
        raise RuntimeError("upstream exploded")

    executor, svc = _make_executor(monkeypatch, tmp_path, model_blows_up)
    try:
        with structlog.testing.capture_logs() as logs:
            events = [
                e async for e in executor.discover(book_title="A Book", author="An Author", user_id=USER)
            ]
        assert any(e["event"] == "discover_error" and e["log_level"] == "error" for e in logs)
        assert events[-1] == WorkflowComplete(job_id=events[0].job_id)
    finally:
        await svc.close()
