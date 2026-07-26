"""MYS-400: compose/local_atmosphere/expand/recommend_books must surface the
same client-safe typed error discover() already does, not a raw str(e).

``core/run_harness.py::run_guarded`` already pins the generic contract
(default vs custom ``map_exception``, tested in ``test_run_harness.py``); what
was missing is that only discover()'s ``GuardSpec`` set ``map_exception`` at
all -- the other four phases fell through to the default
``WorkflowError(message=str(e), ...)`` path, so a child-task
``ExceptionGroup`` (or any other internal exception) reached the client
verbatim on 4 of 5 flows.

These tests intercept ``run_guarded`` to capture the ``GuardSpec`` each phase
builds -- without needing to fake a full ADK Runner/agent run -- then invoke
the captured ``map_exception`` directly with a real exception and assert the
same contract ``test_discovery_errors.py`` pins for discover(): the emitted
message is never the raw exception text, ``error_type`` is
``"DiscoveryComposeError"``, and ``reason`` is the classifier's verdict.
"""

import core.executor as executor_module
from core.executor import APP_NAME, WorkflowExecutor
from core.session_state import SessionStateKeys
from core.types import ExecutorConfig
from services.session_service import create_session_service


def _capture_guard_spec(monkeypatch):
    """Replace ``run_guarded`` with a spy that records the GuardSpec and
    yields nothing, so the phase method reaches its ``spec = GuardSpec(...)``
    line without needing a real Runner/agent/workflow."""
    captured: dict = {}

    async def spy(body, spec):
        captured["spec"] = spec
        return
        yield  # pragma: no cover - makes this an async generator function

    monkeypatch.setattr(executor_module, "run_guarded", spy)
    return captured


def _bare_executor():
    config = ExecutorConfig(model_name="gemini-2.0-flash", google_api_key="test-key")
    return WorkflowExecutor(
        config=config,
        session_service=create_session_service(use_database=False),
        model=object(),
    )


_RAW_LEAK_MARKERS = ("boom-from-child-task", "TaskGroup", "ExceptionGroup")


def _assert_client_safe(err, expected_reason="transient"):
    assert err.error_type == "DiscoveryComposeError"
    assert err.reason == expected_reason
    for marker in _RAW_LEAK_MARKERS:
        assert marker not in err.message


class TestComposeMapsExceptionClientSafe:
    async def test_compose_wires_a_client_safe_mapper(self, monkeypatch):
        captured = _capture_guard_spec(monkeypatch)
        executor = _bare_executor()
        session_service = executor._session_service
        job_id = "job-compose-1"
        await session_service.create_session(
            app_name=APP_NAME,
            user_id="default",
            session_id=job_id,
            state={
                SessionStateKeys.REGION_ANALYSIS: {
                    "regions": [{"region_id": 1, "name": "Bath, England"}]
                }
            },
        )

        [e async for e in executor.compose(job_id=job_id, region_ids=[1])]

        spec = captured["spec"]
        assert spec.map_exception is not None
        err = spec.map_exception(RuntimeError("boom-from-child-task"))
        _assert_client_safe(err)


class TestLocalAtmosphereMapsExceptionClientSafe:
    async def test_local_atmosphere_wires_a_client_safe_mapper(self, monkeypatch):
        captured = _capture_guard_spec(monkeypatch)
        executor = _bare_executor()

        [
            e
            async for e in executor.local_atmosphere(
                book_title="1984",
                author="George Orwell",
                location_label="New York, NY",
                lat=40.7,
                lng=-74.0,
            )
        ]

        spec = captured["spec"]
        assert spec.map_exception is not None
        err = spec.map_exception(RuntimeError("boom-from-child-task"))
        _assert_client_safe(err)


class TestExpandMapsExceptionClientSafe:
    async def test_expand_wires_a_client_safe_mapper(self, monkeypatch):
        captured = _capture_guard_spec(monkeypatch)
        executor = _bare_executor()
        session_service = executor._session_service
        job_id = "job-expand-1"
        await session_service.create_session(
            app_name=APP_NAME,
            user_id="default",
            session_id=job_id,
            state={
                SessionStateKeys.FINAL_ITINERARY: {
                    "cities": [{"name": "Bath", "stops": []}]
                },
                SessionStateKeys.LAST_SUGGESTIONS: [
                    {"id": "chip-1", "label": "More cafes", "action_prompt": "Bath cafes"}
                ],
            },
        )

        [
            e
            async for e in executor.expand(
                job_id=job_id,
                action_id="chip-1",
                action_label="More cafes",
                action_prompt="Bath cafes",
            )
        ]

        spec = captured["spec"]
        assert spec.map_exception is not None
        err = spec.map_exception(RuntimeError("boom-from-child-task"))
        _assert_client_safe(err)


class TestRecommendBooksMapsExceptionClientSafe:
    async def test_recommend_books_wires_a_client_safe_mapper(self, monkeypatch):
        captured = _capture_guard_spec(monkeypatch)
        executor = _bare_executor()
        session_service = executor._session_service
        job_id = "job-recs-1"
        await session_service.create_session(
            app_name=APP_NAME,
            user_id="default",
            session_id=job_id,
            state={
                SessionStateKeys.FINAL_ITINERARY: {
                    "cities": [{"name": "Bath", "stops": []}]
                },
                SessionStateKeys.BOOK_RECOMMENDATION_CHIP_ID: "books-chip-1",
            },
        )

        [
            e
            async for e in executor.recommend_books(
                job_id=job_id,
                action_id="books-chip-1",
                action_label="Find books like this",
                action_prompt="",
            )
        ]

        spec = captured["spec"]
        assert spec.map_exception is not None
        err = spec.map_exception(RuntimeError("boom-from-child-task"))
        _assert_client_safe(err)
