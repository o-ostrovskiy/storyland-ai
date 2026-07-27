"""MYS-403 gap 2: cross-user / session-isolation coverage.

Isolation between users is structural -- ADK's ``session_service.get_session``
is scoped by ``(app_name, user_id, session_id)``, so a session created under
one user_id is simply absent under a different one. But nothing PROVED that:
every existing status/compose/expand test in this suite runs as the implicit
``dev_user``, so a regression that dropped or mis-threaded the ``user_id``
filter anywhere on this path (e.g. a copy-paste that hardcoded a shared id,
or a refactor that read ``job_id`` alone) would have shipped unnoticed.

These tests create a real session as user "alice" via the real
``InMemorySessionService`` (same technique as
``test_executor_expansion_concurrency.py``) and then access it as user "bob":

* ``get_status`` (api/routes.py) is a plain request/response endpoint --
  the user-B case must be a genuine HTTP-shaped 404, same as a job_id that
  never existed at all.
* ``compose`` / ``expand`` (core/executor.py) are streaming generators that
  never raise an HTTP error -- a not-found session surfaces as an SSE
  ``WorkflowError(error_type="JobNotFound")`` inside a 200-shaped stream
  (the same contract exercised by e.g. TestDiscoverEndpoint::
  test_discover_session_create_failure in test_api.py). user B must get
  exactly that, not user A's data.
"""

import json

import pytest
from fastapi import HTTPException

from core.events import WorkflowError
from core.executor import APP_NAME, WorkflowExecutor
from core.session_state import SessionStateKeys
from core.types import ExecutorConfig
from services.session_service import create_session_service


async def _drain(agen):
    return [e async for e in agen]


async def _drain_sse_route(coro):
    """Call a route function that returns an ``EventSourceResponse`` and
    drain its underlying SSE-dict generator (event/data pairs), decoding
    each ``data`` payload's JSON. This exercises the actual HTTP route
    function -- request -> ``compose_stream``/``expand_stream`` ->
    ``executor`` -- not just the executor directly, so a regression that
    stopped forwarding the authenticated ``user_id`` anywhere on that path
    (a route or streaming adapter falling back to a shared default) would
    show up here even though it would leave an executor-level-only test
    green.
    """
    response = await coro
    events = []
    async for sse_dict in response.body_iterator:
        payload = json.loads(sse_dict["data"]) if sse_dict.get("data") else {}
        events.append((sse_dict["event"], payload))
    return events


_SEED_STATE = {
    SessionStateKeys.BOOK_METADATA: {
        "book_title": "Persuasion",
        "author": "Jane Austen",
    },
    SessionStateKeys.REGION_ANALYSIS: {
        "regions": [{"region_id": 1, "region_name": "England"}],
        "analysis_note": "test",
    },
    SessionStateKeys.FINAL_ITINERARY: {
        "cities": [
            {
                "name": "Bath",
                "country": "England",
                "days_suggested": 2,
                "overview": "Austen's Bath",
                "stops": [],
            }
        ],
        "summary_text": "seed",
    },
    SessionStateKeys.LAST_SUGGESTIONS: [
        {"id": "chip-1", "label": "More cafes", "action_prompt": "Find cafes in Bath"}
    ],
    SessionStateKeys.BOOK_TITLE: "Persuasion",
    SessionStateKeys.AUTHOR: "Jane Austen",
}

_OWNER_USER_ID = "alice"
_OTHER_USER_ID = "bob"
_JOB_ID = "job-isolation-1"


@pytest.fixture
async def session_as_alice(monkeypatch):
    """A real session service with one session created under user 'alice',
    plus a WorkflowExecutor wired to it. ``create_expansion_workflow`` /
    ``create_book_to_place_composition_workflow`` are left real (not
    monkeypatched, unlike the MYS-401 concurrency tests) -- these isolation
    tests only need the up-front session lookup to behave correctly, not a
    full successful run, so a dummy ``model=object()`` reaching the agent
    layer is caught by ``run_guarded``'s exception mapping (MYS-400) and
    surfaces as SOME client-safe error, never a crash and never JobNotFound.
    """
    service = create_session_service(use_database=False)
    config = ExecutorConfig(model_name="gemini-2.0-flash", google_api_key="test-key")
    executor = WorkflowExecutor(config=config, session_service=service, model=object())
    await service.create_session(
        app_name=APP_NAME,
        user_id=_OWNER_USER_ID,
        session_id=_JOB_ID,
        state=dict(_SEED_STATE),
    )
    return executor, service


class TestStatusEndpointIsolation:
    """api/routes.py::get_status -- a real 404, not an SSE error."""

    async def test_owner_can_read_their_own_status(self, session_as_alice):
        from api.routes import get_status
        import api.dependencies as deps

        executor, service = session_as_alice
        app_state = deps.AppState(
            config=None,
            executor=executor,
            rate_limiter=None,
            inflight_limiter=None,
        )
        deps._app_state = app_state
        try:
            result = await get_status(job_id=_JOB_ID, user_id=_OWNER_USER_ID)
        finally:
            deps._app_state = None
        assert result.job_id == _JOB_ID
        assert result.book_title == "Persuasion"

    async def test_a_different_user_gets_404_not_alices_data(self, session_as_alice):
        from api.routes import get_status
        import api.dependencies as deps

        executor, service = session_as_alice
        app_state = deps.AppState(
            config=None,
            executor=executor,
            rate_limiter=None,
            inflight_limiter=None,
        )
        deps._app_state = app_state
        try:
            with pytest.raises(HTTPException) as exc:
                await get_status(job_id=_JOB_ID, user_id=_OTHER_USER_ID)
        finally:
            deps._app_state = None
        assert exc.value.status_code == 404


class TestComposeIsolation:
    async def test_owner_lookup_succeeds_not_job_not_found(self, session_as_alice):
        executor, _service = session_as_alice
        events = await _drain(
            executor.compose(job_id=_JOB_ID, region_ids=[1], user_id=_OWNER_USER_ID)
        )
        errors = [e for e in events if isinstance(e, WorkflowError)]
        assert not any(e.error_type == "JobNotFound" for e in errors), (
            f"alice's own session must be found by compose(); got {events!r}"
        )

    async def test_a_different_user_cannot_compose_alices_session(
        self, session_as_alice
    ):
        executor, _service = session_as_alice
        events = await _drain(
            executor.compose(job_id=_JOB_ID, region_ids=[1], user_id=_OTHER_USER_ID)
        )
        errors = [e for e in events if isinstance(e, WorkflowError)]
        assert errors and errors[0].error_type == "JobNotFound", (
            f"bob must not be able to compose alice's session via her job_id; "
            f"got {events!r}"
        )
        # And nothing of alice's leaked into the event stream itself.
        assert not any("Bath" in str(e) for e in events)


class TestComposeRouteIsolation:
    """Codex P2 (MYS-403 review): the tests above call executor.compose()
    directly, which only proves the EXECUTOR's session lookup is
    user-scoped. This PR's stated purpose is protecting the auth boundary
    -- the HTTP surface -- and that's exactly where a regression would
    hide: a route or streaming adapter that stopped forwarding the
    authenticated user_id and fell back to a shared default would leave
    the tests above green while HTTP callers crossed the boundary. These
    call the real ``api.routes.compose`` route function (request ->
    ``compose_stream`` -> executor), the same pattern
    ``TestStatusEndpointIsolation`` above already uses for ``get_status``.
    """

    async def test_owner_route_call_succeeds_not_job_not_found(
        self, session_as_alice
    ):
        from api.models import ComposeRequest
        from api.routes import compose
        import api.dependencies as deps

        executor, _service = session_as_alice
        deps._app_state = deps.AppState(
            config=None, executor=executor, rate_limiter=None, inflight_limiter=None
        )
        try:
            events = await _drain_sse_route(
                compose(
                    job_id=_JOB_ID,
                    request=ComposeRequest(region_ids=[1]),
                    user_id=_OWNER_USER_ID,
                )
            )
        finally:
            deps._app_state = None
        errors = [payload for etype, payload in events if etype == "error"]
        assert not any(e.get("error_type") == "JobNotFound" for e in errors), (
            f"alice's own session must be found via the real compose route; "
            f"got {events!r}"
        )

    async def test_bob_route_call_gets_job_not_found_not_alices_data(
        self, session_as_alice
    ):
        from api.models import ComposeRequest
        from api.routes import compose
        import api.dependencies as deps

        executor, _service = session_as_alice
        deps._app_state = deps.AppState(
            config=None, executor=executor, rate_limiter=None, inflight_limiter=None
        )
        try:
            events = await _drain_sse_route(
                compose(
                    job_id=_JOB_ID,
                    request=ComposeRequest(region_ids=[1]),
                    user_id=_OTHER_USER_ID,
                )
            )
        finally:
            deps._app_state = None
        errors = [payload for etype, payload in events if etype == "error"]
        assert errors and errors[0].get("error_type") == "JobNotFound", (
            f"bob must not be able to compose alice's session through the "
            f"real HTTP route; got {events!r}"
        )
        # Route→stream→executor propagation must not leak alice's data
        # anywhere in the wire-shaped output, not just the domain events.
        assert not any("Bath" in json.dumps(payload) for _etype, payload in events)


class TestExpandIsolation:
    async def test_owner_lookup_succeeds_not_job_not_found(self, session_as_alice):
        executor, _service = session_as_alice
        events = await _drain(
            executor.expand(
                job_id=_JOB_ID,
                action_id="chip-1",
                action_label="More cafes",
                action_prompt="Find cafes in Bath",
                user_id=_OWNER_USER_ID,
            )
        )
        errors = [e for e in events if isinstance(e, WorkflowError)]
        assert not any(e.error_type == "JobNotFound" for e in errors), (
            f"alice's own session must be found by expand(); got {events!r}"
        )

    async def test_a_different_user_cannot_expand_alices_session(
        self, session_as_alice
    ):
        executor, _service = session_as_alice
        events = await _drain(
            executor.expand(
                job_id=_JOB_ID,
                action_id="chip-1",
                action_label="More cafes",
                action_prompt="Find cafes in Bath",
                user_id=_OTHER_USER_ID,
            )
        )
        errors = [e for e in events if isinstance(e, WorkflowError)]
        assert errors and errors[0].error_type == "JobNotFound", (
            f"bob must not be able to expand alice's session via her job_id "
            f"and her chip id; got {events!r}"
        )
        assert not any("Bath" in str(e) for e in events)


class TestExpandRouteIsolation:
    """Codex P2 (MYS-403 review) -- see TestComposeRouteIsolation's
    docstring: the same route→stream→executor gap applies to expand()."""

    async def test_owner_route_call_succeeds_not_job_not_found(
        self, session_as_alice
    ):
        from api.models import ExpandRequest
        from api.routes import expand
        import api.dependencies as deps

        executor, _service = session_as_alice
        deps._app_state = deps.AppState(
            config=None, executor=executor, rate_limiter=None, inflight_limiter=None
        )
        try:
            events = await _drain_sse_route(
                expand(
                    job_id=_JOB_ID,
                    request=ExpandRequest(
                        action_id="chip-1",
                        action_label="More cafes",
                        action_prompt="Find cafes in Bath",
                    ),
                    user_id=_OWNER_USER_ID,
                )
            )
        finally:
            deps._app_state = None
        errors = [payload for etype, payload in events if etype == "error"]
        assert not any(e.get("error_type") == "JobNotFound" for e in errors), (
            f"alice's own session must be found via the real expand route; "
            f"got {events!r}"
        )

    async def test_bob_route_call_gets_job_not_found_not_alices_data(
        self, session_as_alice
    ):
        from api.models import ExpandRequest
        from api.routes import expand
        import api.dependencies as deps

        executor, _service = session_as_alice
        deps._app_state = deps.AppState(
            config=None, executor=executor, rate_limiter=None, inflight_limiter=None
        )
        try:
            events = await _drain_sse_route(
                expand(
                    job_id=_JOB_ID,
                    request=ExpandRequest(
                        action_id="chip-1",
                        action_label="More cafes",
                        action_prompt="Find cafes in Bath",
                    ),
                    user_id=_OTHER_USER_ID,
                )
            )
        finally:
            deps._app_state = None
        errors = [payload for etype, payload in events if etype == "error"]
        assert errors and errors[0].get("error_type") == "JobNotFound", (
            f"bob must not be able to expand alice's session through the "
            f"real HTTP route, even holding her own chip id; got {events!r}"
        )
        assert not any("Bath" in json.dumps(payload) for _etype, payload in events)
