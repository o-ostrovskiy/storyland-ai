"""MYS-401: expand()'s (and recommend_books()'s identical-shape) concurrency
guard was check-then-set (TOCTOU) -- two overlapping requests for the same
job_id could both read "not in progress" before either's set was durable, so
both proceeded: double-billing Gemini and racing to persist the merge (the
later append_event silently drops the first expansion's places, and
expansion_count is a lost update instead of ending at the right value).

Test (a) below forces the two calls to genuinely interleave -- without a
real yield point in the fakes, asyncio would just run them back-to-back on
this single thread and the race would never be exercised -- by wrapping the
real InMemorySessionService's ``get_session`` with one ``asyncio.sleep(0)``
(a real event-loop tick). Confirmed red against the pre-fix executor.py
(both calls completed; the loser silently overwrote the winner's merge and
``expansion_count`` ended at 1 via a lost update rather than the loser being
rejected) before writing the fix -- see the PR description.

Test (b) covers the second finding on the same ticket: a malformed merge
must not be persisted, so a degraded ``FINAL_ITINERARY`` never reaches
``/status``.

Test (c) covers the third: word-boundary city-name matching must not treat
"York" as a match inside "New York".
"""

import asyncio

from core.events import ExpansionReady, WorkflowError
from core.executor import APP_NAME, _matches_city_as_standalone_word
from core.session_state import SessionStateKeys
from core.types import ExecutorConfig
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from services.session_service import create_session_service

_SEED_ITINERARY = {
    "cities": [
        {
            "name": "Bath",
            "country": "England",
            "days_suggested": 2,
            "overview": "Austen's Bath",
            "stops": [
                {
                    "name": "The Pump Room",
                    "type": "landmark",
                    "reason": "Featured in Persuasion",
                    "time_of_day": "morning",
                }
            ],
        }
    ],
    "summary_text": "A Persuasion journey through Bath.",
}

_BROKEN_ITINERARY = {
    "cities": [
        {
            "name": "Bath",
            "country": "England",
            "days_suggested": 2,
            # "overview" deliberately omitted -- CityPlan requires it. Models
            # a pre-existing formatter quirk this expansion merges on top of.
            "stops": [
                {
                    "name": "The Pump Room",
                    "type": "landmark",
                    "reason": "Featured in Persuasion",
                    "time_of_day": "morning",
                }
            ],
        }
    ],
    "summary_text": "A Persuasion journey through Bath.",
}

_SEED_STATE = {
    SessionStateKeys.FINAL_ITINERARY: _SEED_ITINERARY,
    SessionStateKeys.LAST_SUGGESTIONS: [
        {"id": "chip-1", "label": "More cafes", "action_prompt": "Find cafes in Bath"}
    ],
    SessionStateKeys.BOOK_TITLE: "Persuasion",
    SessionStateKeys.AUTHOR: "Jane Austen",
}

_VALID_EXPANSION_DELTA = {
    SessionStateKeys.LAST_EXPANSION: {
        "parent_city": "Bath",
        "places": [
            {
                "name": "Jane Austen Centre",
                "type": "museum",
                "reason": "Dedicated to the author",
                "time_of_day": "afternoon",
            }
        ],
        "suggestions": [],
    }
}


def _yielding(service):
    """Wrap a real session service so get_session() takes one real
    event-loop tick, forcing two concurrently-scheduled expand() calls to
    interleave instead of running back-to-back on this single thread.
    """

    class _Wrapped:
        def __init__(self, inner):
            self._inner = inner

        async def get_session(self, *args, **kwargs):
            await asyncio.sleep(0)
            return await self._inner.get_session(*args, **kwargs)

        async def append_event(self, *args, **kwargs):
            # Also yield before the WRITE, not just the read: the TOCTOU
            # window is between "read the flag" and "the write becomes
            # durable" -- without a yield point here too, the first caller's
            # read-then-write happens as one uninterrupted synchronous burst
            # on this single thread, and the second caller's read (delayed
            # only by the get_session hook above) would always land after
            # the first caller's write anyway, never exercising the race.
            await asyncio.sleep(0)
            return await self._inner.append_event(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    return _Wrapped(service)


def _fake_expansion_runner(state_delta):
    """A fake Runner that writes a state delta via append_event, like a
    real expansion run -- no Gemini call, but exercises the real
    merge/persist/extraction path (same technique as test_golden_stream.py).
    """

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self._session_service = kwargs.get("session_service")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run_async(self, user_id, session_id, new_message):
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id
            )
            ev = Event(
                invocation_id="system",
                author="system",
                actions=EventActions(state_delta=state_delta),
            )
            await self._session_service.append_event(session, ev)
            if False:  # async generator with no yielded events
                yield None

    return _FakeRunner


def _make_executor(monkeypatch, real_service, *, wrap_for_race=False):
    import core.executor as ex

    monkeypatch.setattr(ex, "create_expansion_workflow", lambda *a, **k: object())

    config = ExecutorConfig(model_name="gemini-2.0-flash", google_api_key="test-key")
    executor = ex.WorkflowExecutor(
        config=config, session_service=real_service, model=object()
    )
    if wrap_for_race:
        executor._session_service = _yielding(real_service)
    return executor, ex


async def _drain(agen):
    return [e async for e in agen]


class TestExpandConcurrencyGuardIsAtomic:
    async def test_two_overlapping_expand_calls_only_one_proceeds(self, monkeypatch):
        real_service = create_session_service(use_database=False)
        executor, ex = _make_executor(monkeypatch, real_service, wrap_for_race=True)
        monkeypatch.setattr(
            ex, "Runner", _fake_expansion_runner(_VALID_EXPANSION_DELTA)
        )

        job_id = "job-concurrency-1"
        await real_service.create_session(
            app_name=APP_NAME,
            user_id="default",
            session_id=job_id,
            state=dict(_SEED_STATE),
        )

        def _call():
            return _drain(
                executor.expand(
                    job_id=job_id,
                    action_id="chip-1",
                    action_label="More cafes",
                    action_prompt="Find cafes in Bath",
                )
            )

        results = await asyncio.wait_for(
            asyncio.gather(_call(), _call()), timeout=5
        )

        ready_runs = [
            r for r in results if any(isinstance(e, ExpansionReady) for e in r)
        ]
        rejected_runs = [
            r
            for r in results
            if any(
                isinstance(e, WorkflowError) and e.error_type == "ExpansionInProgress"
                for e in r
            )
        ]
        assert len(ready_runs) == 1, (
            "exactly one of two overlapping expand() calls should complete "
            f"the expansion; got {len(ready_runs)} (results={results!r})"
        )
        assert len(rejected_runs) == 1, (
            "the other overlapping call must be rejected as "
            f"already-in-progress, not silently double-run (results={results!r})"
        )

        final = await real_service.get_session(
            app_name=APP_NAME, user_id="default", session_id=job_id
        )
        # The lost-update half of the bug: both calls incrementing their own
        # stale `expansion_count + 1` would also end at 1 here by coincidence
        # (0+1 twice), so the real regression signal is the rejected-call
        # count above -- this assertion just pins the flag is cleared.
        assert final.state[SessionStateKeys.EXPANSION_COUNT] == 1
        assert final.state[SessionStateKeys.EXPANSION_IN_PROGRESS] is False
        # Exactly one Gemini-equivalent run happened -- the merged itinerary
        # carries the new place exactly once, not twice/dropped.
        bath = next(
            c
            for c in final.state[SessionStateKeys.FINAL_ITINERARY]["cities"]
            if c["name"] == "Bath"
        )
        new_place_count = sum(
            1 for s in bath["stops"] if s["name"] == "Jane Austen Centre"
        )
        assert new_place_count == 1


class TestExpandMergeRevalidation:
    async def test_malformed_merge_is_not_persisted(self, monkeypatch):
        real_service = create_session_service(use_database=False)
        executor, ex = _make_executor(monkeypatch, real_service)
        # The NEW place is well-formed (passes CityStop validation at
        # extraction, same as any real expansion) -- the defect this test
        # pins is in the PRE-EXISTING itinerary this expansion merges into
        # ("overview" missing from the Bath CityPlan, a formatter quirk
        # from an earlier phase per the ticket's own framing). Merging a
        # good place into an already-broken itinerary must still be
        # rejected: the merged whole is what gets re-validated, not just
        # the new place.
        monkeypatch.setattr(
            ex, "Runner", _fake_expansion_runner(_VALID_EXPANSION_DELTA)
        )

        job_id = "job-merge-validation-1"
        await real_service.create_session(
            app_name=APP_NAME,
            user_id="default",
            session_id=job_id,
            state={**_SEED_STATE, SessionStateKeys.FINAL_ITINERARY: _BROKEN_ITINERARY},
        )

        events = await _drain(
            executor.expand(
                job_id=job_id,
                action_id="chip-1",
                action_label="More cafes",
                action_prompt="Find cafes in Bath",
            )
        )

        errors = [e for e in events if isinstance(e, WorkflowError)]
        assert errors, f"expected a WorkflowError, got {events!r}"
        assert not any(isinstance(e, ExpansionReady) for e in events)

        final = await real_service.get_session(
            app_name=APP_NAME, user_id="default", session_id=job_id
        )
        # FINAL_ITINERARY must be exactly what it was before this expansion
        # -- /status must never re-serve the degraded merge.
        assert final.state[SessionStateKeys.FINAL_ITINERARY] == _BROKEN_ITINERARY
        assert final.state[SessionStateKeys.EXPANSION_IN_PROGRESS] is False


class TestCityWordBoundaryMatch:
    """Direct unit tests of the helper -- no session/Runner machinery
    needed, same style as test_core.py's MYS-167 tests of the sibling
    ``_resolve_trusted_action_prompt`` helper.
    """

    def test_rejects_york_inside_new_york(self):
        assert not _matches_city_as_standalone_word(
            "York", "Find landmarks near New York City Hall"
        )

    def test_matches_york_as_its_own_mention(self):
        assert _matches_city_as_standalone_word(
            "York", "Let's explore York a bit more"
        )

    def test_rejects_substring_inside_a_longer_word(self):
        # The old bug: `"york" in "yorkshire"` was True under a plain
        # substring check.
        assert not _matches_city_as_standalone_word(
            "York", "A day trip through Yorkshire"
        )

    def test_matches_at_the_very_start_of_the_prompt(self):
        assert _matches_city_as_standalone_word("Bath", "Bath has lovely cafes")
