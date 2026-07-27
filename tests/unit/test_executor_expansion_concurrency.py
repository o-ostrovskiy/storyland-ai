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

from core.events import ExpansionReady, WorkflowComplete, WorkflowError
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


class TestExpandLockRefetchSessionError:
    """MYS-401 r2 (Codex P2): the under-lock re-fetch that guards against
    the TOCTOU race sat outside the SSE error envelope -- a transient
    session-backend failure on THAT call used to propagate raw and
    truncate the stream with no terminal event, the same client-visible-
    truncation class MYS-400 closed for the phase executors. It must now
    emit the same client-safe SessionError + WorkflowComplete pair as the
    initial session lookup a few lines above it.
    """

    async def test_lock_refetch_failure_emits_client_safe_session_error(
        self, monkeypatch
    ):
        real_service = create_session_service(use_database=False)
        executor, ex = _make_executor(monkeypatch, real_service)
        monkeypatch.setattr(
            ex, "Runner", _fake_expansion_runner(_VALID_EXPANSION_DELTA)
        )

        job_id = "job-lock-refetch-failure-1"
        await real_service.create_session(
            app_name=APP_NAME,
            user_id="default",
            session_id=job_id,
            state=dict(_SEED_STATE),
        )

        class _FailOnSecondGet:
            """The initial lookup (before the lock) succeeds; the re-fetch
            UNDER the lock -- the one this fix wraps -- raises."""

            def __init__(self, inner):
                self._inner = inner
                self._calls = 0

            async def get_session(self, *args, **kwargs):
                self._calls += 1
                if self._calls == 2:
                    raise RuntimeError("session backend unavailable")
                return await self._inner.get_session(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        executor._session_service = _FailOnSecondGet(real_service)

        events = await _drain(
            executor.expand(
                job_id=job_id,
                action_id="chip-1",
                action_label="More cafes",
                action_prompt="Find cafes in Bath",
            )
        )

        errors = [e for e in events if isinstance(e, WorkflowError)]
        assert errors and errors[0].error_type == "SessionError", (
            f"expected a client-safe SessionError terminal event on a "
            f"lock-refetch failure, got {events!r}"
        )
        assert any(isinstance(e, WorkflowComplete) for e in events), (
            f"SessionError must still be paired with WorkflowComplete so "
            f"the stream terminates cleanly, got {events!r}"
        )
        assert not any(isinstance(e, ExpansionReady) for e in events)


class TestExpandAdmittedCallerAdoptsFreshSnapshot:
    """MYS-401 r3 (Codex P1): the lock makes the in-progress flag's
    check-then-set atomic, so a genuinely concurrent second caller is
    correctly rejected (TestExpandConcurrencyGuardIsAtomic above). But an
    ADMITTED caller -- one whose own initial, pre-lock get_session() read
    happened to race ahead of a DIFFERENT, already-completed expansion on
    the same job_id -- was still running body() against that stale outer
    capture: `state`, `itinerary`, and `expansion_count` were never
    refreshed from the under-lock re-fetch that had just proven the flag
    clear. The lock guaranteed nothing else could run WHILE it was held; it
    never made the admitted caller's own pre-lock reads fresh.

    This drives exactly that interleaving directly (no artificial
    scheduling needed): caller A runs a real expand() to completion first;
    caller B's session service is then wired so B's *own* initial lookup
    returns the snapshot from BEFORE A ran, while every later call (the
    under-lock re-fetch, the fake runner's write, the post-run re-fetch)
    goes to the real, current service. B is admitted under the lock (A's
    flag is durably clear) and must adopt A's already-persisted result as
    its merge base -- not silently clobber it.
    """

    async def test_admitted_caller_merges_onto_already_completed_expansion(
        self, monkeypatch
    ):
        real_service = create_session_service(use_database=False)
        executor, ex = _make_executor(monkeypatch, real_service)
        monkeypatch.setattr(
            ex, "Runner", _fake_expansion_runner(_VALID_EXPANSION_DELTA)
        )

        job_id = "job-admitted-stale-snapshot-1"
        await real_service.create_session(
            app_name=APP_NAME,
            user_id="default",
            session_id=job_id,
            state=dict(_SEED_STATE),
        )

        # The snapshot caller B's own initial get_session() will be made to
        # return -- captured now, before A runs, so it is genuinely stale
        # by the time B reaches the lock.
        stale_session = await real_service.get_session(
            app_name=APP_NAME, user_id="default", session_id=job_id
        )

        # Caller A: a real, complete expand() -- persists Jane Austen
        # Centre, bumps expansion_count to 1, clears the suggestions list
        # (per _VALID_EXPANSION_DELTA) and the in-progress flag.
        a_events = await _drain(
            executor.expand(
                job_id=job_id,
                action_id="chip-1",
                action_label="More cafes",
                action_prompt="Find cafes in Bath",
            )
        )
        assert any(isinstance(e, ExpansionReady) for e in a_events), (
            f"caller A must complete for this test to model anything, got {a_events!r}"
        )

        after_a = await real_service.get_session(
            app_name=APP_NAME, user_id="default", session_id=job_id
        )
        assert after_a.state[SessionStateKeys.EXPANSION_COUNT] == 1
        assert after_a.state[SessionStateKeys.EXPANSION_IN_PROGRESS] is False

        class _StaleFirstLookup:
            """B's own initial (pre-lock) get_session() returns the
            snapshot captured before A ran. Every other get_session call --
            the under-lock re-fetch, the fake runner's own read, the
            post-run re-fetch -- goes to the real, current service, so B is
            admitted (A's flag is already durably clear) with a stale outer
            capture: exactly the gap this fix closes.
            """

            def __init__(self, inner, stale):
                self._inner = inner
                self._stale = stale
                self._calls = 0

            async def get_session(self, *args, **kwargs):
                self._calls += 1
                if self._calls == 1:
                    return self._stale
                return await self._inner.get_session(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        executor._session_service = _StaleFirstLookup(real_service, stale_session)

        # B reuses chip-1 -- the only id present in the STALE snapshot's
        # last_suggestions (action_id validation runs against B's own
        # pre-lock `state`, so this is the only id that can pass). This is
        # the realistic shape of the bug: a delayed second click validated
        # against a chip the first click already consumed.
        second_delta = {
            SessionStateKeys.LAST_EXPANSION: {
                "parent_city": "Bath",
                "places": [
                    {
                        "name": "Royal Crescent",
                        "type": "landmark",
                        "reason": "Georgian architecture",
                        "time_of_day": "morning",
                    }
                ],
                "suggestions": [],
            }
        }
        monkeypatch.setattr(ex, "Runner", _fake_expansion_runner(second_delta))

        b_events = await _drain(
            executor.expand(
                job_id=job_id,
                action_id="chip-1",
                action_label="More cafes",
                action_prompt="Find cafes in Bath",
            )
        )

        assert any(isinstance(e, ExpansionReady) for e in b_events), (
            f"the admitted second caller should complete, got {b_events!r}"
        )

        final = await real_service.get_session(
            app_name=APP_NAME, user_id="default", session_id=job_id
        )
        # count -> 2: without the fix this ends at 1 (B increments its own
        # stale pre-A count of 0), a lost update identical in shape to the
        # concurrent-race bug this same ticket already fixed once.
        assert final.state[SessionStateKeys.EXPANSION_COUNT] == 2
        bath = next(
            c
            for c in final.state[SessionStateKeys.FINAL_ITINERARY]["cities"]
            if c["name"] == "Bath"
        )
        stop_names = {s["name"] for s in bath["stops"]}
        # Both expansions' places must survive. Without the fix, B merges
        # its new place onto the STALE pre-A itinerary and persists that as
        # the whole of FINAL_ITINERARY -- silently dropping A's Jane Austen
        # Centre even though A's write was already durable.
        assert "Jane Austen Centre" in stop_names, (
            f"caller A's already-persisted place must survive caller B's "
            f"merge, got stops={stop_names!r}"
        )
        assert "Royal Crescent" in stop_names
        # No chip reuse: B's own fresh (empty) suggestion list is what's
        # left standing, not a residual chip-1 a third caller could replay.
        assert final.state[SessionStateKeys.LAST_SUGGESTIONS] == []


class TestCityWordBoundaryMatch:
    """Direct unit tests of the helper -- no session/Runner machinery
    needed, same style as test_core.py's MYS-167 tests of the sibling
    ``_resolve_trusted_action_prompt`` helper.

    MYS-401 r2 (Codex P1): the original reject rule fired on ANY
    capitalized preceding word, which misrouted every-day action_prompts
    with a capitalized leading verb/adjective ("Explore Bath", "Discover
    Barcelona", "Find Victorian London bookshops") to ``cities[0]`` --
    reintroducing the exact cities[0]-misroute class MYS-660 spent 8
    revisions closing. The reject now only fires when the two-word phrase
    it forms is ITSELF another city in the same trip's itinerary (passed
    as ``other_city_names``), so plain capitalized language no longer
    trips it.
    """

    def test_rejects_york_inside_new_york_when_new_york_is_also_a_trip_city(self):
        # A 2-city itinerary that genuinely contains both "York" and "New
        # York" -- the only case the reject should fire for, since "New
        # York" is a real, longer, competing city name from the same trip.
        assert not _matches_city_as_standalone_word(
            "York",
            "Find landmarks near New York City Hall",
            other_city_names=["York", "New York"],
        )

    def test_matches_york_as_its_own_mention(self):
        assert _matches_city_as_standalone_word(
            "York", "Let's explore York a bit more"
        )

    def test_rejects_substring_inside_a_longer_word(self):
        # The old bug: `"york" in "yorkshire"` was True under a plain
        # substring check. Word-boundary rejection is unaffected by the
        # other_city_names change -- Yorkshire never reaches the
        # preceding-word check at all.
        assert not _matches_city_as_standalone_word(
            "York", "A day trip through Yorkshire"
        )

    def test_matches_at_the_very_start_of_the_prompt(self):
        assert _matches_city_as_standalone_word("Bath", "Bath has lovely cafes")

    def test_capitalized_leading_verb_still_matches_the_correct_city(self):
        # Regression (Codex P1, r2): "Explore" is a capitalized sentence-
        # initial verb, not part of a competing city name -- on a 2-city
        # itinerary (Bath, London) this must route to Bath, not fall back
        # to cities[0].
        assert _matches_city_as_standalone_word(
            "Bath",
            "Explore Bath for hidden cafés",
            other_city_names=["Bath", "London"],
        )

    def test_capitalized_leading_adjective_still_matches_the_correct_city(self):
        # Regression (Codex P1, r2): "Victorian" is a capitalized adjective,
        # not part of a competing city name -- must still route to London.
        assert _matches_city_as_standalone_word(
            "London",
            "Find Victorian London bookshops",
            other_city_names=["Bath", "London"],
        )

    def test_capitalized_leading_verb_matches_with_no_itinerary_context(self):
        # Same as above but with the helper's default (empty)
        # other_city_names, matching how a single-city itinerary calls it.
        assert _matches_city_as_standalone_word(
            "Barcelona", "Discover Barcelona\'s Gothic Quarter"
        )
