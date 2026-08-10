"""MYS-816: places from a researcher that never searched cannot claim grounding.

The defect, measured against production Gemini: discovery researchers skip
``google_search`` stochastically — roughly one to two of the four on most runs,
on realist and fictional books alike — and still emit places from model memory.
One observed run returned the author site "Personal Office", which is not a
visitable place at all. Nothing caught it, because the composer-stage guard
checked stops against the UNION of all four payloads, so a researcher that did
search vouched for one that did not.

The fix deliberately adds no new enforcement. It narrows the haystack that
``downgrade_ungrounded_match_types`` already trusts, and the existing guard
does the rest. These tests pin that seam end to end:

  ledger (which agents searched)
    -> unverified payload keys, persisted to session state
    -> excluded from grounding_research_text
    -> composer stops traceable only there demoted to `vibe`

Note the asymmetry these tests protect: an agent that never RAN is not
"unsearched". Fail-open on missing evidence is the rule every grounding guard
in this codebase follows, and breaking it would turn an early error or a
cache hit into a false fabrication alarm.
"""

from types import SimpleNamespace

from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.genai import types

from core.executor import DISCOVERY_RESEARCHER_AUTHORS, RESEARCHER_PAYLOAD_KEYS
from core.extraction import downgrade_ungrounded_match_types
from core.session_state import SessionStateAccessor, SessionStateKeys
from core.types import ExecutorConfig
from plugins.langfuse_plugin import LangfusePlugin


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------

def _ctx(agent_name):
    return SimpleNamespace(
        agent_name=agent_name,
        _invocation_context=SimpleNamespace(branch=agent_name),
    )


def _searched_response():
    return SimpleNamespace(
        usage_metadata=None,
        grounding_metadata=SimpleNamespace(
            web_search_queries=["real locations"], grounding_chunks=[]
        ),
    )


def _unsearched_response():
    return SimpleNamespace(usage_metadata=None, grounding_metadata=None)


class TestLedger:
    async def test_records_searched_and_unsearched(self):
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
        await plugin.after_model_callback(
            callback_context=_ctx("city_researcher"), llm_response=_searched_response()
        )
        await plugin.after_model_callback(
            callback_context=_ctx("author_researcher"),
            llm_response=_unsearched_response(),
        )
        assert plugin.unsearched_agents(DISCOVERY_RESEARCHER_AUTHORS) == frozenset(
            {"author_researcher"}
        )

    async def test_works_with_langfuse_disabled(self):
        """The guard must not vanish on a deploy with no Langfuse credentials."""
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
        assert plugin.enabled is False
        await plugin.after_model_callback(
            callback_context=_ctx("author_researcher"),
            llm_response=_unsearched_response(),
        )
        assert "author_researcher" in plugin.unsearched_agents(
            DISCOVERY_RESEARCHER_AUTHORS
        )

    async def test_any_call_with_receipts_counts_as_searched(self):
        """A later formatting turn reports no receipts; it must not retract."""
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
        await plugin.after_model_callback(
            callback_context=_ctx("city_researcher"), llm_response=_searched_response()
        )
        await plugin.after_model_callback(
            callback_context=_ctx("city_researcher"),
            llm_response=_unsearched_response(),
        )
        assert plugin.unsearched_agents(DISCOVERY_RESEARCHER_AUTHORS) == frozenset()

    async def test_agent_that_never_ran_is_not_unsearched(self):
        """Absence of evidence is not evidence of absence."""
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
        assert plugin.unsearched_agents(DISCOVERY_RESEARCHER_AUTHORS) == frozenset()


# --------------------------------------------------------------------------
# The haystack
# --------------------------------------------------------------------------

_STATE = {
    SessionStateKeys.CITY_DISCOVERY: {"cities": [{"name": "Dublin"}]},
    SessionStateKeys.AUTHOR_SITES: {
        "author_sites": [{"name": "Personal Office"}]
    },
}


class TestGroundingHaystack:
    def test_unverified_payload_excluded(self):
        state = dict(_STATE)
        state[SessionStateKeys.UNVERIFIED_DISCOVERY] = [SessionStateKeys.AUTHOR_SITES]
        text = SessionStateAccessor(state).grounding_research_text
        assert "Dublin" in text
        assert "Personal Office" not in text

    def test_nothing_unverified_keeps_everything(self):
        text = SessionStateAccessor(dict(_STATE)).grounding_research_text
        assert "Dublin" in text and "Personal Office" in text

    def test_malformed_flag_is_ignored(self):
        """A corrupt value must fail open, not crash discovery's successor."""
        state = dict(_STATE)
        state[SessionStateKeys.UNVERIFIED_DISCOVERY] = "author_sites"  # not a list
        assert "Personal Office" in SessionStateAccessor(state).grounding_research_text


# --------------------------------------------------------------------------
# The payoff: the existing guard now demotes the ungrounded stop
# --------------------------------------------------------------------------

def _itinerary():
    return {
        "cities": [
            {
                "name": "Dublin",
                "stops": [
                    {
                        "name": "Dublin",
                        "match_type": "literal",
                        "grounding_source": "searched",
                    },
                    {
                        "name": "Personal Office",
                        "match_type": "literal",
                        "grounding_source": "invented",
                    },
                ],
            }
        ]
    }


class TestDowngrade:
    def test_stop_only_in_unsearched_payload_is_demoted(self):
        state = dict(_STATE)
        state[SessionStateKeys.UNVERIFIED_DISCOVERY] = [SessionStateKeys.AUTHOR_SITES]
        haystack = SessionStateAccessor(state).grounding_research_text

        result = downgrade_ungrounded_match_types(_itinerary(), haystack)
        stops = {s["name"]: s for s in result["cities"][0]["stops"]}

        assert stops["Personal Office"]["match_type"] == "vibe"
        assert stops["Personal Office"]["grounding_source"] is None
        # The searched payload is untouched — this must not blanket-demote.
        assert stops["Dublin"]["match_type"] == "literal"
        assert stops["Dublin"]["grounding_source"] == "searched"

    def test_without_the_flag_the_fabrication_survives(self):
        """Characterises the bug: proves the flag is what does the work."""
        haystack = SessionStateAccessor(dict(_STATE)).grounding_research_text
        result = downgrade_ungrounded_match_types(_itinerary(), haystack)
        stops = {s["name"]: s for s in result["cities"][0]["stops"]}
        assert stops["Personal Office"]["match_type"] == "literal"


# --------------------------------------------------------------------------
# Executor wiring: computed, persisted, and cached
# --------------------------------------------------------------------------

_REGION_STATE = {
    SessionStateKeys.REGION_ANALYSIS: {
        "regions": [
            {
                "region_id": 1,
                "region_name": "Dublin, Ireland",
                "cities": [{"name": "Dublin", "country": "Ireland"}],
                "travel_note": "n",
                "highlights": "h",
            }
        ],
        "analysis_note": "note",
    },
    **_STATE,
}


def _runner_with_skip(monkeypatch, skipped_agent):
    """Fake Runner where one researcher speaks but reports no search receipts."""
    import core.executor as ex
    from core.executor import APP_NAME

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self._session_service = kwargs.get("session_service")
            self._plugins = kwargs.get("plugins") or []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run_async(self, user_id, session_id, new_message):
            # Drive the ledger as ADK would: every researcher reports, one of
            # them without receipts. Only the Langfuse plugin is driven —
            # ADK's own LoggingPlugin reads fields (error_code, …) that a
            # minimal fake response does not carry, and it is not under test.
            ledgers = [p for p in self._plugins if hasattr(p, "unsearched_agents")]
            for agent in DISCOVERY_RESEARCHER_AUTHORS:
                response = (
                    _unsearched_response()
                    if agent == skipped_agent
                    else _searched_response()
                )
                for plugin in ledgers:
                    await plugin.after_model_callback(
                        callback_context=_ctx(agent), llm_response=response
                    )
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id
            )
            await self._session_service.append_event(
                session,
                Event(
                    invocation_id="i",
                    author="system",
                    actions=EventActions(state_delta=dict(_REGION_STATE)),
                ),
            )
            yield Event(
                invocation_id="i",
                author="city_researcher",
                content=types.Content(
                    role="model", parts=[types.Part(text="Dublin is central.")]
                ),
            )

    monkeypatch.setattr(ex, "Runner", _FakeRunner)
    return ex


def _executor(monkeypatch, cache_enabled):
    import core.executor as ex
    from core.executor import WorkflowExecutor
    from services.session_service import create_session_service

    monkeypatch.setattr(
        ex, "create_book_to_place_discovery_workflow", lambda *a, **k: object()
    )
    return WorkflowExecutor(
        config=ExecutorConfig(
            model_name="gemini-2.0-flash",
            google_api_key="test-key",
            cache_enabled=cache_enabled,
        ),
        session_service=create_session_service(use_database=False),
        model=object(),
    )


class TestExecutorWiring:
    async def test_unverified_key_persisted_for_compose(self, monkeypatch):
        """discover() and compose() are separate requests — it must persist."""
        from core.executor import APP_NAME
        from core.events import JobStarted

        _runner_with_skip(monkeypatch, "author_researcher")
        executor = _executor(monkeypatch, cache_enabled=False)

        job_id = None
        async for ev in executor.discover(book_title="Dune", author="Herbert"):
            if isinstance(ev, JobStarted):
                job_id = ev.job_id

        session = await executor.session_service.get_session(
            app_name=APP_NAME, user_id="default", session_id=job_id
        )
        state = SessionStateAccessor(session.state)
        assert state.unverified_discovery == [SessionStateKeys.AUTHOR_SITES]
        assert "Personal Office" not in state.grounding_research_text

    async def test_all_searched_writes_nothing(self, monkeypatch):
        from core.executor import APP_NAME
        from core.events import JobStarted

        _runner_with_skip(monkeypatch, skipped_agent=None)
        executor = _executor(monkeypatch, cache_enabled=False)

        job_id = None
        async for ev in executor.discover(book_title="Dune", author="Herbert"):
            if isinstance(ev, JobStarted):
                job_id = ev.job_id

        session = await executor.session_service.get_session(
            app_name=APP_NAME, user_id="default", session_id=job_id
        )
        assert SessionStateAccessor(session.state).unverified_discovery == []

    async def test_cache_hit_replays_the_flag(self, monkeypatch):
        """A hit that dropped it would serve UNPROTECTED results on the most
        popular titles — the exact bug class the v2 bundle was created for."""
        from core.executor import APP_NAME
        from core.events import JobStarted

        ex = _runner_with_skip(monkeypatch, "author_researcher")
        executor = _executor(monkeypatch, cache_enabled=True)

        async for _ in executor.discover(book_title="Dune", author="Herbert"):
            pass

        monkeypatch.setattr(ex, "Runner", None)  # a fresh run would crash
        hit_job = None
        async for evt in executor.discover(book_title="Dune", author="Herbert"):
            if isinstance(evt, JobStarted):
                hit_job = evt.job_id

        session = await executor.session_service.get_session(
            app_name=APP_NAME, user_id="default", session_id=hit_job
        )
        state = SessionStateAccessor(session.state)
        assert state.unverified_discovery == [SessionStateKeys.AUTHOR_SITES]
        assert "Personal Office" not in state.grounding_research_text

    def test_payload_map_covers_every_researcher(self):
        """A new researcher without a payload mapping would silently escape."""
        assert set(RESEARCHER_PAYLOAD_KEYS) == set(DISCOVERY_RESEARCHER_AUTHORS)
