"""Contract tests for the discovery-stage grounding audit.

The audit measures how much of what the discovery FORMATTERS emitted traces
back to what the researchers actually found — the one link in the grounding
chain nothing verified. It is observation-only by decision: we need the real
miss rate before deciding whether dropping ungrounded places is safe, because
a miss can equally mean "the formatter invented it" or "the token rule was
strict about a paraphrase".

So these tests pin two things above all: the payloads come back UNMUTATED, and
"no evidence" is reported as None (cannot say) rather than zero (nothing was
grounded). Confusing those two would turn a broken capture seam into a false
fabrication alarm.
"""

import copy

from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.genai import types

from core.extraction import audit_discovery_grounding
from core.session_state import SessionStateKeys
from core.types import ExecutorConfig

_RESEARCH = (
    "The Pump Room in Bath is central to Persuasion. "
    "Lyme Regis and the Cobb also feature. "
    "Chawton House was Austen's home."
)


def _payloads(cities=None, landmarks=None, author_sites=None):
    return {
        "cities": {"cities": cities} if cities is not None else None,
        "landmarks": {"landmarks": landmarks} if landmarks is not None else None,
        "author_sites": (
            {"author_sites": author_sites} if author_sites is not None else None
        ),
    }


class TestCounts:
    def test_fully_grounded(self):
        payloads = _payloads(cities=[{"name": "Bath"}, {"name": "Lyme Regis"}])
        assert audit_discovery_grounding(payloads, _RESEARCH) == {"cities": (2, 2)}

    def test_partially_grounded(self):
        payloads = _payloads(
            cities=[{"name": "Bath"}, {"name": "Casablanca"}, {"name": "Lyme Regis"}]
        )
        assert audit_discovery_grounding(payloads, _RESEARCH) == {"cities": (2, 3)}

    def test_all_three_payload_kinds_counted(self):
        payloads = _payloads(
            cities=[{"name": "Bath"}],
            landmarks=[{"name": "The Pump Room"}, {"name": "Invented Tower"}],
            author_sites=[{"name": "Chawton House"}],
        )
        assert audit_discovery_grounding(payloads, _RESEARCH) == {
            "cities": (1, 1),
            "landmarks": (1, 2),
            "author_sites": (1, 1),
        }

    def test_nothing_grounded_reports_zero_not_none(self):
        """Zero-of-N is a real finding and must be distinguishable from 'cannot say'."""
        payloads = _payloads(cities=[{"name": "Casablanca"}])
        assert audit_discovery_grounding(payloads, _RESEARCH) == {"cities": (0, 1)}


class TestCannotSay:
    """None means no usable evidence — never 'nothing was grounded'."""

    def test_no_researcher_text_fails_open(self):
        payloads = _payloads(cities=[{"name": "Bath"}])
        assert audit_discovery_grounding(payloads, "") is None

    def test_whitespace_only_researcher_text_fails_open(self):
        payloads = _payloads(cities=[{"name": "Bath"}])
        assert audit_discovery_grounding(payloads, "   \n  ") is None

    def test_no_payloads_at_all(self):
        assert audit_discovery_grounding(_payloads(), _RESEARCH) is None

    def test_empty_entry_lists(self):
        assert audit_discovery_grounding(_payloads(cities=[]), _RESEARCH) is None

    def test_non_dict_payload_is_skipped(self):
        """A payload stored as a model object rather than a dict must not crash."""
        payloads = {"cities": object(), "landmarks": None, "author_sites": None}
        assert audit_discovery_grounding(payloads, _RESEARCH) is None

    def test_non_dict_entries_are_skipped_not_counted_as_grounded(self):
        payloads = _payloads(cities=[{"name": "Bath"}, "Lyme Regis"])
        assert audit_discovery_grounding(payloads, _RESEARCH) == {"cities": (1, 2)}


class TestPurity:
    def test_payloads_are_not_mutated(self):
        """The audit observes; enforcement is a separate, later decision."""
        payloads = _payloads(
            cities=[{"name": "Bath"}, {"name": "Casablanca"}],
            landmarks=[{"name": "Invented Tower"}],
            author_sites=[{"name": "Chawton House"}],
        )
        before = copy.deepcopy(payloads)
        audit_discovery_grounding(payloads, _RESEARCH)
        assert payloads == before


# --------------------------------------------------------------------------
# Executor wiring
# --------------------------------------------------------------------------

_STATE_DELTA = {
    SessionStateKeys.REGION_ANALYSIS: {
        "regions": [
            {
                "region_id": 1,
                "region_name": "South West England",
                "cities": [{"name": "Bath", "country": "England"}],
                "travel_note": "n",
                "highlights": "h",
            }
        ],
        "analysis_note": "note",
    },
    SessionStateKeys.CITY_DISCOVERY: {
        "cities": [{"name": "Bath"}, {"name": "Casablanca"}]
    },
}


def _researcher_runner(monkeypatch):
    """Fake Runner that speaks as a researcher, then writes formatter state.

    Mirrors the real chain's shape closely enough to exercise the capture
    seam: pump_events only reads ``event.author`` and ``event.content.parts``.
    """
    import core.executor as ex
    from core.executor import APP_NAME

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self._session_service = kwargs.get("session_service")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run_async(self, user_id, session_id, new_message):
            yield Event(
                invocation_id="i",
                author="city_researcher",
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Bath is central to Persuasion.")],
                ),
            )
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id
            )
            await self._session_service.append_event(
                session,
                Event(
                    invocation_id="i",
                    author="system",
                    actions=EventActions(state_delta=_STATE_DELTA),
                ),
            )

    monkeypatch.setattr(ex, "Runner", _FakeRunner)
    return ex


def _make_executor(monkeypatch, cache_enabled):
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
    async def test_fresh_run_audits_against_captured_researcher_text(
        self, monkeypatch, capsys
    ):
        """The capture seam actually reaches the audit: 1 of 2 cities grounded."""
        _researcher_runner(monkeypatch)
        executor = _make_executor(monkeypatch, cache_enabled=False)

        async for _ in executor.discover(book_title="Persuasion", author="Jane Austen"):
            pass

        out = capsys.readouterr().out
        assert "discovery_grounding_audit" in out
        assert "grounded=1" in out
        assert "total=2" in out

    async def test_no_researcher_text_logs_no_capture_not_a_false_alarm(
        self, monkeypatch, capsys
    ):
        """Missing evidence must never be reported as ungrounded output."""
        import core.executor as ex
        from core.executor import APP_NAME

        class _SilentRunner:
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
                await self._session_service.append_event(
                    session,
                    Event(
                        invocation_id="i",
                        author="system",
                        actions=EventActions(state_delta=_STATE_DELTA),
                    ),
                )
                if False:
                    yield None

        monkeypatch.setattr(ex, "create_book_to_place_discovery_workflow", lambda *a, **k: object())
        monkeypatch.setattr(ex, "Runner", _SilentRunner)
        executor = _make_executor(monkeypatch, cache_enabled=False)

        async for _ in executor.discover(book_title="Persuasion", author="Jane Austen"):
            pass

        out = capsys.readouterr().out
        assert "discovery_grounding_no_capture" in out
        assert "discovery_grounding_audit" not in out

    async def test_cache_hit_does_not_audit(self, monkeypatch, capsys):
        """A replay has no researcher text by construction.

        Auditing there would fire a permanent no-evidence warning on exactly
        the popular, repeated titles the cache serves most.
        """
        ex = _researcher_runner(monkeypatch)
        executor = _make_executor(monkeypatch, cache_enabled=True)

        async for _ in executor.discover(book_title="Persuasion", author="Jane Austen"):
            pass
        capsys.readouterr()  # discard run 1

        monkeypatch.setattr(ex, "Runner", None)  # a fresh run would now crash
        async for _ in executor.discover(book_title="Persuasion", author="Jane Austen"):
            pass

        out = capsys.readouterr().out
        assert "discovery_grounding_audit" not in out
        assert "discovery_grounding_no_capture" not in out
