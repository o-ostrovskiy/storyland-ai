"""Book-identity guard: discovery must not plan a trip for a different book.

Prod 2026-11-21: "dddd" by "ddd" was searched, resolved by google_search to
Inio Asano's "Dead Dead Demon's Dededede Destruction" (set in Tokyo), and the
reader got a Tokyo trip titled "dddd by ddd". The book_context formatter
(prompts v5) now labels whether its research is about the requested book;
core/book_identity.py turns that label into a verdict and discover() stops
the run with a BookNotFound error before any region is emitted or cached.
"""

import json
from pathlib import Path

from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.workflow import Workflow

from agents.prompts import load_prompts
from core.book_identity import NOT_FOUND_MESSAGE, check_book_identity
from core.events import RegionsReady, WorkflowComplete, WorkflowError
from core.session_state import SessionStateKeys
from core.types import ExecutorConfig
from models.book import BookContext

DDDD_FOUND = {
    "primary_locations": [],
    "time_period": None,
    "themes": [],
    "matches_requested_book": False,
    "identified_title": "Dead Dead Demon's Dededede Destruction",
    "identified_author": "Inio Asano",
}


class TestVerdict:
    def test_a_different_named_book_is_a_mismatch_with_a_suggestion(self):
        v = check_book_identity(DDDD_FOUND, "dddd", "ddd")
        assert v.kind == "mismatch"
        assert v.suggested_title == "Dead Dead Demon's Dededede Destruction"
        assert v.suggested_author == "Inio Asano"

    def test_no_book_at_all_is_not_found(self):
        ctx = {"primary_locations": [], "themes": [], "matches_requested_book": False}
        v = check_book_identity(ctx, "Asdkfj", "Nobody")
        assert v.kind == "not_found"
        assert v.suggested_title is None

    def test_a_matching_book_passes(self):
        ctx = {
            "primary_locations": ["Bath"],
            "themes": ["love"],
            "matches_requested_book": True,
            "identified_title": "Persuasion",
            "identified_author": "Jane Austen",
        }
        assert check_book_identity(ctx, "Persuasion", "Jane Austen").kind == "ok"

    def test_silence_fails_open(self):
        # Pre-v5 payloads, the local-atmosphere flow and a null label all mean
        # "unknown" -- never block a real book the model forgot to label.
        legacy = {"primary_locations": ["Bath"], "themes": ["love"]}
        assert check_book_identity(legacy, "Persuasion", "Jane Austen").kind == "ok"
        assert check_book_identity(None, "Persuasion", "Jane Austen").kind == "ok"
        nulled = dict(legacy, matches_requested_book=None)
        assert check_book_identity(nulled, "Persuasion", "Jane Austen").kind == "ok"

    def test_a_false_label_naming_the_requested_book_is_overruled(self):
        # Same title, accents/case aside, author sharing a name token.
        ctx = {
            "primary_locations": ["Paris"],
            "themes": [],
            "matches_requested_book": False,
            "identified_title": "Les Misérables",
            "identified_author": "Victor Hugo",
        }
        assert check_book_identity(ctx, "les miserables", "Hugo").kind == "ok"

    def test_same_title_by_another_author_is_still_a_mismatch(self):
        ctx = {
            "primary_locations": [],
            "themes": [],
            "matches_requested_book": False,
            "identified_title": "Persuasion",
            "identified_author": "Someone Else",
        }
        v = check_book_identity(ctx, "Persuasion", "Jane Austen")
        assert v.kind == "mismatch"

    def test_an_oversized_suggestion_is_not_put_on_screen(self):
        ctx = dict(DDDD_FOUND, identified_title="x" * 500)
        v = check_book_identity(ctx, "dddd", "ddd")
        assert v.kind == "not_found"
        assert v.suggested_title is None


class TestPromptV5:
    def test_v5_is_v4_plus_the_book_context_change_only(self):
        v4 = json.loads(Path("agents/prompts/v4.json").read_text(encoding="utf-8"))
        v5 = json.loads(Path("agents/prompts/v5.json").read_text(encoding="utf-8"))
        changed = [a for a in v4["agents"] if v4["agents"][a] != v5["agents"][a]]
        assert changed == ["book_context_researcher", "book_context_formatter"], (
            "v5 must not quietly re-tune any other agent: it rides a cache flush"
        )
        assert v4["agents"].keys() == v5["agents"].keys()

    def test_v5_asks_for_the_identity_fields(self):
        prompts = load_prompts("v5")
        assert "BOOK IDENTITY" in prompts.book_context_researcher
        for field in ("matches_requested_book", "identified_title", "identified_author"):
            assert field in prompts.book_context_formatter
            assert field in BookContext.model_fields
        # The researcher template still formats with its two placeholders.
        prompts.book_context_researcher.format(book_ref="BOOK: x", search_hint="")


def _make_executor(monkeypatch, book_context, regions):
    """WorkflowExecutor whose discovery run writes the given state, no Gemini."""
    import core.executor as ex
    from core.executor import APP_NAME, WorkflowExecutor
    from services.session_service import create_session_service

    monkeypatch.setattr(
        ex,
        "create_book_to_place_discovery_workflow",
        lambda *a, **k: Workflow(name="stub_workflow"),
    )

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
            await self._session_service.append_event(
                session,
                Event(
                    invocation_id="system",
                    author="system",
                    actions=EventActions(
                        state_delta={
                            SessionStateKeys.BOOK_CONTEXT: book_context,
                            SessionStateKeys.REGION_ANALYSIS: {
                                "regions": regions,
                                "analysis_note": "note",
                            },
                        }
                    ),
                ),
            )
            if False:
                yield None

    monkeypatch.setattr(ex, "Runner", _FakeRunner)
    return WorkflowExecutor(
        config=ExecutorConfig(model_name="gemini-2.0-flash", google_api_key="test-key"),
        session_service=create_session_service(use_database=False),
        model=object(),
    )


TOKYO = [
    {
        "region_id": 1,
        "name": "Tokyo, Japan",
        "country_code": "JP",
        "primary_locality": "Tokyo",
        "cities": [{"name": "Tokyo", "country": "Japan"}],
    }
]


class TestDiscoverGuard:
    async def test_the_dddd_run_stops_with_did_you_mean(self, monkeypatch):
        executor = _make_executor(monkeypatch, DDDD_FOUND, TOKYO)
        events = [e async for e in executor.discover(book_title="dddd", author="ddd")]

        assert not any(isinstance(e, RegionsReady) for e in events), (
            "a trip for a different book must never reach the reader"
        )
        errors = [e for e in events if isinstance(e, WorkflowError)]
        assert len(errors) == 1
        err = errors[0]
        assert err.error_type == "BookNotFound"
        assert err.reason == "book_mismatch"
        assert err.suggested_title == "Dead Dead Demon's Dededede Destruction"
        assert err.suggested_author == "Inio Asano"
        assert "Did you mean" in err.message
        assert isinstance(events[-1], WorkflowComplete)

    async def test_no_such_book_stops_with_not_found(self, monkeypatch):
        ctx = {"primary_locations": [], "themes": [], "matches_requested_book": False}
        executor = _make_executor(monkeypatch, ctx, TOKYO)
        events = [e async for e in executor.discover(book_title="Asdkfj", author="Nobody")]

        errors = [e for e in events if isinstance(e, WorkflowError)]
        assert [e.reason for e in errors] == ["book_not_found"]
        assert errors[0].message == NOT_FOUND_MESSAGE
        assert errors[0].suggested_title is None

    async def test_a_mismatch_is_never_cached(self, monkeypatch):
        executor = _make_executor(monkeypatch, DDDD_FOUND, TOKYO)
        stored = []

        async def _set(key, value):
            stored.append(key)

        monkeypatch.setattr(executor._discovery_cache, "set", _set)
        _ = [e async for e in executor.discover(book_title="dddd", author="ddd")]
        assert stored == []

    async def test_the_requested_book_still_gets_its_regions(self, monkeypatch):
        ctx = {
            "primary_locations": ["Tokyo"],
            "themes": ["urban life"],
            "matches_requested_book": True,
            "identified_title": "Dead Dead Demon's Dededede Destruction",
            "identified_author": "Inio Asano",
        }
        executor = _make_executor(monkeypatch, ctx, TOKYO)
        events = [
            e
            async for e in executor.discover(
                book_title="Dead Dead Demon's Dededede Destruction", author="Inio Asano"
            )
        ]
        assert len([e for e in events if isinstance(e, RegionsReady)]) == 1
        assert not any(isinstance(e, WorkflowError) for e in events)


def test_the_suggestion_reaches_the_sse_error_event():
    from api.streaming import domain_event_to_sse

    sse = domain_event_to_sse(
        WorkflowError(
            message="Did you mean ...?",
            error_type="BookNotFound",
            reason="book_mismatch",
            suggested_title="Dead Dead Demon's Dededede Destruction",
            suggested_author="Inio Asano",
        )
    )
    data = json.loads(sse["data"])
    assert sse["event"] == "error"
    assert data["error_type"] == "BookNotFound"
    assert data["reason"] == "book_mismatch"
    assert data["suggested_title"] == "Dead Dead Demon's Dededede Destruction"
    assert data["suggested_author"] == "Inio Asano"
