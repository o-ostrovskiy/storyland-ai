"""Unit tests for the place→book reverse-routing capability (AI layer).

Covers the pure helpers (normalization, grounding filter, label invariants,
extraction), the agent/workflow factories, the candidate models, and the
PlaceToBookResolver end-to-end with the Gemini chain stubbed out (no network):
grounded path, literal/vibe labelling, fabricated-title drop, ungroundable
not-found state, and result caching.
"""

import pytest

from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.tools import FunctionTool
from google.genai import types

from agents import (
    create_place_to_book_agents,
    create_place_to_book_workflow,
)
from models.place_to_book import (
    PlaceBookCandidate,
    PlaceToBookCandidates,
    PlaceToBookResult,
)
from core.place_to_book import (
    PlaceToBookResolver,
    normalize_place,
    cache_key,
    validate_place_to_book_candidates,
    extract_place_to_book_from_state,
    filter_grounded_candidates,
    enforce_label_invariants,
)
from core.session_state import SessionStateAccessor, SessionStateKeys


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_google_search_tool():
    def mock_search(query: str) -> str:
        return "Search results for: " + query

    return FunctionTool(mock_search)


def _candidate(title, author="A", match_type="literal", maps_to="Somewhere",
               why="It fits.", description="A real book."):
    return {
        "title": title,
        "author": author,
        "description": description,
        "why_it_fits": why,
        "match_type": match_type,
        "maps_to": maps_to,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_candidates_allow_empty(self):
        """Ungroundable place → an empty candidate list is valid (not-found)."""
        result = PlaceToBookCandidates(candidates=[])
        assert result.candidates == []

    def test_candidate_round_trips(self):
        c = PlaceBookCandidate(**_candidate("The Book of Disquiet", match_type="literal",
                                            maps_to="Baixa, Lisbon"))
        assert c.match_type == "literal"
        assert c.maps_to == "Baixa, Lisbon"

    def test_invalid_match_type_rejected(self):
        with pytest.raises(Exception):
            PlaceBookCandidate(**_candidate("X", match_type="bogus"))

    def test_result_not_found_shape(self):
        r = PlaceToBookResult(place="Gotham", query="gotham", found=False,
                              message="We haven't mapped Gotham yet.", candidates=[])
        assert r.found is False
        assert r.candidates == []


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lower_and_strip(self):
        assert normalize_place("  Lisbon  ") == "lisbon"

    def test_strips_trailing_country(self):
        assert normalize_place("Lisbon, Portugal") == "lisbon"
        assert normalize_place("Dublin, Ireland") == "dublin"

    def test_keeps_non_country_comma_tail(self):
        # "Baixa, Lisbon" — Lisbon is not a country suffix, so keep as-is.
        assert normalize_place("Baixa, Lisbon") == "baixa, lisbon"

    def test_collapses_whitespace(self):
        assert normalize_place("the   scottish   highlands") == "the scottish highlands"

    def test_non_string_is_empty(self):
        assert normalize_place(None) == ""
        assert normalize_place(123) == ""

    def test_cache_key_versioned(self):
        assert cache_key("NYC") == "place2book:v1:nyc"


# ---------------------------------------------------------------------------
# Extraction / validation
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_validate_good_payload(self):
        payload = {"candidates": [_candidate("T1")]}
        out = validate_place_to_book_candidates(payload)
        assert isinstance(out, list) and len(out) == 1
        assert out[0]["title"] == "T1"

    def test_validate_empty_payload(self):
        assert validate_place_to_book_candidates({"candidates": []}) == []

    def test_validate_json_string(self):
        import json
        out = validate_place_to_book_candidates(json.dumps({"candidates": [_candidate("T")]}))
        assert len(out) == 1

    def test_validate_garbage(self):
        assert validate_place_to_book_candidates("not json") is None
        assert validate_place_to_book_candidates(42) is None

    def test_extract_from_state(self):
        state = SessionStateAccessor({SessionStateKeys.LAST_PLACE_TO_BOOK: {"candidates": [_candidate("T")]}})
        out = extract_place_to_book_from_state(state)
        assert out and out[0]["title"] == "T"

    def test_extract_missing_state(self):
        assert extract_place_to_book_from_state(SessionStateAccessor({})) is None


# ---------------------------------------------------------------------------
# Grounding filter
# ---------------------------------------------------------------------------

class TestGroundingFilter:
    def test_drops_ungrounded_title(self):
        cands = [_candidate("Real Title"), _candidate("Invented Title")]
        text = "The researcher discussed Real Title at length."
        out = filter_grounded_candidates(cands, text)
        assert [c["title"] for c in out] == ["Real Title"]

    def test_dropping_all_is_allowed(self):
        # Unlike book recs, an all-ungrounded result is a valid not-found signal.
        cands = [_candidate("Invented")]
        out = filter_grounded_candidates(cands, "totally unrelated text")
        assert out == []

    def test_fail_open_when_no_researcher_text(self):
        cands = [_candidate("Anything")]
        assert filter_grounded_candidates(cands, "") == cands

    def test_empty_candidates(self):
        assert filter_grounded_candidates([], "text") == []
        assert filter_grounded_candidates(None, "text") == []


# ---------------------------------------------------------------------------
# Label invariants
# ---------------------------------------------------------------------------

class TestLabelInvariants:
    def test_vibe_maps_to_forced_none(self):
        cands = [_candidate("V", match_type="vibe", maps_to="Wrongly Set")]
        out = enforce_label_invariants(cands)
        assert out[0]["maps_to"] is None

    def test_literal_without_location_dropped(self):
        cands = [_candidate("L", match_type="literal", maps_to=None)]
        assert enforce_label_invariants(cands) == []

    def test_literal_with_location_kept(self):
        cands = [_candidate("L", match_type="literal", maps_to="Lisbon")]
        out = enforce_label_invariants(cands)
        assert out[0]["maps_to"] == "Lisbon"

    def test_unknown_match_type_dropped(self):
        cands = [{"title": "X", "author": "A", "why_it_fits": "w", "match_type": "other"}]
        assert enforce_label_invariants(cands) == []


# ---------------------------------------------------------------------------
# Agent / workflow factories
# ---------------------------------------------------------------------------

class TestFactories:
    def test_agent_pair(self, mock_google_search_tool):
        researcher, formatter = create_place_to_book_agents(
            "gemini-2.0-flash", mock_google_search_tool, place="Lisbon"
        )
        assert researcher.name == "place_to_book_researcher"
        assert formatter.name == "place_to_book_formatter"

    def test_researcher_has_search_tool_formatter_does_not(self, mock_google_search_tool):
        researcher, formatter = create_place_to_book_agents(
            "gemini-2.0-flash", mock_google_search_tool, place="Tokyo"
        )
        assert isinstance(researcher, LlmAgent) and researcher.tools
        # Formatter stays tool-less: ADK 2.x allows tools + output_schema, but
        # the researcher/formatter separation is the ADR #2 anti-hallucination
        # contract, kept deliberately.
        assert not formatter.tools

    def test_workflow_is_researcher_to_formatter_graph(self, mock_google_search_tool):
        from google.adk.workflow import Workflow

        wf = create_place_to_book_workflow("gemini-2.0-flash", mock_google_search_tool, place="Dublin")
        assert isinstance(wf, Workflow)
        assert wf.name == "place_to_book_workflow"
        chain = [(e.from_node.name, e.to_node.name) for e in wf.graph.edges]
        assert chain == [
            ("__START__", "place_to_book_researcher"),
            ("place_to_book_researcher", "place_to_book_formatter"),
        ]


# ---------------------------------------------------------------------------
# Resolver (Gemini chain stubbed — no network)
# ---------------------------------------------------------------------------

def _make_resolver(monkeypatch, researcher_text, formatter_candidates, run_counter=None):
    """Build a PlaceToBookResolver whose pipeline run is stubbed.

    ``researcher_text`` is emitted as a place_to_book_researcher event; the
    ``formatter_candidates`` list is written to session state under the
    formatter output_key, exactly as the real run does.
    """
    import core.place_to_book as p2b
    from services.session_service import create_session_service

    monkeypatch.setattr(p2b, "create_place_to_book_workflow", lambda *a, **k: object())

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self._session_service = kwargs.get("session_service")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run_async(self, user_id, session_id, new_message):
            if run_counter is not None:
                run_counter.append(1)
            # Write the formatter output into session state.
            session = await self._session_service.get_session(
                app_name=p2b.APP_NAME, user_id=user_id, session_id=session_id
            )
            from google.adk.events.event_actions import EventActions
            ev_state = Event(
                invocation_id="system",
                author="system",
                actions=EventActions(
                    state_delta={
                        SessionStateKeys.LAST_PLACE_TO_BOOK: {
                            "candidates": formatter_candidates
                        }
                    }
                ),
            )
            await self._session_service.append_event(session, ev_state)
            # Emit a researcher event carrying the grounded text.
            yield Event(
                invocation_id="r",
                author="place_to_book_researcher",
                content=types.Content(role="model", parts=[types.Part(text=researcher_text)]),
            )

    monkeypatch.setattr(p2b, "Runner", _FakeRunner)

    return PlaceToBookResolver(
        model=object(),
        session_service=create_session_service(use_database=False),
    )


class TestResolver:
    async def test_grounded_path_labels(self, monkeypatch):
        researcher_text = (
            "Books set in Lisbon include The Book of Disquiet (Baixa, Lisbon). "
            "A vibe pick is The Shadow of the Wind, set in Barcelona."
        )
        cands = [
            _candidate("The Book of Disquiet", match_type="literal", maps_to="Baixa, Lisbon"),
            _candidate("The Shadow of the Wind", match_type="vibe", maps_to="Barcelona"),
        ]
        resolver = _make_resolver(monkeypatch, researcher_text, cands)
        result = await resolver.resolve("Lisbon")

        assert result.found is True
        assert result.query == "lisbon"
        assert len(result.candidates) == 2
        literal = [c for c in result.candidates if c.match_type == "literal"][0]
        vibe = [c for c in result.candidates if c.match_type == "vibe"][0]
        assert literal.maps_to == "Baixa, Lisbon"
        assert vibe.maps_to is None  # vibe must never claim a setting

    async def test_fabricated_title_dropped_to_not_found(self, monkeypatch):
        # Formatter invents a title the researcher never mentioned.
        researcher_text = "The researcher found nothing solid for this place."
        cands = [_candidate("Totally Invented Novel", match_type="literal", maps_to="Nowhere")]
        resolver = _make_resolver(monkeypatch, researcher_text, cands)
        result = await resolver.resolve("Atlantis")

        assert result.found is False
        assert result.candidates == []
        assert "Atlantis" in result.message

    async def test_ungroundable_empty_candidates(self, monkeypatch):
        resolver = _make_resolver(monkeypatch, "no grounded books here", [])
        result = await resolver.resolve("Wakanda")
        assert result.found is False
        assert result.candidates == []

    async def test_literal_without_location_dropped(self, monkeypatch):
        researcher_text = "Mentions A Real Sounding Book somewhere."
        cands = [_candidate("A Real Sounding Book", match_type="literal", maps_to=None)]
        resolver = _make_resolver(monkeypatch, researcher_text, cands)
        result = await resolver.resolve("Somewhere")
        assert result.found is False

    async def test_cache_hit_skips_second_run(self, monkeypatch):
        researcher_text = "Persuasion is set in Bath."
        cands = [_candidate("Persuasion", match_type="literal", maps_to="Bath")]
        runs = []
        resolver = _make_resolver(monkeypatch, researcher_text, cands, run_counter=runs)

        first = await resolver.resolve("Bath")
        second = await resolver.resolve("Bath, England")  # normalizes to "bath"

        assert first.found and second.found
        assert len(runs) == 1  # second call served from cache
        assert first.query == second.query == "bath"

    async def test_blank_place_short_circuits(self, monkeypatch):
        runs = []
        resolver = _make_resolver(monkeypatch, "x", [_candidate("Y")], run_counter=runs)
        result = await resolver.resolve("   ")
        assert result.found is False
        assert runs == []  # never ran the pipeline
