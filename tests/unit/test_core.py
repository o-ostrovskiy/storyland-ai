"""
Unit tests for the core SDK layer.

Tests domain events, session state, prompts, extraction, and regions.
"""

import json
import pytest
from unittest.mock import MagicMock

from core.events import (
    Phase,
    ProgressEvent,
    JobStarted,
    MetadataReady,
    RegionsReady,
    ItineraryReady,
    BookRecommendationsReady,
    WorkflowError,
    WorkflowComplete,
)
from core.types import ExecutorConfig
from core.session_state import SessionStateKeys, SessionStateAccessor
from core.prompts import (
    build_discovery_prompt,
    build_composition_prompt,
    build_local_atmosphere_prompt,
)
from core.extraction import (
    validate_trip_itinerary,
    validate_composer_envelope,
    validate_expansion_result,
    validate_book_recommendations_result,
    extract_json_from_text,
    extract_itinerary_from_response,
    extract_expansion_from_state,
    extract_book_recommendations_from_state,
    downgrade_ungrounded_match_types,
    reconcile_stop_city_grouping,
    _address_locality,
    grounding_token_set,
    is_title_grounded,
)
from core.events import ExpansionReady
from core.executor import WorkflowExecutor
from core.regions import get_valid_region_ids, validate_region_selection
from evaluation.tools.run_scheduled_eval import select_first_region, select_all_regions


# =============================================================================
# Domain Events
# =============================================================================


class TestPhase:
    def test_phase_values(self):
        assert Phase.BOOK_SEARCH == 1
        assert Phase.DISCOVERY == 2
        assert Phase.COMPOSITION == 3


class TestProgressEvent:
    def test_construction(self):
        event = ProgressEvent(phase=Phase.DISCOVERY, step="Finding cities")
        assert event.phase == Phase.DISCOVERY
        assert event.step == "Finding cities"
        assert event.detail is None

    def test_with_detail(self):
        event = ProgressEvent(
            phase=Phase.DISCOVERY, step="Finding cities", detail="city_pipeline"
        )
        assert event.detail == "city_pipeline"

    def test_frozen(self):
        event = ProgressEvent(phase=Phase.DISCOVERY, step="test")
        with pytest.raises(AttributeError):
            event.step = "modified"


class TestJobStarted:
    def test_construction(self):
        event = JobStarted(job_id="abc-123")
        assert event.job_id == "abc-123"

    def test_frozen(self):
        event = JobStarted(job_id="abc-123")
        with pytest.raises(AttributeError):
            event.job_id = "other"


class TestMetadataReady:
    def test_construction(self):
        metadata = {"book_title": "1984", "author": "George Orwell"}
        event = MetadataReady(metadata=metadata)
        assert event.metadata["book_title"] == "1984"


class TestRegionsReady:
    def test_construction(self):
        regions = [{"region_id": 1, "region_name": "England"}]
        event = RegionsReady(
            job_id="abc-123", regions=regions, analysis_note="test"
        )
        assert event.job_id == "abc-123"
        assert len(event.regions) == 1


class TestItineraryReady:
    def test_construction(self):
        itinerary = {"cities": [], "summary_text": "test"}
        event = ItineraryReady(itinerary=itinerary)
        assert event.itinerary["summary_text"] == "test"


class TestWorkflowError:
    def test_without_phase(self):
        event = WorkflowError(message="fail", error_type="TestError")
        assert event.phase is None

    def test_with_phase(self):
        event = WorkflowError(
            message="timeout", error_type="Timeout", phase=Phase.DISCOVERY
        )
        assert event.phase == Phase.DISCOVERY


class TestWorkflowComplete:
    def test_without_token_usage(self):
        event = WorkflowComplete(job_id="abc")
        assert event.token_usage is None

    def test_with_token_usage(self):
        usage = {"input_tokens": 100, "output_tokens": 50}
        event = WorkflowComplete(job_id="abc", token_usage=usage)
        assert event.token_usage["input_tokens"] == 100


class TestPatternMatching:
    """Verify events work with structural pattern matching."""

    def test_match_progress(self):
        event = ProgressEvent(phase=Phase.DISCOVERY, step="test")
        match event:
            case ProgressEvent(phase=p, step=s):
                assert p == Phase.DISCOVERY
                assert s == "test"
            case _:
                pytest.fail("Should match ProgressEvent")

    def test_match_error(self):
        event = WorkflowError(
            message="fail", error_type="TestError", phase=Phase.BOOK_SEARCH
        )
        match event:
            case WorkflowError(message=m, phase=p):
                assert m == "fail"
                assert p == Phase.BOOK_SEARCH
            case _:
                pytest.fail("Should match WorkflowError")


# =============================================================================
# ExecutorConfig
# =============================================================================


class TestExecutorConfig:
    def test_defaults(self):
        config = ExecutorConfig(
            model_name="gemini-2.0-flash", google_api_key="test-key"
        )
        assert config.workflow_timeout == 300
        assert config.use_database is False
        assert config.langfuse_secret_key is None

    def test_from_config(self):
        mock_config = MagicMock()
        mock_config.model_name = "gemini-2.0-flash"
        mock_config.google_api_key = "key"
        mock_config.workflow_timeout = 600
        mock_config.database_url = "sqlite:///test.db"
        mock_config.use_database = True
        mock_config.langfuse_secret_key = "sk"
        mock_config.langfuse_public_key = "pk"
        mock_config.langfuse_host = "https://langfuse.example.com"

        executor_config = ExecutorConfig.from_config(mock_config)
        assert executor_config.model_name == "gemini-2.0-flash"
        assert executor_config.workflow_timeout == 600
        assert executor_config.use_database is True


# =============================================================================
# Session State
# =============================================================================


class TestSessionStateKeys:
    def test_key_constants(self):
        assert SessionStateKeys.BOOK_METADATA == "book_metadata"
        assert SessionStateKeys.REGION_ANALYSIS == "region_analysis"
        assert SessionStateKeys.FINAL_ITINERARY == "final_itinerary"
        assert SessionStateKeys.USER_PREFERENCES == "user:preferences"
        assert SessionStateKeys.USER_LOCATION == "user_location"


class TestSessionStateAccessor:
    def test_empty_state(self):
        accessor = SessionStateAccessor({})
        assert accessor.book_metadata is None
        assert accessor.region_analysis is None
        assert accessor.regions == []
        assert accessor.selected_regions == []
        assert accessor.final_itinerary is None
        assert accessor.book_title == ""
        assert accessor.author == ""

    def test_read_book_metadata(self):
        state = {"book_metadata": {"book_title": "1984", "author": "George Orwell"}}
        accessor = SessionStateAccessor(state)
        assert accessor.book_title == "1984"
        assert accessor.author == "George Orwell"

    def test_read_regions(self):
        state = {
            "region_analysis": {
                "regions": [{"region_id": 1}, {"region_id": 2}],
                "analysis_note": "test note",
            }
        }
        accessor = SessionStateAccessor(state)
        assert len(accessor.regions) == 2
        assert accessor.analysis_note == "test note"

    def test_read_final_itinerary(self):
        itinerary = {"cities": [], "summary_text": "test"}
        state = {"final_itinerary": itinerary}
        accessor = SessionStateAccessor(state)
        assert accessor.final_itinerary == itinerary

    def test_failed_defaults_false(self):
        accessor = SessionStateAccessor({})
        assert accessor.failed is False

    def test_failed_reads_true_when_set_on_the_underlying_dict(self):
        state = {"job_failed": True}
        accessor = SessionStateAccessor(state)
        assert accessor.failed is True

    # MYS-172: the accessor is deliberately read-only -- writing through it
    # used to be a silent no-op against persisted ADK session state (two
    # `core/executor.py` call sites did exactly this). These regression
    # tests pin that the setters/clear method are GONE, not merely unused,
    # so a future re-add doesn't quietly reintroduce the trap.
    def test_book_metadata_has_no_setter(self):
        accessor = SessionStateAccessor({})
        with pytest.raises(AttributeError):
            accessor.book_metadata = {"book_title": "1984"}

    def test_selected_regions_has_no_setter(self):
        accessor = SessionStateAccessor({})
        with pytest.raises(AttributeError):
            accessor.selected_regions = [{"region_id": 1}]

    def test_failed_has_no_setter(self):
        accessor = SessionStateAccessor({})
        with pytest.raises(AttributeError):
            accessor.failed = True

    def test_clear_final_itinerary_method_is_gone(self):
        accessor = SessionStateAccessor({"final_itinerary": {"cities": []}})
        assert not hasattr(accessor, "clear_final_itinerary")


# =============================================================================
# Prompts
# =============================================================================


class TestPrompts:
    def test_discovery_prompt(self):
        prompt = build_discovery_prompt("1984", "George Orwell")
        assert '"1984"' in prompt
        assert "George Orwell" in prompt
        assert "cities" in prompt.lower()

    def test_discovery_prompt_vibe_absent_is_byte_identical(self):
        # The whole point of the optional vibe: omitting it must not change the
        # prompt at all (no behavior drift for existing callers).
        assert build_discovery_prompt("1984", "George Orwell", None) == (
            build_discovery_prompt("1984", "George Orwell")
        )

    def test_discovery_prompt_vibe_present_biases_and_names_mood(self):
        prompt = build_discovery_prompt("1984", "George Orwell", "melancholic")
        # Still contains the base discovery instruction...
        assert '"1984"' in prompt
        assert "cities" in prompt.lower()
        # ...plus the mood bias, named explicitly for the "why this fits" copy.
        assert "melancholic" in prompt
        # ...and an explicit grounding-wins guard so vibe can't invent links.
        assert "grounding" in prompt.lower()

    def test_discovery_prompt_taste_absent_is_byte_identical(self):
        # Omitting taste_context must not change the prompt at all.
        assert build_discovery_prompt("1984", "George Orwell", None, None) == (
            build_discovery_prompt("1984", "George Orwell")
        )

    def test_discovery_prompt_empty_taste_is_byte_identical(self):
        # A structurally-empty taste block is treated as absent.
        assert build_discovery_prompt(
            "1984", "George Orwell", None, {"titles": [], "moods": []}
        ) == build_discovery_prompt("1984", "George Orwell")

    def test_discovery_prompt_taste_present_biases_and_keeps_grounding(self):
        prompt = build_discovery_prompt(
            "1984",
            "George Orwell",
            None,
            {"titles": ["Wuthering Heights"], "moods": ["melancholic"]},
        )
        assert '"1984"' in prompt
        assert "cities" in prompt.lower()
        # The reader's titles + moods surface in the bias clause...
        assert "Wuthering Heights" in prompt
        assert "melancholic" in prompt
        assert "reading history" in prompt.lower()
        # ...with the same grounding-wins guard as vibe.
        assert "grounding" in prompt.lower()

    def test_discovery_prompt_vibe_and_taste_compose(self):
        # Both biases append independently; neither replaces the other.
        prompt = build_discovery_prompt(
            "1984",
            "George Orwell",
            "cozy",
            {"titles": ["Dune"], "moods": ["adventurous"]},
        )
        assert "cozy" in prompt
        assert "Dune" in prompt
        assert "adventurous" in prompt
        assert prompt.count("grounding") >= 2

    def test_composition_prompt(self):
        regions = [{"region_id": 1, "region_name": "England"}]
        prompt = build_composition_prompt("1984", "George Orwell", regions)
        assert '"1984"' in prompt
        assert "George Orwell" in prompt
        assert "England" in prompt
        assert "ONLY" in prompt

    def test_local_atmosphere_prompt(self):
        prompt = build_local_atmosphere_prompt(
            "Wuthering Heights", "Emily Brontë", "New York, NY 10013", 80
        )
        assert "Wuthering Heights" in prompt
        assert "Emily Brontë" in prompt
        assert "New York, NY 10013" in prompt
        assert "80" in prompt
        # The flow's defining instruction is to skip the actual setting.
        assert "atmospheric" in prompt.lower()


# =============================================================================
# Extraction
# =============================================================================


class TestValidateTripItinerary:
    def test_valid_itinerary(self):
        data = {
            "cities": [
                {
                    "name": "London",
                    "country": "UK",
                    "days_suggested": 2,
                    "overview": "test",
                    "stops": [
                        {
                            "name": "British Library",
                            "type": "museum",
                            "reason": "test",
                            "time_of_day": "morning",
                            "notes": "test",
                        }
                    ],
                }
            ],
            "summary_text": "A journey",
        }
        result = validate_trip_itinerary(data)
        assert result is not None
        assert result["cities"][0]["name"] == "London"

    def test_invalid_itinerary(self):
        data = {"cities": [{"name": "London"}]}  # missing required fields
        result = validate_trip_itinerary(data)
        assert result is None

    def test_string_input(self):
        data = json.dumps(
            {
                "cities": [
                    {
                        "name": "London",
                        "country": "UK",
                        "days_suggested": 2,
                        "overview": "test",
                        "stops": [],
                    }
                ],
                "summary_text": "test",
            }
        )
        result = validate_trip_itinerary(data)
        assert result is not None

    def test_none_input(self):
        assert validate_trip_itinerary(None) is None

    def test_invalid_json_string(self):
        assert validate_trip_itinerary("not json") is None


class TestExtractJsonFromText:
    def test_json_in_text(self):
        text = 'Here is the result: {"key": "value"} done.'
        result = extract_json_from_text(text)
        assert result == {"key": "value"}

    def test_no_json(self):
        assert extract_json_from_text("no json here") is None

    def test_invalid_json(self):
        assert extract_json_from_text("{not valid json}") is None

    def test_nested_json(self):
        text = '{"outer": {"inner": 1}}'
        result = extract_json_from_text(text)
        assert result["outer"]["inner"] == 1


_VALID_ITINERARY = {
    "cities": [
        {"name": "London", "country": "UK", "days_suggested": 2, "overview": "test", "stops": []}
    ],
    "summary_text": "test",
}

_VALID_CHIP = {"id": "x", "label": "Add restaurants", "action_prompt": "Find atmospheric restaurants."}


class TestExtractItineraryFromResponse:
    def test_from_envelope_in_state(self):
        envelope = {"itinerary": _VALID_ITINERARY, "suggestions": [_VALID_CHIP]}
        accessor = SessionStateAccessor({"composer_envelope": envelope})
        result = extract_itinerary_from_response(None, accessor)
        assert result is not None
        itinerary, suggestions = result
        assert itinerary["cities"][0]["name"] == "London"
        assert len(suggestions) == 1
        assert suggestions[0]["label"] == "Add restaurants"

    def test_from_bare_itinerary_state_fallback(self):
        accessor = SessionStateAccessor({"final_itinerary": _VALID_ITINERARY})
        result = extract_itinerary_from_response(None, accessor)
        assert result is not None
        itinerary, suggestions = result
        assert itinerary["cities"][0]["name"] == "London"
        assert suggestions == []

    def test_from_text_fallback(self):
        accessor = SessionStateAccessor({})

        itinerary_json = json.dumps(_VALID_ITINERARY)
        mock_part = MagicMock()
        mock_part.text = f"Result: {itinerary_json}"
        mock_response = MagicMock()
        mock_response.content.parts = [mock_part]

        result = extract_itinerary_from_response(mock_response, accessor)
        assert result is not None
        itinerary, suggestions = result
        assert suggestions == []

    def test_no_itinerary_anywhere(self):
        accessor = SessionStateAccessor({})
        mock_part = MagicMock()
        mock_part.text = "Sorry, I cannot do that."
        mock_response = MagicMock()
        mock_response.content.parts = [mock_part]

        result = extract_itinerary_from_response(mock_response, accessor)
        assert result is None


class TestValidateComposerEnvelope:
    def test_valid_envelope(self):
        data = {"itinerary": _VALID_ITINERARY, "suggestions": [_VALID_CHIP]}
        result = validate_composer_envelope(data)
        assert result is not None
        itinerary, suggestions = result
        assert itinerary["summary_text"] == "test"
        assert len(suggestions) == 1

    def test_envelope_no_suggestions(self):
        data = {"itinerary": _VALID_ITINERARY}
        result = validate_composer_envelope(data)
        assert result is not None
        _, suggestions = result
        assert suggestions == []

    def test_invalid_envelope_missing_itinerary(self):
        data = {"suggestions": [_VALID_CHIP]}
        assert validate_composer_envelope(data) is None

    def test_invalid_input(self):
        assert validate_composer_envelope(None) is None
        assert validate_composer_envelope("not json") is None
        assert validate_composer_envelope({"bad": "data"}) is None


class TestValidateExpansionResult:
    def _stop(self, name="The Ritz"):
        return {"name": name, "type": "restaurant", "reason": "Mood", "time_of_day": "evening", "source": "expansion"}

    def test_valid_result(self):
        data = {"parent_city": "London", "places": [self._stop()], "suggestions": [_VALID_CHIP]}
        result = validate_expansion_result(data)
        assert result is not None
        assert result["parent_city"] == "London"
        assert len(result["places"]) == 1

    def test_result_missing_parent_city(self):
        data = {"places": [self._stop()]}
        assert validate_expansion_result(data) is None

    def test_result_missing_places(self):
        data = {"parent_city": "London"}
        assert validate_expansion_result(data) is None

    def test_result_none(self):
        assert validate_expansion_result(None) is None


class TestExtractExpansionFromState:
    def _stop(self):
        return {"name": "Brasserie Zédel", "type": "restaurant", "reason": "Mood", "time_of_day": "evening", "source": "expansion"}

    def test_valid_expansion_in_state(self):
        data = {"parent_city": "London", "places": [self._stop()]}
        accessor = SessionStateAccessor({"last_expansion": data})
        result = extract_expansion_from_state(accessor)
        assert result is not None
        assert result["parent_city"] == "London"

    def test_no_expansion_in_state(self):
        accessor = SessionStateAccessor({})
        assert extract_expansion_from_state(accessor) is None


class TestResolveTrustedActionPrompt:
    """MYS-167: expand() must resolve the expansion instruction from the
    server-stored chip matching action_id, never from the client-echoed
    action_prompt request field -- a caller holding one valid chip id must
    not be able to steer the researcher's system instruction with an
    arbitrary string. Tested as a pure staticmethod: no session service,
    Runner, or LLM agent construction needed to prove the resolution logic
    itself is correct.
    """

    def test_resolves_the_stored_prompt_for_a_matching_chip(self):
        last_suggestions = [
            {"id": "chip-1", "label": "Add restaurants", "action_prompt": "Find atmospheric restaurants near the stops."},
            {"id": "chip-2", "label": "More cafés", "action_prompt": "Find cafés matching the mood."},
        ]
        resolved = WorkflowExecutor._resolve_trusted_action_prompt(last_suggestions, "chip-2")
        assert resolved == "Find cafés matching the mood."

    def test_ignores_a_client_supplied_string_entirely(self):
        """The injection case: a caller sends a valid action_id alongside an
        attacker-controlled action_prompt string. The resolver never even
        receives that argument -- it can't leak into the result no matter
        what the client sent alongside the id."""
        last_suggestions = [
            {"id": "chip-1", "label": "Add restaurants", "action_prompt": "Find atmospheric restaurants near the stops."},
        ]
        resolved = WorkflowExecutor._resolve_trusted_action_prompt(last_suggestions, "chip-1")
        assert resolved == "Find atmospheric restaurants near the stops."
        assert "Ignore previous instructions" not in resolved

    def test_unmatched_action_id_resolves_to_empty_string_not_an_error(self):
        """expand() only reaches this after action_id passed the valid_ids
        check, so this path isn't reachable in practice -- but the resolver
        itself must fail closed (empty string) rather than raise or fall
        back to any caller-supplied value."""
        last_suggestions = [
            {"id": "chip-1", "label": "Add restaurants", "action_prompt": "Find restaurants."},
        ]
        resolved = WorkflowExecutor._resolve_trusted_action_prompt(last_suggestions, "no-such-id")
        assert resolved == ""

    def test_empty_last_suggestions_resolves_to_empty_string(self):
        assert WorkflowExecutor._resolve_trusted_action_prompt([], "chip-1") == ""

    def test_chip_missing_action_prompt_key_resolves_to_empty_string(self):
        """The 'Find books like this' chip shape stores action_prompt="" (and
        is kept out of last_suggestions entirely per _build_book_recommendation_chip's
        docstring) -- but defensively, a chip dict with no key at all must not
        raise either."""
        last_suggestions = [{"id": "chip-1", "label": "No prompt field"}]
        resolved = WorkflowExecutor._resolve_trusted_action_prompt(last_suggestions, "chip-1")
        assert resolved == ""

    def test_duplicate_ids_resolves_the_first_match(self):
        """Chip ids are server-stamped uuid4s (_stamp_suggestion_ids) so this
        shouldn't occur in practice; pin deterministic first-match behavior
        rather than leaving it as an accident of `next()`."""
        last_suggestions = [
            {"id": "chip-1", "label": "First", "action_prompt": "First prompt."},
            {"id": "chip-1", "label": "Second", "action_prompt": "Second prompt."},
        ]
        resolved = WorkflowExecutor._resolve_trusted_action_prompt(last_suggestions, "chip-1")
        assert resolved == "First prompt."


class TestExpansionReadyEvent:
    def test_construction(self):
        event = ExpansionReady(
            parent_city="London",
            places=[{"name": "Café A"}],
            suggestions=[{"id": "x", "label": "More cafés", "action_prompt": "Find more."}]
        )
        assert event.parent_city == "London"
        assert len(event.places) == 1
        assert len(event.suggestions) == 1

    def test_default_suggestions(self):
        event = ExpansionReady(parent_city="Paris", places=[])
        assert event.suggestions == []

    def test_frozen(self):
        event = ExpansionReady(parent_city="Rome", places=[])
        with pytest.raises(AttributeError):
            event.parent_city = "Milan"


# =============================================================================
# Regions
# =============================================================================


class TestGetValidRegionIds:
    def test_valid_ids(self):
        regions = [
            {"region_id": 1, "region_name": "England"},
            {"region_id": 2, "region_name": "Scotland"},
        ]
        assert get_valid_region_ids(regions) == {1, 2}

    def test_filters_none(self):
        regions = [
            {"region_id": 1},
            {"region_id": None},
            {"region_name": "No ID"},
        ]
        assert get_valid_region_ids(regions) == {1}

    def test_empty_list(self):
        assert get_valid_region_ids([]) == set()

    def test_filters_non_int(self):
        regions = [{"region_id": "abc"}, {"region_id": 1}]
        assert get_valid_region_ids(regions) == {1}


class TestValidateRegionSelection:
    def test_valid_selection(self):
        all_regions = [
            {"region_id": 1, "region_name": "England"},
            {"region_id": 2, "region_name": "Scotland"},
        ]
        selected, invalid = validate_region_selection([1], all_regions)
        assert len(selected) == 1
        assert selected[0]["region_name"] == "England"
        assert invalid == []

    def test_multiple_selection(self):
        all_regions = [
            {"region_id": 1, "region_name": "England"},
            {"region_id": 2, "region_name": "Scotland"},
        ]
        selected, invalid = validate_region_selection([1, 2], all_regions)
        assert len(selected) == 2
        assert invalid == []

    def test_invalid_ids(self):
        all_regions = [{"region_id": 1, "region_name": "England"}]
        selected, invalid = validate_region_selection([99], all_regions)
        assert selected == []
        assert invalid == [99]

    def test_mixed_valid_invalid(self):
        all_regions = [
            {"region_id": 1, "region_name": "England"},
            {"region_id": 2, "region_name": "Scotland"},
        ]
        selected, invalid = validate_region_selection([1, 99], all_regions)
        assert selected == []
        assert invalid == [99]


# =============================================================================
# Eval Runner Region Selection
# =============================================================================


class TestSelectFirstRegion:
    def test_returns_first_region(self):
        region_analysis = {
            "regions": [
                {"region_id": 1, "region_name": "England"},
                {"region_id": 2, "region_name": "Scotland"},
                {"region_id": 3, "region_name": "Wales"},
            ]
        }
        result = select_first_region(region_analysis)
        assert len(result) == 1
        assert result[0]["region_name"] == "England"

    def test_empty_regions(self):
        assert select_first_region({"regions": []}) == []

    def test_missing_regions_key(self):
        assert select_first_region({}) == []


class TestSelectAllRegions:
    def test_returns_all_regions(self):
        regions = [
            {"region_id": 1, "region_name": "England"},
            {"region_id": 2, "region_name": "Scotland"},
            {"region_id": 3, "region_name": "Wales"},
        ]
        result = select_all_regions({"regions": regions})
        assert result == regions

    def test_single_region(self):
        regions = [{"region_id": 1, "region_name": "England"}]
        result = select_all_regions({"regions": regions})
        assert len(result) == 1

    def test_empty_regions(self):
        assert select_all_regions({"regions": []}) == []

    def test_missing_regions_key(self):
        assert select_all_regions({}) == []


# =============================================================================
# BookRecommendationsReady Event Tests
# =============================================================================


class TestBookRecommendationsReady:
    def test_construction(self):
        event = BookRecommendationsReady(
            recommendations=[{"title": "Les Misérables", "author": "Victor Hugo"}],
            book_recommendation_count=1,
        )
        assert len(event.recommendations) == 1
        assert event.book_recommendation_count == 1

    def test_frozen(self):
        event = BookRecommendationsReady(recommendations=[], book_recommendation_count=0)
        with pytest.raises(AttributeError):
            event.book_recommendation_count = 2

    def test_empty_recommendations(self):
        event = BookRecommendationsReady(recommendations=[], book_recommendation_count=0)
        assert event.recommendations == []


# =============================================================================
# Session State - Book Recommendations Keys
# =============================================================================


class TestBookRecommendationSessionState:
    def test_new_keys_exist(self):
        from core.session_state import SessionStateKeys
        assert hasattr(SessionStateKeys, "LAST_BOOK_RECOMMENDATIONS")
        assert hasattr(SessionStateKeys, "BOOK_RECOMMENDATION_COUNT")
        assert hasattr(SessionStateKeys, "BOOK_RECS_IN_PROGRESS")
        assert hasattr(SessionStateKeys, "BOOK_RECOMMENDATION_CHIP_ID")
        assert hasattr(SessionStateKeys, "BOOK_RECOMMENDATION_CHIP")

    def test_accessor_defaults(self):
        accessor = SessionStateAccessor({})
        assert accessor.book_recommendation_count == 0
        assert accessor.book_recs_in_progress is False
        assert accessor.last_book_recommendations is None
        assert accessor.book_recommendation_chip_id is None
        assert accessor.book_recommendation_chip is None
        assert accessor.book_context is None

    def test_accessor_reads_state(self):
        chip_id = "test-chip-uuid"
        chip = {"id": chip_id, "label": "Find books like this", "action_prompt": ""}
        state = {
            "book_recommendation_count": 3,
            "book_recs_in_progress": True,
            "book_recommendation_chip_id": chip_id,
            "book_recommendation_chip": chip,
            "book_context": {"themes": ["mystery"], "primary_locations": ["Paris"]},
            "last_book_recommendations": {"recommendations": []},
        }
        accessor = SessionStateAccessor(state)
        assert accessor.book_recommendation_count == 3
        assert accessor.book_recs_in_progress is True
        assert accessor.book_recommendation_chip_id == chip_id
        assert accessor.book_recommendation_chip == chip
        assert accessor.book_context["themes"] == ["mystery"]
        assert accessor.last_book_recommendations == {"recommendations": []}


# =============================================================================
# Extraction - Book Recommendations
# =============================================================================


class TestValidateBookRecommendationsResult:
    def _make_rec(self, title="Book"):
        return {
            "title": title,
            "author": "Author",
            "reason": "A reason.",
            "recommendation_basis": "themes",
        }

    def test_valid_payload(self):
        data = {"recommendations": [self._make_rec() for _ in range(5)]}
        result = validate_book_recommendations_result(data)
        assert result is not None
        assert len(result["recommendations"]) == 5

    def test_json_string_input(self):
        import json
        data = json.dumps({"recommendations": [self._make_rec(f"Book {i}") for i in range(5)]})
        result = validate_book_recommendations_result(data)
        assert result is not None

    def test_invalid_basis_returns_none(self):
        recs = [self._make_rec(f"Book {i}") for i in range(4)]
        recs.append({"title": "X", "author": "Y", "reason": "R", "recommendation_basis": "bad_value"})
        assert validate_book_recommendations_result({"recommendations": recs}) is None

    def test_floor_three_accepted(self):
        """Three recommendations now satisfy the relaxed floor (was hard 5)."""
        data = {"recommendations": [self._make_rec(f"Book {i}") for i in range(3)]}
        result = validate_book_recommendations_result(data)
        assert result is not None
        assert len(result["recommendations"]) == 3

    def test_below_floor_rejected(self):
        """Fewer than the floor (default 3) is still not a successful response."""
        data = {"recommendations": [self._make_rec(f"Book {i}") for i in range(2)]}
        assert validate_book_recommendations_result(data) is None

    def test_too_many_recommendations_rejected(self):
        data = {"recommendations": [self._make_rec(f"Book {i}") for i in range(6)]}
        assert validate_book_recommendations_result(data) is None

    def test_none_input_returns_none(self):
        assert validate_book_recommendations_result(None) is None

    def test_empty_string_returns_none(self):
        assert validate_book_recommendations_result("not-json") is None


class TestExtractBookRecommendationsFromState:
    def _make_rec(self, title="Book"):
        return {
            "title": title,
            "author": "Author",
            "reason": "A reason.",
            "recommendation_basis": "destination",
        }

    def test_extracts_from_state(self):
        recs = [self._make_rec(f"Book {i}") for i in range(5)]
        recs[0]["title"] = "Moby Dick"
        accessor = SessionStateAccessor({"last_book_recommendations": {"recommendations": recs}})
        result = extract_book_recommendations_from_state(accessor)
        assert result is not None
        assert result["recommendations"][0]["title"] == "Moby Dick"
        assert len(result["recommendations"]) == 5

    def test_returns_none_when_missing(self):
        accessor = SessionStateAccessor({})
        assert extract_book_recommendations_from_state(accessor) is None

    def test_returns_none_for_invalid_data(self):
        accessor = SessionStateAccessor({"last_book_recommendations": {"bad": "data"}})
        assert extract_book_recommendations_from_state(accessor) is None


# =============================================================================
# Hallucination guardrail — match_type grounding assertion (PR 2 of 4)
# =============================================================================


def _itinerary_with_stops(stops):
    return {
        "cities": [
            {
                "name": "Atlanta",
                "country": "United States",
                "days_suggested": 2,
                "overview": "test",
                "stops": stops,
            }
        ],
        "summary_text": "test",
    }


def _stop(name, match_type, grounding_source=None):
    return {
        "name": name,
        "type": "landmark",
        "reason": "test",
        "time_of_day": "morning",
        "source": "composed",
        "match_type": match_type,
        "grounding_source": grounding_source,
    }


# Grounded research that names a real, sourced place but NOT the fabricated one.
_RESEARCH = "Margaret Mitchell House is the author's home in Atlanta where she wrote the novel."


class TestDowngradeUngroundedMatchTypes:
    def test_ungrounded_literal_downgraded_to_vibe(self):
        itin = _itinerary_with_stops(
            [_stop("Faulkner Invented Manor", "literal", grounding_source="chapter 3")]
        )
        out = downgrade_ungrounded_match_types(itin, _RESEARCH)
        stop = out["cities"][0]["stops"][0]
        assert stop["match_type"] == "vibe"
        # the cited source did not hold up -> cleared
        assert stop["grounding_source"] is None

    def test_grounded_literal_preserved(self):
        itin = _itinerary_with_stops(
            [_stop("Margaret Mitchell House", "literal", grounding_source="author's home")]
        )
        out = downgrade_ungrounded_match_types(itin, _RESEARCH)
        stop = out["cities"][0]["stops"][0]
        assert stop["match_type"] == "literal"
        assert stop["grounding_source"] == "author's home"

    def test_ungrounded_historical_downgraded(self):
        itin = _itinerary_with_stops([_stop("Nonexistent Battlefield", "historical")])
        out = downgrade_ungrounded_match_types(itin, _RESEARCH)
        assert out["cities"][0]["stops"][0]["match_type"] == "vibe"

    def test_thematic_and_vibe_untouched(self):
        itin = _itinerary_with_stops(
            [_stop("Some Atmospheric Cafe", "thematic"), _stop("A Moody Park", "vibe")]
        )
        out = downgrade_ungrounded_match_types(itin, _RESEARCH)
        assert out["cities"][0]["stops"][0]["match_type"] == "thematic"
        assert out["cities"][0]["stops"][1]["match_type"] == "vibe"

    def test_never_drops_stops(self):
        itin = _itinerary_with_stops(
            [
                _stop("Margaret Mitchell House", "literal"),
                _stop("Invented Place", "literal"),
                _stop("Mood Cafe", "vibe"),
            ]
        )
        out = downgrade_ungrounded_match_types(itin, _RESEARCH)
        assert len(out["cities"][0]["stops"]) == 3

    def test_fail_open_when_no_research(self):
        itin = _itinerary_with_stops([_stop("Invented Place", "literal", grounding_source="x")])
        out = downgrade_ungrounded_match_types(itin, "")
        stop = out["cities"][0]["stops"][0]
        # cannot prove ungrounded -> labels untouched
        assert stop["match_type"] == "literal"
        assert stop["grounding_source"] == "x"

    def test_matching_is_case_insensitive(self):
        itin = _itinerary_with_stops([_stop("margaret mitchell HOUSE", "literal")])
        out = downgrade_ungrounded_match_types(itin, _RESEARCH)
        assert out["cities"][0]["stops"][0]["match_type"] == "literal"

    def test_none_itinerary_returns_none(self):
        assert downgrade_ungrounded_match_types(None, _RESEARCH) is None

    def test_short_generic_name_no_substring_false_positive(self):
        # "The Mill" used to substring-match "the millionaire" in unrelated text.
        itin = _itinerary_with_stops([_stop("The Mill", "literal", grounding_source="x")])
        research = "A tour of the millionaire's mansion district in Atlanta."
        out = downgrade_ungrounded_match_types(itin, research)
        stop = out["cities"][0]["stops"][0]
        assert stop["match_type"] == "vibe"
        assert stop["grounding_source"] is None

    def test_surface_variant_name_still_grounded(self):
        # One missing token ("Grand") must not downgrade a grounded stop.
        itin = _itinerary_with_stops(
            [_stop("The Grand Pump Room", "literal", grounding_source="ch. 2")]
        )
        research = "Austen's characters take the waters at the Pump Room in Bath."
        out = downgrade_ungrounded_match_types(itin, research)
        stop = out["cities"][0]["stops"][0]
        assert stop["match_type"] == "literal"
        assert stop["grounding_source"] == "ch. 2"


class TestReconcileStopCityGrouping:
    """MYS-660: a stop's own address must agree with the CityPlan it's filed
    under, verified against the live Colombia case (3 Cartagena-addressed
    restaurants served under Aracataca, ~250km away, while Cartagena was ALSO
    its own city on the same itinerary) plus the drop/fail-open/fold edges
    the tech plan calls out.
    """

    @staticmethod
    def _stop(name, address=None):
        return {
            "name": name,
            "type": "restaurant",
            "reason": "x",
            "address": address,
            "time_of_day": "evening",
        }

    @staticmethod
    def _city(name, stops, country="Colombia"):
        return {
            "name": name,
            "country": country,
            "days_suggested": 1,
            "overview": "o",
            "stops": stops,
        }

    def test_colombia_case_refiles_under_the_existing_city(self):
        # The exact live defect: 3 Cartagena-addressed stops filed under
        # Aracataca, with Cartagena ALSO present as its own city.
        itin = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Aracataca",
                    [
                        self._stop("Museo Casa", "Cra 5 #4-40, Aracataca, Colombia"),
                        self._stop(
                            "La Cocina de Pepina",
                            "Calle 25 #10B-15, Getsemani, Cartagena, Colombia",
                        ),
                        self._stop(
                            "La Cevicheria",
                            "Calle Stuart #7-14, Centro Historico, Cartagena, Colombia",
                        ),
                        self._stop(
                            "La Vitrola",
                            "Calle de los Estribos #33-65, Centro Historico, Cartagena, Colombia",
                        ),
                    ],
                ),
                self._city(
                    "Cartagena",
                    [
                        self._stop("Walled City", "Centro Historico, Cartagena, Colombia"),
                        self._stop("Cafe del Mar", "Baluarte de Santo Domingo, Cartagena, Colombia"),
                    ],
                ),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        by_city = {c["name"]: {s["name"] for s in c["stops"]} for c in out["cities"]}
        assert by_city["Aracataca"] == {"Museo Casa"}
        assert by_city["Cartagena"] == {
            "Walled City",
            "Cafe del Mar",
            "La Cocina de Pepina",
            "La Cevicheria",
            "La Vitrola",
        }

    def test_mismatch_with_no_matching_city_is_dropped(self):
        itin = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Paris",
                    [
                        self._stop("Eiffel Tower", "Champ de Mars, Paris, France"),
                        self._stop("Ghost Cafe", "Rue de Rivoli, Lyon, France"),
                    ],
                    country="France",
                ),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        names = [s["name"] for s in out["cities"][0]["stops"]]
        assert names == ["Eiffel Tower"]

    def test_fail_open_on_no_address_or_single_fragment(self):
        # No signal is not evidence of a mismatch -- left exactly where the
        # composer put it, matching this file's other guardrails.
        itin = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Rome",
                    [
                        self._stop("Trevi Fountain", None),
                        self._stop("Mystery Spot", "Somewhere magical"),
                    ],
                    country="Italy",
                ),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        names = [s["name"] for s in out["cities"][0]["stops"]]
        assert names == ["Trevi Fountain", "Mystery Spot"]

    def test_city_left_with_zero_stops_is_dropped(self):
        itin = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Lyon",
                    [self._stop("Ghost Cafe", "Rue de Rivoli, Paris, France")],
                    country="France",
                ),
                self._city(
                    "Paris",
                    [self._stop("Louvre", "Rue de Rivoli, Paris, France")],
                    country="France",
                ),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        assert [c["name"] for c in out["cities"]] == ["Paris"]
        assert {s["name"] for s in out["cities"][0]["stops"]} == {"Louvre", "Ghost Cafe"}

    def test_accent_and_case_fold_match(self):
        # Address tail "CIENAGA" (no accent, upper) must fold-match the
        # CityPlan name "Ciénaga" (accented) -- same helper mint_place_key
        # uses, so the two paths can never drift on what counts as a match.
        itin = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Aracataca",
                    [self._stop("Wrong filed", "Calle 1, CIENAGA, Colombia")],
                ),
                self._city(
                    "Ciénaga",
                    [self._stop("Plaza Centenario", "Plaza Centenario, Ciénaga, Colombia")],
                ),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        by_city = {c["name"]: {s["name"] for s in c["stops"]} for c in out["cities"]}
        assert by_city == {"Ciénaga": {"Plaza Centenario", "Wrong filed"}}

    def test_non_deterministic_recomposition_both_internally_consistent(self):
        # MYS-563: composition is non-deterministic -- this is deterministic
        # POST-processing, so it must hold regardless of which of two
        # differently-shaped compositions of "the same book" comes out of
        # the model. Two structurally different fixtures, both must end up
        # with zero remaining address/city mismatches.
        fixture_a = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Paris",
                    [
                        self._stop("Eiffel Tower", "Champ de Mars, Paris, France"),
                        self._stop("Ghost Cafe", "Rue de Rivoli, Lyon, France"),
                    ],
                    country="France",
                ),
            ],
        }
        fixture_b = {
            "summary_text": "t",
            "cities": [
                self._city("Aracataca", [self._stop("Wrong filed", "Calle 1, Cienaga, Colombia")]),
                self._city("Ciénaga", [self._stop("Plaza Centenario", "Plaza Centenario, Ciénaga, Colombia")]),
            ],
        }

        def is_internally_consistent(result):
            for city in result["cities"]:
                for stop in city["stops"]:
                    locality = _address_locality(stop.get("address"))
                    if locality is None:
                        continue
                    from models.place_key import slug

                    if slug(locality) and slug(locality) != slug(city["name"]):
                        return False
            return True

        assert is_internally_consistent(reconcile_stop_city_grouping(fixture_a))
        assert is_internally_consistent(reconcile_stop_city_grouping(fixture_b))

    def test_none_itinerary_returns_none(self):
        assert reconcile_stop_city_grouping(None) is None

    def test_city_that_arrives_already_empty_is_left_alone(self):
        # Regression: a city with 0 stops BEFORE this pass runs (e.g. an
        # extraction fixture/edge case unrelated to city/address mismatch)
        # must not be swept up by the "empty city is worse than no section"
        # rule -- that rule is scoped to cities THIS PASS emptied, never to
        # a city that arrived already empty. A prior implementation dropped
        # every zero-stop city unconditionally, which crashed callers
        # indexing into `cities[0]` on a single-city, zero-stop itinerary.
        itin = {
            "summary_text": "t",
            "cities": [self._city("London", [])],
        }
        out = reconcile_stop_city_grouping(itin)
        assert [c["name"] for c in out["cities"]] == ["London"]
        assert out["cities"][0]["stops"] == []

    def test_city_emptied_by_this_pass_is_still_dropped(self):
        # Contrast with the above: a city that HAD stops before this pass,
        # and ends with none because every stop was re-filed/dropped, is
        # still removed per MYS-268 -- only the "arrived empty" case is
        # exempt, not the "this pass emptied it" case.
        itin = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Aracataca",
                    [
                        self._stop(
                            "La Cocina de Pepina",
                            "Calle 25 #10B-15, Getsemani, Cartagena, Colombia",
                        )
                    ],
                ),
                self._city(
                    "Cartagena",
                    [self._stop("Walled City", "Centro Historico, Cartagena, Colombia")],
                ),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        assert [c["name"] for c in out["cities"]] == ["Cartagena"]

    def test_address_locality_country_suffix_stripped(self):
        assert (
            _address_locality("Calle 25 #10B-15, Getsemani, Cartagena, Colombia")
            == "Cartagena"
        )

    def test_address_locality_no_country_uses_last_segment(self):
        assert _address_locality("221B Baker Street, London") == "London"

    def test_address_locality_single_fragment_is_none(self):
        assert _address_locality("Somewhere magical") is None

    def test_address_locality_missing_is_none(self):
        assert _address_locality(None) is None


class TestGroundingTokenMatch:
    """The shared matching primitive behind all three grounding guards."""

    def test_exact_match(self):
        haystack = grounding_token_set("Visit the Margaret Mitchell House today")
        assert is_title_grounded("Margaret Mitchell House", haystack)

    def test_two_token_title_requires_both_tokens(self):
        haystack = grounding_token_set("only the pump is mentioned here")
        assert not is_title_grounded("Pump Room", haystack)

    def test_three_token_title_tolerates_one_missing(self):
        haystack = grounding_token_set("scenes at the pump room in bath")
        assert is_title_grounded("Grand Pump Room", haystack)

    def test_long_title_never_tolerates_two_missing(self):
        # The allowance is a fixed count, not a ratio: a flat 0.6 threshold
        # would pass this 5-token title on 3/5 scattered support.
        haystack = grounding_token_set("the pump room and roman spa")
        assert not is_title_grounded("Grand Pump Room Roman Baths", haystack)

    def test_word_boundaries_respected(self):
        haystack = grounding_token_set("the millionaire's mansion")
        assert not is_title_grounded("The Mill", haystack)

    def test_empty_or_articles_only_title_never_grounded(self):
        haystack = grounding_token_set("the a an anything")
        assert not is_title_grounded("The", haystack)
        assert not is_title_grounded("", haystack)
        assert not is_title_grounded(None, haystack)

    def test_non_string_grounding_text_yields_empty_set(self):
        assert grounding_token_set(None) == frozenset()
        assert grounding_token_set(42) == frozenset()


class TestGroundingResearchText:
    def test_concatenates_discovery_keys(self):
        accessor = SessionStateAccessor(
            {
                "book_context": {"themes": ["war"]},
                "landmark_discovery": "Margaret Mitchell House, Atlanta",
                "city_discovery": {"cities": ["Atlanta"]},
            }
        )
        text = accessor.grounding_research_text
        assert "Margaret Mitchell House" in text
        assert "Atlanta" in text
        assert "war" in text

    def test_empty_when_no_discovery(self):
        assert SessionStateAccessor({}).grounding_research_text == ""


class TestExtractItineraryAppliesDowngrade:
    def test_envelope_path_downgrades_ungrounded_literal(self):
        itin = _itinerary_with_stops([_stop("Invented Place", "literal", grounding_source="x")])
        envelope = {"itinerary": itin, "suggestions": []}
        accessor = SessionStateAccessor(
            {
                "composer_envelope": envelope,
                "landmark_discovery": "Margaret Mitchell House is in Atlanta.",
            }
        )
        result = extract_itinerary_from_response(None, accessor)
        assert result is not None
        itinerary, _ = result
        stop = itinerary["cities"][0]["stops"][0]
        assert stop["match_type"] == "vibe"
        assert stop["grounding_source"] is None

    def test_envelope_path_fail_open_without_discovery(self):
        itin = _itinerary_with_stops([_stop("Invented Place", "literal")])
        envelope = {"itinerary": itin, "suggestions": []}
        accessor = SessionStateAccessor({"composer_envelope": envelope})
        itinerary, _ = extract_itinerary_from_response(None, accessor)
        assert itinerary["cities"][0]["stops"][0]["match_type"] == "literal"
