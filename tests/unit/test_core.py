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
    _find_reconciliation_target,
    _city_names,
    _drop_suggestions_naming_removed_cities,
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


class TestClampActionPrompt:
    """MYS-494 item 1. SuggestionChip.action_prompt is deliberately
    unbounded at the schema (Eng Lead's ruling -- it's a structured-output
    field the LLM must fill; a schema max_length would fail generation
    rather than truncate). `_clamp_action_prompt` is the persist-time
    bound instead, called from `_persist_suggestions`.
    """

    def test_short_prompt_passes_through_unchanged(self):
        chip = {"id": "chip-1", "label": "Add cafes", "action_prompt": "Find cafes."}
        clamped = WorkflowExecutor._clamp_action_prompt(chip)
        assert clamped is chip, "must not copy/mutate a chip already in bounds"

    def test_overlong_prompt_is_truncated_not_dropped(self):
        overlong = "x" * (WorkflowExecutor._MAX_ACTION_PROMPT_CHARS + 50)
        chip = {"id": "chip-1", "label": "Add cafes", "action_prompt": overlong}
        clamped = WorkflowExecutor._clamp_action_prompt(chip)
        assert clamped["action_prompt"] == overlong[: WorkflowExecutor._MAX_ACTION_PROMPT_CHARS]
        # Clamp, don't reject: every other field survives untouched.
        assert clamped["id"] == "chip-1" and clamped["label"] == "Add cafes"
        # Must not mutate the caller's dict in place (a truncated-in-place
        # chip would silently change under anything upstream still
        # holding the original reference).
        assert clamped is not chip
        assert chip["action_prompt"] == overlong

    def test_missing_or_non_string_action_prompt_does_not_raise(self):
        assert WorkflowExecutor._clamp_action_prompt({"id": "c"}) == {"id": "c"}
        no_prompt = {"id": "c", "action_prompt": None}
        assert WorkflowExecutor._clamp_action_prompt(no_prompt) == no_prompt


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

    def test_mismatch_with_no_matching_city_fails_open(self):
        # MYS-660 r3: Lyon is a real, resolvable place, but it is not a
        # CityPlan anywhere on THIS itinerary -- only Paris is. r1 treated
        # "locality resolved but matches no known city" as grounds to drop
        # the stop; the Eng Lead's r3 fix-list retired that path precisely
        # because a fixed-position locality guess can misfire (see the
        # Salem/Massachusetts test below), so "no known-city match anywhere
        # in the address" is now fail-open, not a drop -- acting only
        # happens on a positive, unambiguous DIFFERENT-known-city signal.
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
        names = {s["name"] for s in out["cities"][0]["stops"]}
        assert names == {"Eiffel Tower", "Ghost Cafe"}

    def test_state_qualified_address_does_not_misidentify_locality_as_state(self):
        # The exact live regression (MYS-660 r3 / Codex P1): r1's fixed
        # "second-to-last segment" heuristic read
        # "...Salem, Massachusetts, USA" as locality "Massachusetts" (a
        # state, not a city) -- "massachusetts" matched no CityPlan, so a
        # perfectly valid Salem stop was dropped. Salem IS a CityPlan on
        # this itinerary; the fix must re-file to it, not drop.
        itin = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Boston",
                    [
                        self._stop(
                            "Salem Witch Museum",
                            "19 1/2 Washington Square N, Salem, Massachusetts, USA",
                        ),
                    ],
                    country="USA",
                ),
                self._city(
                    "Salem",
                    [self._stop("Peabody Essex Museum", "East India Sq, Salem, Massachusetts, USA")],
                    country="USA",
                ),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        by_city = {c["name"]: {s["name"] for s in c["stops"]} for c in out["cities"]}
        assert by_city == {
            "Salem": {"Peabody Essex Museum", "Salem Witch Museum"},
        } or ("Boston" not in by_city)
        # Boston is left with zero stops after the re-file and is correctly
        # dropped (MYS-268); Salem carries both.
        assert by_city["Salem"] == {"Peabody Essex Museum", "Salem Witch Museum"}

    def test_ambiguous_same_named_city_disambiguated_by_country(self):
        # MYS-660 r3 / Codex P2: two CityPlans named "London" (combined-book
        # itinerary) -- the address's own trailing country segment picks the
        # right one, same lesson as MYS-548 (identity match must not be
        # blind to a same-named disambiguator).
        itin = {
            "summary_text": "t",
            "cities": [
                self._city("Paris", [self._stop("Misfiled", "221B Baker Street, London, Canada")], country="France"),
                self._city("London", [self._stop("Tower Bridge", "Tower Bridge Rd, London, UK")], country="United Kingdom"),
                self._city("London", [self._stop("CN Tower area cafe", "1 Front St, London, Canada")], country="Canada"),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        by_country = {(c["name"], c["country"]): {s["name"] for s in c["stops"]} for c in out["cities"]}
        assert by_country[("London", "Canada")] == {"CN Tower area cafe", "Misfiled"}
        assert by_country[("London", "United Kingdom")] == {"Tower Bridge"}

    def test_ambiguous_same_named_city_without_country_signal_fails_open(self):
        # Same two same-named CityPlans, but the address has no trailing
        # country segment to disambiguate with -- there is no honest way to
        # pick one, so no action is taken (fail open, never guess).
        itin = {
            "summary_text": "t",
            "cities": [
                self._city("Paris", [self._stop("Misfiled", "Some Street, London")], country="France"),
                self._city("London", [self._stop("Tower Bridge", "Tower Bridge Rd, London, UK")], country="United Kingdom"),
                self._city("London", [self._stop("Local cafe", "1 Front St, London, Canada")], country="Canada"),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        by_country = {(c["name"], c["country"]): {s["name"] for s in c["stops"]} for c in out["cities"]}
        assert by_country[("Paris", "France")] == {"Misfiled"}

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

        from models.place_key import slug

        def is_internally_consistent(result):
            # Post-reconciliation, re-running the same target-finder against
            # the FINAL city list must find no further action for any stop
            # -- a fixed point. If it did, this pass left a stop mismatched
            # against its own address.
            cities = result["cities"]
            city_indices_by_slug = {}
            for idx, city in enumerate(cities):
                if not isinstance(city, dict):
                    continue
                name_slug = slug(city.get("name")) if isinstance(city.get("name"), str) else ""
                if name_slug:
                    city_indices_by_slug.setdefault(name_slug, []).append(idx)
            for idx, city in enumerate(cities):
                for stop in city["stops"]:
                    target = _find_reconciliation_target(
                        stop.get("address"), cities, city_indices_by_slug, own_index=idx
                    )
                    if target is not None:
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

    # ── MYS-660 r6 (Eng Lead review of r5 on ai#261): a city THIS PASS drops
    # can orphan a composer suggestion chip that named it ────────────────────
    #
    # Before this guard existed, no city was ever removed post-composition,
    # so a suggestion chip's action_prompt always named a city that still
    # existed. This guard's own re-file/drop can now empty and remove a
    # city a chip named -- executor.expand()'s target-city scan then finds
    # no match on any persisted city and silently falls back to cities[0],
    # persisting the expansion under the WRONG city. _city_names +
    # _drop_suggestions_naming_removed_cities close that gap; these tests
    # exercise them the same way _finalize composes them (diff before/after,
    # then filter), since _finalize itself is a private closure inside
    # extract_itinerary_from_response.

    def test_suggestion_naming_a_city_this_pass_drops_is_removed(self):
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
        suggestions = [
            {"id": "1", "label": "More in Aracataca", "action_prompt": "Find more spots in Aracataca"},
            {"id": "2", "label": "More in Cartagena", "action_prompt": "Find more spots in Cartagena"},
        ]

        identities_before = _city_names(itin)
        out = reconcile_stop_city_grouping(itin)
        # Sanity: reproduces the existing r2 fixture -- Aracataca is emptied
        # by this pass (its only stop re-files to Cartagena) and dropped.
        assert [c["name"] for c in out["cities"]] == ["Cartagena"]

        identities_after = _city_names(out)
        removed_identities = identities_before - identities_after
        assert removed_identities == {("aracataca", "CO")}

        kept = _drop_suggestions_naming_removed_cities(suggestions, removed_identities, identities_after)
        assert [chip["id"] for chip in kept] == ["2"]

    def test_suggestion_naming_a_surviving_city_is_kept_even_when_another_city_is_dropped(self):
        removed_identities = {("aracataca", "CO")}
        surviving_identities = {("cartagena", "CO")}
        suggestions = [
            {"id": "1", "action_prompt": "Find more spots in Aracataca"},
            {"id": "2", "action_prompt": "Find more spots in Cartagena"},
            {"id": "3", "label": "Find books like this", "action_prompt": ""},
        ]
        kept = _drop_suggestions_naming_removed_cities(suggestions, removed_identities, surviving_identities)
        # The Cartagena chip and the city-agnostic "Find books like this"
        # chip (empty action_prompt, per _build_book_recommendation_chip)
        # both survive -- only the chip positively naming a REMOVED city,
        # with no surviving city to resolve to instead, goes.
        assert [chip["id"] for chip in kept] == ["2", "3"]

    # ── MYS-660 r7 (Codex P1, valid): the r6 filter's ONE substring test
    # over-dropped -- a removed name that's merely a SUBSTRING of unrelated
    # surviving prompt text (a real different city, or ordinary English)
    # was enough to drop a perfectly resolvable chip. The fix checks BOTH
    # directions: names a removed city AND does not ALSO still resolve to a
    # surviving one -- mirroring executor.expand()'s own first-match scan
    # over the CURRENT cities list, not a guess about what was removed.

    def test_suggestion_surviving_a_substring_collision_with_a_removed_name_is_kept(self):
        # "York" was removed; "New York" survives. The naive substring test
        # ("york" in "find more spots in new york") would false-drop this
        # perfectly valid, still-resolvable chip.
        removed_identities = {("york", "US")}
        surviving_identities = {("new york", "US")}
        suggestions = [{"id": "1", "action_prompt": "Find more spots in New York"}]
        kept = _drop_suggestions_naming_removed_cities(suggestions, removed_identities, surviving_identities)
        assert [chip["id"] for chip in kept] == ["1"]

    def test_suggestion_naming_only_a_removed_city_with_no_surviving_match_is_dropped(self):
        # The genuine orphan case: names a removed city, and nothing
        # surviving would resolve it either -- this is exactly the chip
        # that would silently fall back to cities[0] in expand().
        removed_identities = {("aracataca", "CO")}
        surviving_identities = {("cartagena", "CO")}
        suggestions = [{"id": "1", "action_prompt": "Find more spots in Aracataca"}]
        kept = _drop_suggestions_naming_removed_cities(suggestions, removed_identities, surviving_identities)
        assert kept == []

    # ── MYS-401 r4 (Codex P1, valid): the r7 fix's condition (b) used the
    # SAME plain substring test as condition (a) -- consistent with
    # expand()'s OWN matching at the time, but expand() has since switched
    # to the stricter, word-boundary `_matches_city_as_standalone_word`.
    # A chip could survive here on a substring match ("York" found inside
    # a prompt mentioning "Yorkshire") and then still get rejected by
    # expand()'s stricter scan and fall back to cities[0] -- reaching the
    # exact defect this filter exists to prevent through a different door.

    def test_suggestion_surviving_only_via_substring_inside_a_longer_word_is_dropped(self):
        # "York" removed; "Yorkshire" survives as an unrelated city name.
        # The OLD substring test ("york" in "...yorkshire...") would
        # incorrectly count this as "still resolves to a survivor" and
        # keep the chip -- but expand()'s real standalone-word scan
        # rejects "York" inside "Yorkshire" (MYS-401's own fix), so the
        # chip would silently fall back to cities[0] in expand() if kept.
        # This filter must agree with expand() and drop it too.
        removed_identities = {("york", "GB")}
        surviving_identities = {("yorkshire", "GB")}
        suggestions = [{"id": "1", "action_prompt": "Find more spots near York"}]
        kept = _drop_suggestions_naming_removed_cities(suggestions, removed_identities, surviving_identities)
        assert kept == []

    def test_suggestion_surviving_a_genuine_standalone_word_match_is_kept(self):
        # Contrast: "Bath" removed, "Bath" is ALSO a standalone mention
        # elsewhere via a different surviving city sharing the exact word
        # boundary rules expand() uses (e.g. re-filed under a synonymous
        # surviving identity in the same trip). Uses the same fixture
        # shape as the New York case above but through the standalone-word
        # path specifically, confirming r9 didn't just narrow matching --
        # a real standalone match still survives correctly.
        removed_identities = {("bath", "GB")}
        surviving_identities = {("bath", "US")}  # e.g. Bath, Maine
        suggestions = [{"id": "1", "action_prompt": "Find more spots in Bath"}]
        kept = _drop_suggestions_naming_removed_cities(suggestions, removed_identities, surviving_identities)
        assert kept == suggestions

    # ── MYS-660 r8 (Codex P2, lower severity): the (a)/(b) test is
    # name-only, same limit `expand()` itself has -- `action_prompt` is
    # free text, no country field to qualify against. That's inherent, not
    # a bug. What r6/r7 got wrong is going silent about it: a removed
    # identity and a surviving identity can share a bare name but resolve
    # to DIFFERENT countries (a removed London, GB "surviving" only
    # because a same-trip London, CA also exists) -- the chip is correctly
    # KEPT (nothing safer to do with no country signal), but a warning
    # must fire so the ambiguity is visible, unlike the ordinary
    # unambiguous "still resolves" case.

    def test_drop_suggestions_keeps_a_cross_country_name_collision_chip(self):
        # No safe alternative exists (no country in free text) -- the
        # behavioral contract is "still kept, same as the ordinary case";
        # the r8 fix additionally logs a warning naming the collision
        # (`suggestion_kept_despite_removed_city_name_collision`) so it's
        # visible rather than silently indistinguishable from the
        # unambiguous survivor case below.
        removed_identities = {("london", "GB")}
        surviving_identities = {("london", "CA")}
        suggestions = [{"id": "1", "action_prompt": "More like London"}]
        kept = _drop_suggestions_naming_removed_cities(suggestions, removed_identities, surviving_identities)
        assert kept == suggestions

    def test_drop_suggestions_does_not_flag_the_ordinary_unambiguous_survivor(self):
        # Contrast case: no country collision at all (both unresolved) --
        # the ordinary "still resolves" path, no warning expected beyond
        # what the plain substring test already covers.
        removed_identities = {("york", None)}
        surviving_identities = {("new york", None)}
        suggestions = [{"id": "1", "action_prompt": "Find more spots in New York"}]
        kept = _drop_suggestions_naming_removed_cities(suggestions, removed_identities, surviving_identities)
        assert kept == suggestions

    def test_drop_suggestions_is_a_no_op_when_nothing_was_removed(self):
        suggestions = [{"id": "1", "action_prompt": "Find more spots in Cartagena"}]
        assert _drop_suggestions_naming_removed_cities(suggestions, set(), {("cartagena", "CO")}) == suggestions
        assert _drop_suggestions_naming_removed_cities([], {("aracataca", "CO")}, set()) == []
        assert _drop_suggestions_naming_removed_cities(None, {("aracataca", "CO")}, set()) is None

    def test_city_names_is_case_insensitive_and_tolerates_malformed_input(self):
        assert _city_names(None) == set()
        assert _city_names({"cities": "not-a-list"}) == set()
        assert _city_names(
            {"cities": [{"name": "Cartagena", "country": "Colombia"}, {"no_name": True}, "not-a-dict"]}
        ) == {("cartagena", "CO")}

    def test_city_names_tolerates_an_unresolvable_country(self):
        assert _city_names({"cities": [{"name": "Neverland", "country": "Nowhereland"}]}) == {
            ("neverland", None)
        }

    # ── MYS-660 r7 (Codex P2, valid): _city_names must be COUNTRY-qualified,
    # same lesson as MYS-548 and _find_reconciliation_target's own country
    # check -- two same-trip cities sharing a bare name collapse to ONE
    # set entry, so removing one of them while the other survives looked
    # like "nothing was removed" under a name-only diff.

    def test_city_names_distinguishes_same_named_cities_in_different_countries(self):
        itin = {
            "cities": [
                self._city("London", [], country="United Kingdom"),
                self._city("London", [], country="Canada"),
            ]
        }
        assert _city_names(itin) == {("london", "GB"), ("london", "CA")}

    def test_removal_of_one_same_named_city_is_detected_even_though_another_survives(self):
        before = {("london", "GB"), ("london", "CA"), ("cartagena", "CO")}
        after = {("london", "CA"), ("cartagena", "CO")}
        # A bare-name diff would see "london" on both sides and report NO
        # removal at all -- the country-qualified diff correctly isolates
        # the specific (name, country) identity that's actually gone.
        assert before - after == {("london", "GB")}

    def test_reconciliation_target_matches_a_known_different_city(self):
        cities = [
            self._city("Aracataca", []),
            self._city("Cartagena", []),
        ]
        city_indices_by_slug = {"aracataca": [0], "cartagena": [1]}
        target = _find_reconciliation_target(
            "Calle 25 #10B-15, Getsemani, Cartagena, Colombia",
            cities,
            city_indices_by_slug,
            own_index=0,
        )
        assert target == 1

    def test_reconciliation_target_no_country_still_matches_last_segment(self):
        cities = [self._city("Aracataca", []), self._city("London", [])]
        city_indices_by_slug = {"aracataca": [0], "london": [1]}
        target = _find_reconciliation_target(
            "221B Baker Street, London", cities, city_indices_by_slug, own_index=0
        )
        assert target == 1

    def test_reconciliation_target_single_fragment_or_missing_is_none(self):
        cities = [self._city("Aracataca", [])]
        city_indices_by_slug = {"aracataca": [0]}
        assert (
            _find_reconciliation_target("Somewhere magical", cities, city_indices_by_slug, own_index=0)
            is None
        )
        assert _find_reconciliation_target(None, cities, city_indices_by_slug, own_index=0) is None

    def test_reconciliation_target_matches_own_city_is_none(self):
        # The address names the SAME city the stop is already filed under --
        # not a mismatch, no action.
        cities = [self._city("Cartagena", [])]
        city_indices_by_slug = {"cartagena": [0]}
        target = _find_reconciliation_target(
            "Calle 1, Cartagena, Colombia", cities, city_indices_by_slug, own_index=0
        )
        assert target is None

    def test_reconciliation_target_state_segment_not_misread_as_city(self):
        # The MYS-660 r3 regression, unit-tested directly against the
        # helper: "Massachusetts" must never be treated as a locality
        # candidate that could match (or fail to match) a CityPlan -- it is
        # a state, and Salem is the correct, positively-identified target.
        #
        # r7: explicit country="USA" on BOTH cities -- this test predates
        # r5's country-qualification (every segment, including a unique
        # match, is now qualified against the address's own country) and
        # was never updated for it. Left at `_city`'s "Colombia" default,
        # Salem's own CityPlan country (Colombia) disagreed with the
        # address's resolved country (USA) and the match was silently
        # excluded -- `target` came back None instead of 1, a latent
        # fixture bug this run's standalone verification caught (r5/r6
        # were never actually run against the real suite, only reasoned
        # about by hand -- see PR notes). The address is USA; the fixture
        # must be too.
        cities = [self._city("Boston", [], country="USA"), self._city("Salem", [], country="USA")]
        city_indices_by_slug = {"boston": [0], "salem": [1]}
        target = _find_reconciliation_target(
            "19 1/2 Washington Square N, Salem, Massachusetts, USA",
            cities,
            city_indices_by_slug,
            own_index=0,
        )
        assert target == 1

    def test_reconciliation_target_no_known_city_anywhere_is_none(self):
        # Lyon resolves as a real place but matches no CityPlan on this
        # itinerary at all -- fail open, not a drop signal.
        cities = [self._city("Paris", [])]
        city_indices_by_slug = {"paris": [0]}
        target = _find_reconciliation_target(
            "Rue de Rivoli, Lyon, France", cities, city_indices_by_slug, own_index=0
        )
        assert target is None

    def test_reconciliation_target_own_city_wins_over_a_different_segment_match(self):
        # MYS-660 r4 / Codex P1, unit-tested directly against the helper:
        # a stop filed under Buffalo whose address is
        # "..., Buffalo, New York, USA" names BOTH its own city (Buffalo)
        # AND a state that happens to share a name with a different
        # same-trip CityPlan (New York). r3 let the later "New York"
        # segment outvote the earlier "Buffalo" (own-city) evidence and
        # re-filed a correct stop into the wrong city. Own-city evidence
        # anywhere in the address must win, regardless of segment order.
        cities = [self._city("Buffalo", []), self._city("New York", [])]
        city_indices_by_slug = {"buffalo": [0], "new-york": [1]}
        target = _find_reconciliation_target(
            "1 Symphony Cir, Buffalo, New York, USA",
            cities,
            city_indices_by_slug,
            own_index=0,
        )
        assert target is None

    # ── MYS-660 r8 (Codex P1, blocking): a US state/territory name can
    # collide with a DIFFERENT same-trip city's name -- "Washington" the
    # STATE happens to equal "Washington" a same-trip CityPlan (Washington,
    # D.C.). The all-segment scan (r3-r7) treated any segment matching a
    # known CityPlan as locality evidence, so a Seattle-addressed stop
    # (Seattle itself is NOT a CityPlan on this trip) got actively RE-FILED
    # into Washington on state-name-only evidence -- the exact wrong-city
    # placement this guard exists to prevent, and could delete Washington
    # if it was that city's only stop. A state-name segment must never be
    # the sole different-city signal; own-city attestation is unaffected.

    def test_reconciliation_target_state_name_colliding_with_a_different_city_is_not_a_locality(self):
        cities = [self._city("Portland", [], country="USA"), self._city("Washington", [], country="USA")]
        city_indices_by_slug = {"portland": [0], "washington": [1]}
        target = _find_reconciliation_target(
            "Pike Place, Seattle, Washington, USA",
            cities,
            city_indices_by_slug,
            own_index=0,
        )
        assert target is None

    def test_seattle_stop_addressed_in_washington_state_is_not_misfiled_to_washington_dc(self):
        # Full reconcile_stop_city_grouping-level regression for the same
        # bug: a Portland-filed, Seattle-addressed stop must not be
        # re-filed into a same-trip Washington CityPlan just because the
        # address's STATE field is the string "Washington".
        itin = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Portland",
                    [self._stop("Voodoo Doughnut", "Pike Place, Seattle, Washington, USA")],
                    country="USA",
                ),
                self._city(
                    "Washington",
                    [self._stop("Lincoln Memorial", "2 Lincoln Memorial Cir NW, Washington, USA")],
                    country="USA",
                ),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        by_city = {c["name"]: {s["name"] for s in c["stops"]} for c in out["cities"]}
        assert by_city == {
            "Portland": {"Voodoo Doughnut"},
            "Washington": {"Lincoln Memorial"},
        }

    def test_reconciliation_target_state_name_exclusion_does_not_block_own_city_attestation(self):
        # A real Washington, D.C. stop -- own-city "still home" evidence
        # must stay unrestricted (only the DIFFERENT-city re-file signal is
        # restricted): the address's own "Washington" segment matching the
        # stop's OWN CityPlan still short-circuits to "no action".
        cities = [self._city("Portland", [], country="USA"), self._city("Washington", [], country="USA")]
        city_indices_by_slug = {"portland": [0], "washington": [1]}
        target = _find_reconciliation_target(
            "2 Lincoln Memorial Cir NW, Washington, USA",
            cities,
            city_indices_by_slug,
            own_index=1,
        )
        assert target is None

    def test_buffalo_stop_stays_put_and_is_not_deleted(self):
        # Full reconcile_stop_city_grouping-level regression for the same
        # bug: a same-trip itinerary with both Buffalo and New York as
        # CityPlans must leave the Buffalo-filed, Buffalo-addressed stop
        # exactly where it is -- and Buffalo, left with one stop, must
        # survive (not be swept by the "emptied by this pass" drop rule).
        itin = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Buffalo",
                    [
                        self._stop(
                            "Buffalo City Hall",
                            "65 Niagara Sq, Buffalo, New York, USA",
                        )
                    ],
                    country="USA",
                ),
                self._city(
                    "New York",
                    [self._stop("Empire State Building", "20 W 34th St, New York, USA")],
                    country="USA",
                ),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        by_city = {c["name"]: {s["name"] for s in c["stops"]} for c in out["cities"]}
        assert by_city == {
            "Buffalo": {"Buffalo City Hall"},
            "New York": {"Empire State Building"},
        }

    def test_reconciliation_target_unique_match_still_country_checked(self):
        # MYS-660 r5 / Codex P2, unit-tested directly against the helper:
        # the OLD code only country-qualified a segment when more than one
        # CityPlan shared its slug -- a LONE same-named CityPlan matched on
        # name alone, ignoring the address's own country. Here the trip's
        # only "Paris" CityPlan is in France, but the address says
        # "Paris, Texas, USA" -- the country disagreement must be honored
        # even though "paris" resolves to exactly one candidate.
        cities = [
            self._city("Dallas", [], country="USA"),
            self._city("Paris", [], country="France"),
        ]
        city_indices_by_slug = {"dallas": [0], "paris": [1]}
        target = _find_reconciliation_target(
            "123 Main St, Paris, Texas, USA",
            cities,
            city_indices_by_slug,
            own_index=0,
        )
        assert target is None

    def test_paris_texas_stop_is_not_misfiled_to_paris_france(self):
        # Full reconcile_stop_city_grouping-level regression for the same
        # bug: a Dallas-filed stop addressed in Paris, TEXAS must not be
        # re-filed into a same-trip Paris, FRANCE CityPlan just because the
        # city name matches -- the country must agree too, unique match or
        # not.
        itin = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Dallas",
                    [self._stop("Reunion Tower", "300 Reunion Blvd, Paris, Texas, USA")],
                    country="USA",
                ),
                self._city(
                    "Paris",
                    [self._stop("Eiffel Tower", "Champ de Mars, Paris, France")],
                    country="France",
                ),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        by_city = {c["name"]: {s["name"] for s in c["stops"]} for c in out["cities"]}
        assert by_city == {
            "Dallas": {"Reunion Tower"},
            "Paris": {"Eiffel Tower"},
        }

    def test_reconciliation_target_unique_match_with_agreeing_country_still_refiles(self):
        # Contrast with the two tests above: when the trip's only "Paris"
        # CityPlan agrees with the address's country, the unique-match path
        # must still positively re-file -- the r5 fix adds a country check,
        # it must not turn into "never trust a unique match".
        cities = [
            self._city("Dallas", [], country="USA"),
            self._city("Paris", [], country="USA"),
        ]
        city_indices_by_slug = {"dallas": [0], "paris": [1]}
        target = _find_reconciliation_target(
            "123 Main St, Paris, Texas, USA",
            cities,
            city_indices_by_slug,
            own_index=0,
        )
        assert target == 1

    # ── MYS-660 r7 (Codex P2, lowest severity, fail-open under-reach not a
    # regression): a resolved trailing segment was ALWAYS dropped as a
    # locality candidate, even when it's also the city itself -- a
    # city-state address ("Marina Bay, Singapore") lost the only segment
    # that could ever have named that CityPlan, so a same-trip misfile
    # there went unreconciled. Keep it as a candidate too when it ALSO
    # slug-matches a CityPlan already on the trip.

    def test_reconciliation_target_city_state_trailing_segment_still_matches(self):
        cities = [
            self._city("Kuala Lumpur", [], country="Malaysia"),
            self._city("Singapore", [], country="Singapore"),
        ]
        city_indices_by_slug = {"kuala-lumpur": [0], "singapore": [1]}
        target = _find_reconciliation_target(
            "Marina Bay, Singapore",
            cities,
            city_indices_by_slug,
            own_index=0,
        )
        assert target == 1

    def test_reconciliation_target_trailing_segment_still_excluded_when_it_names_no_city(self):
        # The unchanged case: a trailing segment that resolves as a country
        # AND names no CityPlan on this trip is still excluded as a
        # locality candidate -- this fix only ADDS a candidate when one
        # genuinely exists, it doesn't stop stripping otherwise.
        cities = [self._city("Kuala Lumpur", [], country="Malaysia")]
        city_indices_by_slug = {"kuala-lumpur": [0]}
        target = _find_reconciliation_target(
            "Somewhere, Malaysia",
            cities,
            city_indices_by_slug,
            own_index=0,
        )
        assert target is None

    def test_marina_bay_singapore_stop_is_reconciled_to_the_city_state(self):
        # Full reconcile_stop_city_grouping-level regression: a stop
        # addressed in the city-state itself, filed under the wrong city,
        # must be re-filed -- previously the trailing "Singapore" segment
        # was stripped outright as "just the country" and this mismatch
        # went undetected.
        itin = {
            "summary_text": "t",
            "cities": [
                self._city(
                    "Kuala Lumpur",
                    [self._stop("Marina Bay Sands", "Marina Bay, Singapore")],
                    country="Malaysia",
                ),
                self._city("Singapore", [], country="Singapore"),
            ],
        }
        out = reconcile_stop_city_grouping(itin)
        by_city = {c["name"]: {s["name"] for s in c["stops"]} for c in out["cities"]}
        assert by_city == {"Singapore": {"Marina Bay Sands"}}


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
