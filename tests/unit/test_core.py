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
    extract_json_from_text,
    extract_itinerary_from_response,
)
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

    def test_write_book_metadata(self):
        state = {}
        accessor = SessionStateAccessor(state)
        accessor.book_metadata = {"book_title": "1984"}
        assert state["book_metadata"]["book_title"] == "1984"

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

    def test_write_selected_regions(self):
        state = {}
        accessor = SessionStateAccessor(state)
        accessor.selected_regions = [{"region_id": 1}]
        assert state["selected_regions"] == [{"region_id": 1}]

    def test_read_final_itinerary(self):
        itinerary = {"cities": [], "summary_text": "test"}
        state = {"final_itinerary": itinerary}
        accessor = SessionStateAccessor(state)
        assert accessor.final_itinerary == itinerary

    def test_failed_defaults_false(self):
        accessor = SessionStateAccessor({})
        assert accessor.failed is False

    def test_write_failed_flag(self):
        state = {}
        accessor = SessionStateAccessor(state)
        accessor.failed = True
        assert state["job_failed"] is True

    def test_clear_failed_flag(self):
        state = {"job_failed": True}
        accessor = SessionStateAccessor(state)
        accessor.failed = False
        assert accessor.failed is False

    def test_clear_final_itinerary(self):
        state = {"final_itinerary": {"cities": [], "summary_text": "done"}}
        accessor = SessionStateAccessor(state)
        accessor.clear_final_itinerary()
        assert accessor.final_itinerary is None
        assert "final_itinerary" not in state

    def test_clear_final_itinerary_when_absent_is_safe(self):
        state = {}
        accessor = SessionStateAccessor(state)
        accessor.clear_final_itinerary()  # must not raise
        assert accessor.final_itinerary is None


# =============================================================================
# Prompts
# =============================================================================


class TestPrompts:
    def test_discovery_prompt(self):
        prompt = build_discovery_prompt("1984", "George Orwell")
        assert '"1984"' in prompt
        assert "George Orwell" in prompt
        assert "cities" in prompt.lower()

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


class TestExtractItineraryFromResponse:
    def test_from_session_state(self):
        itinerary = {
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
        accessor = SessionStateAccessor({"final_itinerary": itinerary})
        result = extract_itinerary_from_response(None, accessor)
        assert result is not None
        assert result["cities"][0]["name"] == "London"

    def test_from_text_fallback(self):
        accessor = SessionStateAccessor({})  # no final_itinerary

        itinerary_json = json.dumps(
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
        mock_part = MagicMock()
        mock_part.text = f"Result: {itinerary_json}"
        mock_response = MagicMock()
        mock_response.content.parts = [mock_part]

        result = extract_itinerary_from_response(mock_response, accessor)
        assert result is not None

    def test_no_itinerary_anywhere(self):
        accessor = SessionStateAccessor({})
        mock_part = MagicMock()
        mock_part.text = "Sorry, I cannot do that."
        mock_response = MagicMock()
        mock_response.content.parts = [mock_part]

        result = extract_itinerary_from_response(mock_response, accessor)
        assert result is None


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
