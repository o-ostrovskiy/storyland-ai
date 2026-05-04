"""
Unit tests for FastAPI SSE API layer.

Tests API models, endpoint responses, and SSE streaming with mocked executor.
"""

import json

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    DiscoverRequest,
    ComposeRequest,
    ExpandRequest,
    RecommendBooksRequest,
    LocalAtmosphereRequest,
    UserLocation,
    SSEProgressEvent,
    SSEStartedEvent,
    SSEMetadataEvent,
    SSERegionsEvent,
    SSEItineraryEvent,
    SSEExpansionEvent,
    SSEBookRecommendationsEvent,
    SSEErrorEvent,
    SSEDoneEvent,
    HealthResponse,
    JobStatusResponse,
    JobStatus,
)
from api.streaming import _sse, domain_event_to_sse
from core.events import (
    Phase,
    ProgressEvent,
    JobStarted,
    MetadataReady,
    RegionsReady,
    ItineraryReady,
    ExpansionReady,
    BookRecommendationsReady,
    WorkflowError,
    WorkflowComplete,
)


# =============================================================================
# Model Validation Tests
# =============================================================================


class TestDiscoverRequest:
    """Tests for DiscoverRequest model."""

    def test_minimal_request(self):
        req = DiscoverRequest(book_title="1984", author="George Orwell")
        assert req.book_title == "1984"
        assert req.author == "George Orwell"
        assert req.preferences is None

    def test_missing_author_raises(self):
        with pytest.raises(ValidationError):
            DiscoverRequest(book_title="1984")

    def test_blank_author_raises(self):
        with pytest.raises(ValidationError):
            DiscoverRequest(book_title="1984", author="   ")

    def test_author_is_trimmed(self):
        req = DiscoverRequest(book_title="1984", author="  George Orwell  ")
        assert req.author == "George Orwell"

    def test_full_request(self):
        req = DiscoverRequest(
            book_title="1984",
            author="George Orwell",
            preferences={"budget": "luxury"},
        )
        assert req.author == "George Orwell"
        assert req.preferences["budget"] == "luxury"

    def test_missing_book_title_raises(self):
        with pytest.raises(ValidationError):
            DiscoverRequest()

    def test_blank_book_title_raises(self):
        with pytest.raises(ValidationError):
            DiscoverRequest(book_title="   ")

    def test_book_title_is_trimmed(self):
        req = DiscoverRequest(book_title="  1984  ", author="George Orwell")
        assert req.book_title == "1984"

    def test_serialization_roundtrip(self):
        req = DiscoverRequest(book_title="1984", author="George Orwell")
        data = json.loads(req.model_dump_json())
        restored = DiscoverRequest(**data)
        assert restored.book_title == req.book_title
        assert restored.author == req.author


class TestComposeRequest:
    """Tests for ComposeRequest model."""

    def test_valid_request(self):
        req = ComposeRequest(region_ids=[1, 2])
        assert req.region_ids == [1, 2]

    def test_single_region(self):
        req = ComposeRequest(region_ids=[3])
        assert req.region_ids == [3]

    def test_empty_region_ids_rejected(self):
        with pytest.raises(ValidationError):
            ComposeRequest(region_ids=[])

    def test_missing_region_ids_raises(self):
        with pytest.raises(ValidationError):
            ComposeRequest()


class TestUserLocation:
    """Tests for UserLocation model."""

    def test_valid_location(self):
        loc = UserLocation(lat=40.7128, lng=-74.0060, label="New York, NY")
        assert loc.lat == 40.7128
        assert loc.label == "New York, NY"

    def test_lat_out_of_range(self):
        with pytest.raises(ValidationError):
            UserLocation(lat=91.0, lng=0.0, label="X")

    def test_lng_out_of_range(self):
        with pytest.raises(ValidationError):
            UserLocation(lat=0.0, lng=181.0, label="X")

    def test_blank_label_raises(self):
        with pytest.raises(ValidationError):
            UserLocation(lat=0.0, lng=0.0, label="   ")

    def test_label_is_trimmed(self):
        loc = UserLocation(lat=0.0, lng=0.0, label="  Boston  ")
        assert loc.label == "Boston"


class TestLocalAtmosphereRequest:
    """Tests for LocalAtmosphereRequest model."""

    def _location(self) -> UserLocation:
        return UserLocation(lat=40.7128, lng=-74.0060, label="New York, NY")

    def test_minimal_request(self):
        req = LocalAtmosphereRequest(
            book_title="Wuthering Heights",
            author="Emily Brontë",
            user_location=self._location(),
        )
        assert req.radius_km == 80
        assert req.preferences is None

    def test_full_request(self):
        req = LocalAtmosphereRequest(
            book_title="Wuthering Heights",
            author="Emily Brontë",
            user_location=self._location(),
            radius_km=120,
            preferences={"budget": "luxury"},
        )
        assert req.radius_km == 120
        assert req.preferences["budget"] == "luxury"

    def test_radius_below_min_rejected(self):
        with pytest.raises(ValidationError):
            LocalAtmosphereRequest(
                book_title="X",
                author="Y",
                user_location=self._location(),
                radius_km=5,
            )

    def test_radius_above_max_rejected(self):
        with pytest.raises(ValidationError):
            LocalAtmosphereRequest(
                book_title="X",
                author="Y",
                user_location=self._location(),
                radius_km=500,
            )

    def test_blank_book_title_rejected(self):
        with pytest.raises(ValidationError):
            LocalAtmosphereRequest(
                book_title="   ",
                author="Y",
                user_location=self._location(),
            )

    def test_missing_user_location_rejected(self):
        with pytest.raises(ValidationError):
            LocalAtmosphereRequest(book_title="X", author="Y")

    def test_serialization_roundtrip(self):
        req = LocalAtmosphereRequest(
            book_title="X",
            author="Y",
            user_location=self._location(),
        )
        data = json.loads(req.model_dump_json())
        restored = LocalAtmosphereRequest(**data)
        assert restored.book_title == "X"
        assert restored.user_location.lat == 40.7128


# =============================================================================
# SSE Event Model Tests
# =============================================================================


class TestSSEProgressEvent:
    def test_basic_progress(self):
        event = SSEProgressEvent(phase=1, step="Searching")
        data = json.loads(event.model_dump_json())
        assert data["event"] == "progress"
        assert data["phase"] == 1
        assert data["step"] == "Searching"
        assert data["detail"] is None

    def test_progress_with_detail(self):
        event = SSEProgressEvent(
            phase=2, step="Discovering cities", detail="city_pipeline"
        )
        data = json.loads(event.model_dump_json())
        assert data["detail"] == "city_pipeline"

    def test_event_literal_is_fixed(self):
        event = SSEProgressEvent(phase=1, step="test")
        assert event.event == "progress"


class TestSSEMetadataEvent:
    def test_full_metadata(self):
        event = SSEMetadataEvent(
            book_title="1984",
            author="George Orwell",
            description="A dystopian novel",
            published_date="1949",
            categories=["Fiction", "Dystopian"],
            image_url="https://example.com/cover.jpg",
        )
        data = json.loads(event.model_dump_json())
        assert data["event"] == "metadata"
        assert data["book_title"] == "1984"
        assert len(data["categories"]) == 2

    def test_minimal_metadata(self):
        event = SSEMetadataEvent(book_title="1984", author="George Orwell")
        data = json.loads(event.model_dump_json())
        assert data["description"] is None
        assert data["categories"] == []


class TestSSERegionsEvent:
    def test_regions_event(self):
        regions = [
            {"region_id": 1, "region_name": "England", "cities": []},
            {"region_id": 2, "region_name": "Scotland", "cities": []},
        ]
        event = SSERegionsEvent(
            job_id="abc-123",
            regions=regions,
            analysis_note="Grouped by proximity",
        )
        data = json.loads(event.model_dump_json())
        assert data["event"] == "regions"
        assert data["job_id"] == "abc-123"
        assert len(data["regions"]) == 2

    def test_empty_regions(self):
        event = SSERegionsEvent(job_id="abc", regions=[])
        data = json.loads(event.model_dump_json())
        assert data["regions"] == []


class TestExpandRequest:
    def test_valid_request(self):
        req = ExpandRequest(
            action_id="abc-123",
            action_label="Add restaurants",
            action_prompt="Find atmospheric restaurants near the stops.",
        )
        assert req.action_id == "abc-123"
        assert req.action_label == "Add restaurants"

    def test_missing_action_id_raises(self):
        with pytest.raises(ValidationError):
            ExpandRequest(action_label="Add restaurants", action_prompt="Find restaurants.")

    def test_missing_action_prompt_raises(self):
        with pytest.raises(ValidationError):
            ExpandRequest(action_id="x", action_label="Add restaurants")

    def test_action_prompt_max_length(self):
        with pytest.raises(ValidationError):
            ExpandRequest(action_id="x", action_label="Test", action_prompt="x" * 501)


class TestSSEItineraryEvent:
    def test_itinerary_event(self):
        itinerary = {
            "cities": [{"name": "London", "country": "UK"}],
            "summary_text": "A literary journey",
        }
        event = SSEItineraryEvent(itinerary=itinerary)
        data = json.loads(event.model_dump_json())
        assert data["event"] == "itinerary"
        assert data["itinerary"]["cities"][0]["name"] == "London"

    def test_itinerary_event_with_suggestions(self):
        chip = {"id": "x", "label": "Add restaurants", "action_prompt": "Find restaurants."}
        event = SSEItineraryEvent(itinerary={"cities": [], "summary_text": "test"}, suggestions=[chip])
        data = json.loads(event.model_dump_json())
        assert len(data["suggestions"]) == 1
        assert data["suggestions"][0]["label"] == "Add restaurants"

    def test_itinerary_event_default_suggestions(self):
        event = SSEItineraryEvent(itinerary={"cities": [], "summary_text": "test"})
        data = json.loads(event.model_dump_json())
        assert data["suggestions"] == []

    def test_itinerary_event_book_recommendation_chip(self):
        chip = {"id": "books-1", "label": "Find books like this", "action_prompt": ""}
        event = SSEItineraryEvent(
            itinerary={"cities": [], "summary_text": "test"},
            book_recommendation_chip=chip,
        )
        data = json.loads(event.model_dump_json())
        assert data["book_recommendation_chip"]["id"] == "books-1"
        assert data["book_recommendation_chip"]["label"] == "Find books like this"

    def test_itinerary_event_default_book_recommendation_chip(self):
        event = SSEItineraryEvent(itinerary={"cities": [], "summary_text": "test"})
        data = json.loads(event.model_dump_json())
        assert data["book_recommendation_chip"] is None


class TestSSEExpansionEvent:
    def test_expansion_event(self):
        event = SSEExpansionEvent(
            parent_city="London",
            places=[{"name": "The Ritz", "type": "restaurant"}],
            suggestions=[{"id": "y", "label": "More cafés", "action_prompt": "Find cafés."}],
        )
        data = json.loads(event.model_dump_json())
        assert data["event"] == "expansion"
        assert data["parent_city"] == "London"
        assert len(data["places"]) == 1
        assert len(data["suggestions"]) == 1

    def test_expansion_event_default_suggestions(self):
        event = SSEExpansionEvent(parent_city="Paris", places=[])
        data = json.loads(event.model_dump_json())
        assert data["suggestions"] == []

    def test_expansion_event_book_recommendation_chip(self):
        chip = {"id": "books-2", "label": "Find books like this", "action_prompt": ""}
        event = SSEExpansionEvent(
            parent_city="London",
            places=[],
            book_recommendation_chip=chip,
        )
        data = json.loads(event.model_dump_json())
        assert data["book_recommendation_chip"]["id"] == "books-2"

    def test_expansion_event_default_book_recommendation_chip(self):
        event = SSEExpansionEvent(parent_city="Paris", places=[])
        data = json.loads(event.model_dump_json())
        assert data["book_recommendation_chip"] is None


class TestDomainEventToSSEExpansion:
    def test_expansion_ready_event(self):
        event = ExpansionReady(
            parent_city="London",
            places=[{"name": "Café A"}],
            suggestions=[{"id": "z", "label": "More spots", "action_prompt": "Find more."}],
        )
        sse = domain_event_to_sse(event)
        assert sse["event"] == "expansion"
        data = json.loads(sse["data"])
        assert data["parent_city"] == "London"
        assert len(data["places"]) == 1

    def test_itinerary_ready_with_suggestions(self):
        event = ItineraryReady(
            itinerary={"cities": [], "summary_text": "test"},
            suggestions=[{"id": "s", "label": "Add restaurants", "action_prompt": "Find places."}],
        )
        sse = domain_event_to_sse(event)
        assert sse["event"] == "itinerary"
        data = json.loads(sse["data"])
        assert len(data["suggestions"]) == 1

    def test_itinerary_ready_book_recommendation_chip_passthrough(self):
        chip = {"id": "books-3", "label": "Find books like this", "action_prompt": ""}
        event = ItineraryReady(
            itinerary={"cities": [], "summary_text": "test"},
            suggestions=[],
            book_recommendation_chip=chip,
        )
        sse = domain_event_to_sse(event)
        data = json.loads(sse["data"])
        assert data["book_recommendation_chip"]["id"] == "books-3"

    def test_expansion_ready_book_recommendation_chip_passthrough(self):
        chip = {"id": "books-4", "label": "Find books like this", "action_prompt": ""}
        event = ExpansionReady(
            parent_city="London",
            places=[],
            suggestions=[],
            book_recommendation_chip=chip,
        )
        sse = domain_event_to_sse(event)
        data = json.loads(sse["data"])
        assert data["book_recommendation_chip"]["id"] == "books-4"


class TestSSEErrorEvent:
    def test_error_with_phase(self):
        event = SSEErrorEvent(
            message="Something failed", error_type="TimeoutError", phase=2,
        )
        data = json.loads(event.model_dump_json())
        assert data["event"] == "error"
        assert data["message"] == "Something failed"
        assert data["phase"] == 2

    def test_error_without_phase(self):
        event = SSEErrorEvent(message="Generic error")
        data = json.loads(event.model_dump_json())
        assert data["phase"] is None
        assert data["error_type"] == "WorkflowError"


class TestSSEDoneEvent:
    def test_done_event(self):
        event = SSEDoneEvent(job_id="abc-123")
        data = json.loads(event.model_dump_json())
        assert data["event"] == "done"
        assert data["job_id"] == "abc-123"
        assert data["token_usage"] is None

    def test_done_with_token_usage(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        event = SSEDoneEvent(job_id="abc", token_usage=usage)
        data = json.loads(event.model_dump_json())
        assert data["token_usage"]["total_tokens"] == 150


# =============================================================================
# REST Response Model Tests
# =============================================================================


class TestHealthResponse:
    def test_defaults(self):
        resp = HealthResponse()
        assert resp.status == "healthy"
        assert resp.version == "0.1.0"

    def test_with_model_name(self):
        resp = HealthResponse(model_name="gemini-2.0-flash")
        assert resp.model_name == "gemini-2.0-flash"


class TestJobStatusResponse:
    def test_all_statuses_valid(self):
        for status in JobStatus:
            resp = JobStatusResponse(job_id="test", status=status)
            assert resp.status == status

    def test_status_with_metadata(self):
        resp = JobStatusResponse(
            job_id="test",
            status=JobStatus.REGIONS_READY,
            book_title="1984",
            author="George Orwell",
            has_regions=True,
        )
        assert resp.has_regions is True
        assert resp.has_itinerary is False

    def test_completed_status(self):
        resp = JobStatusResponse(
            job_id="test", status=JobStatus.COMPLETED, has_itinerary=True,
        )
        assert resp.has_itinerary is True


# =============================================================================
# SSE Helper & Adapter Tests
# =============================================================================


class TestSSEHelper:
    def test_sse_format(self):
        result = _sse("progress", '{"phase": 1}')
        assert result == {"event": "progress", "data": '{"phase": 1}'}

    def test_sse_with_model(self):
        event = SSEProgressEvent(phase=1, step="test")
        result = _sse("progress", event.model_dump_json())
        assert result["event"] == "progress"
        parsed = json.loads(result["data"])
        assert parsed["phase"] == 1


class TestDomainEventToSSE:
    """Tests for domain_event_to_sse adapter function."""

    def test_progress_event(self):
        event = ProgressEvent(phase=Phase.DISCOVERY, step="Finding cities")
        result = domain_event_to_sse(event)
        assert result["event"] == "progress"
        data = json.loads(result["data"])
        assert data["phase"] == 2
        assert data["step"] == "Finding cities"

    def test_job_started(self):
        event = JobStarted(job_id="abc-123")
        result = domain_event_to_sse(event)
        assert result["event"] == "started"
        data = json.loads(result["data"])
        assert data["job_id"] == "abc-123"
        assert data["event"] == "started"

    def test_metadata_ready(self):
        event = MetadataReady(
            metadata={"book_title": "1984", "author": "George Orwell", "categories": ["Fiction"]}
        )
        result = domain_event_to_sse(event)
        assert result["event"] == "metadata"
        data = json.loads(result["data"])
        assert data["book_title"] == "1984"

    def test_regions_ready(self):
        event = RegionsReady(job_id="abc", regions=[{"region_id": 1}], analysis_note="test")
        result = domain_event_to_sse(event)
        assert result["event"] == "regions"
        data = json.loads(result["data"])
        assert data["job_id"] == "abc"

    def test_itinerary_ready(self):
        event = ItineraryReady(itinerary={"cities": []})
        result = domain_event_to_sse(event)
        assert result["event"] == "itinerary"

    def test_workflow_error_with_phase(self):
        event = WorkflowError(message="fail", error_type="TestError", phase=Phase.BOOK_SEARCH)
        result = domain_event_to_sse(event)
        data = json.loads(result["data"])
        assert data["message"] == "fail"
        assert data["phase"] == 1

    def test_workflow_error_no_phase(self):
        event = WorkflowError(message="fail", error_type="TestError")
        result = domain_event_to_sse(event)
        data = json.loads(result["data"])
        assert data["phase"] is None

    def test_workflow_complete(self):
        event = WorkflowComplete(job_id="abc", token_usage={"total": 100})
        result = domain_event_to_sse(event)
        assert result["event"] == "done"
        data = json.loads(result["data"])
        assert data["job_id"] == "abc"


# =============================================================================
# Endpoint Tests (with mocked executor)
# =============================================================================


@pytest.fixture
def mock_executor():
    """Create a mock WorkflowExecutor."""
    from core.executor import WorkflowExecutor

    executor = MagicMock(spec=WorkflowExecutor)
    executor.session_service = AsyncMock()
    executor.config = MagicMock()
    executor.config.model_name = "gemini-2.0-flash-lite"
    executor.close = AsyncMock()
    return executor


@pytest.fixture
def mock_app_state(mock_executor):
    """Create a mock AppState for testing."""
    from api.dependencies import AppState

    mock_config = MagicMock()
    mock_config.model_name = "gemini-2.0-flash-lite"

    return AppState(config=mock_config, executor=mock_executor)


@pytest.fixture
def test_client(mock_app_state):
    """Create an async test client with mocked app state."""
    import api.dependencies as deps
    from api.app import create_app
    from httpx import AsyncClient, ASGITransport

    app = create_app()
    app.router.lifespan_context = _null_lifespan

    deps._app_state = mock_app_state

    client = AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    )
    return client


from contextlib import asynccontextmanager


@asynccontextmanager
async def _null_lifespan(app):
    yield


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, test_client):
        response = await test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["model_name"] == "gemini-2.0-flash-lite"


class TestStatusEndpoint:
    """Tests for GET /api/v1/itinerary/{job_id}/status (derived status)."""

    @pytest.mark.asyncio
    async def test_status_not_found(self, test_client, mock_executor):
        mock_executor.session_service.get_session.return_value = None
        response = await test_client.get("/api/v1/itinerary/bad-id/status")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_status_backend_error_returns_500(self, test_client, mock_executor):
        mock_executor.session_service.get_session.side_effect = RuntimeError("DB connection lost")
        response = await test_client.get("/api/v1/itinerary/some-id/status")
        assert response.status_code == 500
        assert "Failed to retrieve job status" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_status_derives_regions_ready(self, test_client, mock_executor):
        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {"book_title": "1984", "author": "George Orwell"},
            "region_analysis": {
                "regions": [{"region_id": 1, "region_name": "England"}]
            },
        }
        mock_executor.session_service.get_session.return_value = mock_session
        response = await test_client.get("/api/v1/itinerary/job-123/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "regions_ready"
        assert data["book_title"] == "1984"
        assert data["has_regions"] is True
        assert data["has_itinerary"] is False

    @pytest.mark.asyncio
    async def test_status_derives_completed(self, test_client, mock_executor):
        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {"book_title": "1984", "author": "George Orwell"},
            "region_analysis": {"regions": [{"region_id": 1}]},
            "selected_regions": [{"region_id": 1}],
            "final_itinerary": {"cities": [], "summary_text": "test"},
        }
        mock_executor.session_service.get_session.return_value = mock_session
        response = await test_client.get("/api/v1/itinerary/job-123/status")
        data = response.json()
        assert data["status"] == "completed"
        assert data["has_itinerary"] is True

    @pytest.mark.asyncio
    async def test_status_derives_discovering(self, test_client, mock_executor):
        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {"book_title": "1984", "author": "George Orwell"},
        }
        mock_executor.session_service.get_session.return_value = mock_session
        response = await test_client.get("/api/v1/itinerary/job-123/status")
        data = response.json()
        assert data["status"] == "discovering"

    @pytest.mark.asyncio
    async def test_status_derives_searching(self, test_client, mock_executor):
        mock_session = MagicMock()
        mock_session.state = {}
        mock_executor.session_service.get_session.return_value = mock_session
        response = await test_client.get("/api/v1/itinerary/job-123/status")
        data = response.json()
        assert data["status"] == "searching"

    @pytest.mark.asyncio
    async def test_status_derives_composing(self, test_client, mock_executor):
        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {"book_title": "1984", "author": "George Orwell"},
            "region_analysis": {"regions": [{"region_id": 1}]},
            "selected_regions": [{"region_id": 1}],
        }
        mock_executor.session_service.get_session.return_value = mock_session
        response = await test_client.get("/api/v1/itinerary/job-123/status")
        data = response.json()
        assert data["status"] == "composing"

    @pytest.mark.asyncio
    async def test_status_failed_marker_returns_failed(self, test_client, mock_executor):
        """Regression: job_failed=True in session state must yield status=failed."""
        mock_session = MagicMock()
        mock_session.state = {"job_failed": True}
        mock_executor.session_service.get_session.return_value = mock_session
        response = await test_client.get("/api/v1/itinerary/job-123/status")
        assert response.status_code == 200
        assert response.json()["status"] == "failed"

    @pytest.mark.asyncio
    async def test_status_failed_takes_priority_over_partial_state(self, test_client, mock_executor):
        """Regression: failed flag beats partial data (e.g. book_metadata present but job errored)."""
        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {"book_title": "1984", "author": "George Orwell"},
            "job_failed": True,
        }
        mock_executor.session_service.get_session.return_value = mock_session
        response = await test_client.get("/api/v1/itinerary/job-123/status")
        assert response.status_code == 200
        assert response.json()["status"] == "failed"

    @pytest.mark.asyncio
    async def test_status_failed_takes_priority_over_composing_state(self, test_client, mock_executor):
        """Regression: failed flag beats composing-phase partial state."""
        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {"book_title": "1984", "author": "George Orwell"},
            "region_analysis": {"regions": [{"region_id": 1}]},
            "selected_regions": [{"region_id": 1}],
            "job_failed": True,
        }
        mock_executor.session_service.get_session.return_value = mock_session
        response = await test_client.get("/api/v1/itinerary/job-123/status")
        assert response.status_code == 200
        assert response.json()["status"] == "failed"

    @pytest.mark.asyncio
    async def test_status_failed_beats_stale_itinerary(self, test_client, mock_executor):
        """Regression: job_failed=True must win over a stale final_itinerary from a prior
        successful compose run, so clients see the current failure rather than a past success."""
        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {"book_title": "1984", "author": "George Orwell"},
            "region_analysis": {"regions": [{"region_id": 1}]},
            "selected_regions": [{"region_id": 1}],
            "final_itinerary": {"cities": [], "summary_text": "done"},
            "job_failed": True,
        }
        mock_executor.session_service.get_session.return_value = mock_session
        response = await test_client.get("/api/v1/itinerary/job-123/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["has_itinerary"] is False


class TestDiscoverEndpoint:
    """Tests for POST /api/v1/itinerary/discover."""

    @pytest.mark.asyncio
    async def test_discover_session_create_failure(self, test_client, mock_executor):
        async def mock_discover(**kwargs):
            yield WorkflowError(message="Failed to initialize session", error_type="SessionError")
            yield WorkflowComplete(job_id="test-job")

        mock_executor.discover = mock_discover
        response = await test_client.post(
            "/api/v1/itinerary/discover", json={"book_title": "1984", "author": "George Orwell"}
        )
        assert response.status_code == 200
        events = _parse_sse_response(response.text)
        event_types = [e["event"] for e in events]
        assert "error" in event_types
        assert event_types[-1] == "done"
        error_event = next(e for e in events if e["event"] == "error")
        assert error_event["error_type"] == "SessionError"

    @pytest.mark.asyncio
    async def test_discover_book_not_found(self, test_client, mock_executor):
        async def mock_discover(**kwargs):
            yield ProgressEvent(phase=Phase.BOOK_SEARCH, step="Searching Google Books API")
            yield WorkflowError(
                message='Could not find "Nonexistent Book XYZ" in Google Books.',
                error_type="BookNotFound", phase=Phase.BOOK_SEARCH,
            )
            yield WorkflowComplete(job_id="test-job")

        mock_executor.discover = mock_discover
        response = await test_client.post(
            "/api/v1/itinerary/discover", json={"book_title": "Nonexistent Book XYZ", "author": "Unknown Author"},
        )
        assert response.status_code == 200
        events = _parse_sse_response(response.text)
        event_types = [e["event"] for e in events]
        assert "progress" in event_types
        assert "error" in event_types
        assert event_types[-1] == "done"
        error_event = next(e for e in events if e["event"] == "error")
        assert "Nonexistent Book XYZ" in error_event["message"]
        assert error_event["phase"] == 1

    @pytest.mark.asyncio
    async def test_discover_search_api_error(self, test_client, mock_executor):
        async def mock_discover(**kwargs):
            yield ProgressEvent(phase=Phase.BOOK_SEARCH, step="Searching Google Books API")
            yield WorkflowError(
                message="Could not search Google Books API: Network error",
                error_type="ConnectionError", phase=Phase.BOOK_SEARCH,
            )
            yield WorkflowComplete(job_id="test-job")

        mock_executor.discover = mock_discover
        response = await test_client.post(
            "/api/v1/itinerary/discover", json={"book_title": "1984", "author": "George Orwell"}
        )
        assert response.status_code == 200
        events = _parse_sse_response(response.text)
        assert events[-1]["event"] == "done"
        error_event = next(e for e in events if e["event"] == "error")
        assert "Network error" in error_event["message"]
        assert error_event["error_type"] == "ConnectionError"

    @pytest.mark.asyncio
    async def test_discover_success_streams_events(self, test_client, mock_executor):
        async def mock_discover(**kwargs):
            yield ProgressEvent(phase=Phase.BOOK_SEARCH, step="Searching Google Books API")
            yield MetadataReady(
                metadata={
                    "book_title": "1984", "author": "George Orwell",
                    "description": "A dystopian novel", "categories": ["Fiction"],
                }
            )
            yield ProgressEvent(phase=Phase.DISCOVERY, step="Starting location discovery")
            yield ProgressEvent(
                phase=Phase.DISCOVERY, step="Analyzing geographic regions", detail="region_analyzer",
            )
            yield RegionsReady(
                job_id="test-job-id",
                regions=[{"region_id": 1, "region_name": "England"}],
                analysis_note="test",
            )
            yield WorkflowComplete(job_id="test-job-id")

        mock_executor.discover = mock_discover
        response = await test_client.post(
            "/api/v1/itinerary/discover", json={"book_title": "1984", "author": "George Orwell"},
        )
        assert response.status_code == 200
        events = _parse_sse_response(response.text)
        event_types = [e["event"] for e in events]
        assert event_types[0] == "progress"
        assert "metadata" in event_types
        assert "regions" in event_types
        assert event_types[-1] == "done"
        metadata = next(e for e in events if e["event"] == "metadata")
        assert metadata["book_title"] == "1984"
        assert metadata["author"] == "George Orwell"
        done = next(e for e in events if e["event"] == "done")
        assert done["job_id"] == "test-job-id"


class TestComposeEndpoint:
    """Tests for POST /api/v1/itinerary/{job_id}/compose."""

    @pytest.mark.asyncio
    async def test_compose_session_backend_error(self, test_client, mock_executor):
        async def mock_compose(**kwargs):
            yield WorkflowError(message="Failed to retrieve session", error_type="SessionError")
            yield WorkflowComplete(job_id="bad-job")

        mock_executor.compose = mock_compose
        response = await test_client.post(
            "/api/v1/itinerary/bad-job/compose", json={"region_ids": [1]},
        )
        assert response.status_code == 200
        events = _parse_sse_response(response.text)
        assert events[-1]["event"] == "done"
        error_event = next(e for e in events if e["event"] == "error")
        assert error_event["error_type"] == "SessionError"

    @pytest.mark.asyncio
    async def test_compose_session_returns_none(self, test_client, mock_executor):
        async def mock_compose(**kwargs):
            yield WorkflowError(
                message="Job bad-job not found. Run discover first.", error_type="JobNotFound",
            )
            yield WorkflowComplete(job_id="bad-job")

        mock_executor.compose = mock_compose
        response = await test_client.post(
            "/api/v1/itinerary/bad-job/compose", json={"region_ids": [1]},
        )
        assert response.status_code == 200
        events = _parse_sse_response(response.text)
        assert events[-1]["event"] == "done"
        error_event = next(e for e in events if e["event"] == "error")
        assert "not found" in error_event["message"].lower()
        assert error_event["error_type"] == "JobNotFound"

    @pytest.mark.asyncio
    async def test_compose_no_regions_in_session(self, test_client, mock_executor):
        async def mock_compose(**kwargs):
            yield WorkflowError(
                message="No regions found in session. Discovery may not have completed.",
                error_type="NoRegions", phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id="job-123")

        mock_executor.compose = mock_compose
        response = await test_client.post(
            "/api/v1/itinerary/job-123/compose", json={"region_ids": [1]},
        )
        assert response.status_code == 200
        events = _parse_sse_response(response.text)
        error_event = next(e for e in events if e["event"] == "error")
        assert "No regions" in error_event["message"]

    @pytest.mark.asyncio
    async def test_compose_invalid_region_ids(self, test_client, mock_executor):
        async def mock_compose(**kwargs):
            yield WorkflowError(
                message="Invalid region_ids: [99]. Valid IDs: [1, 2]",
                error_type="InvalidRegionIds", phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id="job-123")

        mock_executor.compose = mock_compose
        response = await test_client.post(
            "/api/v1/itinerary/job-123/compose", json={"region_ids": [99]},
        )
        assert response.status_code == 200
        events = _parse_sse_response(response.text)
        error_event = next(e for e in events if e["event"] == "error")
        assert error_event["error_type"] == "InvalidRegionIds"
        assert "99" in error_event["message"]

    @pytest.mark.asyncio
    async def test_compose_success(self, test_client, mock_executor):
        async def mock_compose(**kwargs):
            yield ProgressEvent(
                phase=Phase.COMPOSITION, step="Creating personalized itinerary",
                detail="1 region(s) selected",
            )
            yield ItineraryReady(itinerary={
                "cities": [{
                    "name": "London", "country": "UK", "days_suggested": 3,
                    "overview": "Explore Orwell's London", "stops": [],
                }],
                "summary_text": "A journey through 1984",
            })
            yield WorkflowComplete(job_id="job-123")

        mock_executor.compose = mock_compose
        response = await test_client.post(
            "/api/v1/itinerary/job-123/compose", json={"region_ids": [1]},
        )
        assert response.status_code == 200
        events = _parse_sse_response(response.text)
        event_types = [e["event"] for e in events]
        assert "progress" in event_types
        assert "itinerary" in event_types
        assert event_types[-1] == "done"
        itinerary_event = next(e for e in events if e["event"] == "itinerary")
        assert itinerary_event["itinerary"]["cities"][0]["name"] == "London"

    @pytest.mark.asyncio
    async def test_compose_extraction_failure(self, test_client, mock_executor):
        async def mock_compose(**kwargs):
            yield ProgressEvent(phase=Phase.COMPOSITION, step="Creating personalized itinerary")
            yield WorkflowError(
                message="Failed to extract itinerary from agent response",
                error_type="ExtractionError", phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id="job-123")

        mock_executor.compose = mock_compose
        response = await test_client.post(
            "/api/v1/itinerary/job-123/compose", json={"region_ids": [1]},
        )
        assert response.status_code == 200
        events = _parse_sse_response(response.text)
        event_types = [e["event"] for e in events]
        assert "itinerary" not in event_types
        assert "error" in event_types


# =============================================================================
# CORS Configuration Tests
# =============================================================================


class TestCORSConfiguration:
    def test_wildcard_origin_disables_credentials(self):
        import os
        from unittest.mock import patch as mock_patch

        with mock_patch.dict(os.environ, {"CORS_ORIGINS": "*"}):
            from api.app import create_app
            app = create_app()
            cors_middleware = None
            for middleware in app.user_middleware:
                if middleware.cls is CORSMiddleware:
                    cors_middleware = middleware
                    break
            assert cors_middleware is not None
            assert cors_middleware.kwargs["allow_credentials"] is False

    def test_specific_origins_enable_credentials(self):
        import os
        from unittest.mock import patch as mock_patch

        with mock_patch.dict(
            os.environ,
            {"CORS_ORIGINS": "http://localhost:3000,https://app.example.com"},
        ):
            from api.app import create_app
            app = create_app()
            cors_middleware = None
            for middleware in app.user_middleware:
                if middleware.cls is CORSMiddleware:
                    cors_middleware = middleware
                    break
            assert cors_middleware is not None
            assert cors_middleware.kwargs["allow_credentials"] is True


# =============================================================================
# Regression Tests
# =============================================================================


class TestBookContextTimePeriod:
    def test_book_context_null_time_period(self):
        from models.book import BookContext
        ctx = BookContext(primary_locations=["Paris"], time_period=None, themes=["war"])
        assert ctx.time_period is None

    def test_book_context_missing_time_period_defaults_none(self):
        from models.book import BookContext
        ctx = BookContext(primary_locations=["Paris"], themes=["war"])
        assert ctx.time_period is None

    def test_book_context_with_time_period(self):
        from models.book import BookContext
        ctx = BookContext(
            primary_locations=["Paris"],
            time_period="World War II (1940-1944)",
            themes=["war"],
        )
        assert ctx.time_period == "World War II (1940-1944)"


class TestEmptyRegionIdsRejected:
    @pytest.mark.asyncio
    async def test_compose_empty_region_ids_returns_422(self, test_client):
        response = await test_client.post(
            "/api/v1/itinerary/job-123/compose", json={"region_ids": []},
        )
        assert response.status_code == 422


class TestBlankBookTitleRejected:
    @pytest.mark.asyncio
    async def test_discover_blank_book_title_returns_422(self, test_client):
        response = await test_client.post(
            "/api/v1/itinerary/discover", json={"book_title": "   "},
        )
        assert response.status_code == 422


class TestDiscoveryProgressMapping:
    def test_reader_profile_agent_in_progress_map(self):
        from core.executor import DISCOVERY_AGENT_STEPS
        assert "reader_profile_agent" in DISCOVERY_AGENT_STEPS
        assert "reader_profile" not in DISCOVERY_AGENT_STEPS


# =============================================================================
# Test Helpers
# =============================================================================


def _parse_sse_response(text: str) -> list[dict]:
    """Parse SSE response text into list of event dicts."""
    events = []
    current_event = None
    current_data = None

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            current_data = line[5:].strip()
        elif line == "" and current_data is not None:
            try:
                parsed = json.loads(current_data)
                if current_event and "event" not in parsed:
                    parsed["event"] = current_event
                events.append(parsed)
            except json.JSONDecodeError:
                pass
            current_event = None
            current_data = None

    if current_data is not None:
        try:
            parsed = json.loads(current_data)
            if current_event and "event" not in parsed:
                parsed["event"] = current_event
            events.append(parsed)
        except json.JSONDecodeError:
            pass

    return events


# =============================================================================
# RecommendBooksRequest Tests
# =============================================================================


class TestRecommendBooksRequest:
    def test_valid_request(self):
        req = RecommendBooksRequest(
            action_id="chip-uuid-123",
            action_label="Find books like this",
            action_prompt="",
        )
        assert req.action_id == "chip-uuid-123"
        assert req.action_label == "Find books like this"

    def test_default_action_prompt(self):
        req = RecommendBooksRequest(
            action_id="chip-uuid-123",
            action_label="Find books like this",
        )
        assert req.action_prompt == ""

    def test_missing_action_id_raises(self):
        with pytest.raises(ValidationError):
            RecommendBooksRequest(action_label="Find books like this")

    def test_blank_action_id_raises(self):
        with pytest.raises(ValidationError):
            RecommendBooksRequest(action_id="", action_label="Find books like this")

    def test_action_label_max_length(self):
        with pytest.raises(ValidationError):
            RecommendBooksRequest(action_id="x", action_label="a" * 101)


# =============================================================================
# SSEBookRecommendationsEvent Tests
# =============================================================================


class TestSSEBookRecommendationsEvent:
    def _make_rec(self, title="Book"):
        return {
            "title": title,
            "author": "Author",
            "reason": "A reason.",
            "recommendation_basis": "themes",
        }

    def test_basic_event(self):
        event = SSEBookRecommendationsEvent(
            recommendations=[self._make_rec("1984"), self._make_rec("Brave New World")],
            book_recommendation_count=1,
        )
        data = json.loads(event.model_dump_json())
        assert data["event"] == "book_recommendations"
        assert len(data["recommendations"]) == 2
        assert data["book_recommendation_count"] == 1

    def test_event_literal_is_fixed(self):
        event = SSEBookRecommendationsEvent(recommendations=[], book_recommendation_count=0)
        assert event.event == "book_recommendations"

    def test_empty_recommendations(self):
        event = SSEBookRecommendationsEvent(recommendations=[], book_recommendation_count=0)
        data = json.loads(event.model_dump_json())
        assert data["recommendations"] == []


# =============================================================================
# domain_event_to_sse - BookRecommendationsReady
# =============================================================================


class TestDomainEventToSSEBookRecommendations:
    def _make_rec(self, title="Book"):
        return {
            "title": title,
            "author": "Author",
            "reason": "A reason.",
            "recommendation_basis": "destination",
        }

    def test_book_recommendations_ready_event(self):
        event = BookRecommendationsReady(
            recommendations=[self._make_rec("The Remains of the Day")],
            book_recommendation_count=1,
        )
        sse = domain_event_to_sse(event)
        assert sse["event"] == "book_recommendations"
        data = json.loads(sse["data"])
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["title"] == "The Remains of the Day"
        assert data["book_recommendation_count"] == 1

    def test_multiple_recommendations(self):
        recs = [self._make_rec(f"Book {i}") for i in range(5)]
        event = BookRecommendationsReady(
            recommendations=recs,
            book_recommendation_count=2,
        )
        sse = domain_event_to_sse(event)
        data = json.loads(sse["data"])
        assert len(data["recommendations"]) == 5
        assert data["book_recommendation_count"] == 2
