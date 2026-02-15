"""
Unit tests for FastAPI SSE API layer.

Tests API models, endpoint responses, and SSE streaming with mocked Runner.
"""

import json

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    DiscoverRequest,
    ComposeRequest,
    SSEProgressEvent,
    SSEMetadataEvent,
    SSERegionsEvent,
    SSEItineraryEvent,
    SSEErrorEvent,
    SSEDoneEvent,
    HealthResponse,
    JobStatusResponse,
    JobStatus,
)
from api.streaming import _sse


# =============================================================================
# Model Validation Tests
# =============================================================================


class TestDiscoverRequest:
    """Tests for DiscoverRequest model."""

    def test_minimal_request(self):
        req = DiscoverRequest(book_title="1984")
        assert req.book_title == "1984"
        assert req.author is None
        assert req.preferences is None
        assert req.user_id == "api_user"

    def test_full_request(self):
        req = DiscoverRequest(
            book_title="1984",
            author="George Orwell",
            preferences={"budget": "luxury"},
            user_id="alice",
        )
        assert req.author == "George Orwell"
        assert req.preferences["budget"] == "luxury"
        assert req.user_id == "alice"

    def test_missing_book_title_raises(self):
        with pytest.raises(ValidationError):
            DiscoverRequest()

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

    def test_empty_region_ids(self):
        req = ComposeRequest(region_ids=[])
        assert req.region_ids == []

    def test_default_user_id(self):
        req = ComposeRequest(region_ids=[1])
        assert req.user_id == "api_user"

    def test_missing_region_ids_raises(self):
        with pytest.raises(ValidationError):
            ComposeRequest()


# =============================================================================
# SSE Event Model Tests
# =============================================================================


class TestSSEProgressEvent:
    """Tests for SSEProgressEvent model."""

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
    """Tests for SSEMetadataEvent model."""

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
        assert data["author"] == "George Orwell"
        assert len(data["categories"]) == 2

    def test_minimal_metadata(self):
        event = SSEMetadataEvent(book_title="1984", author="George Orwell")
        data = json.loads(event.model_dump_json())
        assert data["description"] is None
        assert data["categories"] == []


class TestSSERegionsEvent:
    """Tests for SSERegionsEvent model."""

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
        assert data["analysis_note"] == "Grouped by proximity"

    def test_empty_regions(self):
        event = SSERegionsEvent(job_id="abc", regions=[])
        data = json.loads(event.model_dump_json())
        assert data["regions"] == []


class TestSSEItineraryEvent:
    """Tests for SSEItineraryEvent model."""

    def test_itinerary_event(self):
        itinerary = {
            "cities": [{"name": "London", "country": "UK"}],
            "summary_text": "A literary journey",
        }
        event = SSEItineraryEvent(itinerary=itinerary)
        data = json.loads(event.model_dump_json())
        assert data["event"] == "itinerary"
        assert data["itinerary"]["cities"][0]["name"] == "London"


class TestSSEErrorEvent:
    """Tests for SSEErrorEvent model."""

    def test_error_with_phase(self):
        event = SSEErrorEvent(
            message="Something failed",
            error_type="TimeoutError",
            phase=2,
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
    """Tests for SSEDoneEvent model."""

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
    """Tests for HealthResponse model."""

    def test_defaults(self):
        resp = HealthResponse()
        assert resp.status == "healthy"
        assert resp.version == "0.1.0"

    def test_with_model_name(self):
        resp = HealthResponse(model_name="gemini-2.0-flash")
        assert resp.model_name == "gemini-2.0-flash"


class TestJobStatusResponse:
    """Tests for JobStatusResponse model."""

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
            job_id="test",
            status=JobStatus.COMPLETED,
            has_itinerary=True,
        )
        assert resp.has_itinerary is True


# =============================================================================
# SSE Helper Tests
# =============================================================================


class TestSSEHelper:
    """Tests for the _sse() helper function."""

    def test_sse_format(self):
        result = _sse("progress", '{"phase": 1}')
        assert result == {"event": "progress", "data": '{"phase": 1}'}

    def test_sse_with_model(self):
        event = SSEProgressEvent(phase=1, step="test")
        result = _sse("progress", event.model_dump_json())
        assert result["event"] == "progress"
        parsed = json.loads(result["data"])
        assert parsed["phase"] == 1


# =============================================================================
# Endpoint Tests (with mocked workflow)
# =============================================================================


@pytest.fixture
def mock_app_state():
    """Create a mock AppState for testing."""
    from api.dependencies import AppState

    mock_config = MagicMock()
    mock_config.model_name = "gemini-2.0-flash-lite"
    mock_config.workflow_timeout = 300
    mock_config.langfuse_secret_key = None
    mock_config.langfuse_public_key = None
    mock_config.langfuse_host = None

    mock_session_service = AsyncMock()
    mock_model = MagicMock()
    mock_langfuse = MagicMock()
    mock_langfuse.enabled = False

    return AppState(
        config=mock_config,
        model=mock_model,
        session_service=mock_session_service,
        langfuse_plugin=mock_langfuse,
    )


@pytest.fixture
def test_client(mock_app_state):
    """Create an async test client with mocked app state."""
    import api.dependencies as deps
    from api.app import create_app
    from httpx import AsyncClient, ASGITransport

    # Skip lifespan to avoid real initialization
    app = create_app()
    app.router.lifespan_context = _null_lifespan

    # Set the mock app state
    deps._app_state = mock_app_state

    client = AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    )
    return client


from contextlib import asynccontextmanager


@asynccontextmanager
async def _null_lifespan(app):
    """No-op lifespan for testing (skip real initialization)."""
    yield


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, test_client):
        response = await test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["model_name"] == "gemini-2.0-flash-lite"


class TestStatusEndpoint:
    """Tests for GET /api/v1/itinerary/{job_id}/status."""

    @pytest.mark.asyncio
    async def test_status_not_found(self, test_client, mock_app_state):
        mock_app_state.session_service.get_session.return_value = None
        response = await test_client.get("/api/v1/itinerary/bad-id/status")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_status_backend_error_returns_500(
        self, test_client, mock_app_state
    ):
        """Backend exceptions should return 500, not 404."""
        mock_app_state.session_service.get_session.side_effect = RuntimeError(
            "DB connection lost"
        )
        response = await test_client.get("/api/v1/itinerary/some-id/status")
        assert response.status_code == 500
        assert "Failed to retrieve job status" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_status_returns_none(self, test_client, mock_app_state):
        mock_app_state.session_service.get_session.return_value = None
        response = await test_client.get("/api/v1/itinerary/bad-id/status")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_status_regions_ready(self, test_client, mock_app_state):
        mock_session = MagicMock()
        mock_session.state = {
            "_job_status": "regions_ready",
            "book_metadata": {
                "book_title": "1984",
                "author": "George Orwell",
            },
            "region_analysis": {
                "regions": [{"region_id": 1, "region_name": "England"}]
            },
        }
        mock_app_state.session_service.get_session.return_value = mock_session

        response = await test_client.get("/api/v1/itinerary/job-123/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "regions_ready"
        assert data["book_title"] == "1984"
        assert data["has_regions"] is True
        assert data["has_itinerary"] is False


class TestDiscoverEndpoint:
    """Tests for POST /api/v1/itinerary/discover."""

    @pytest.mark.asyncio
    async def test_discover_session_create_failure(
        self, test_client, mock_app_state
    ):
        """When session backend fails, stream should emit error + done, not crash."""
        mock_app_state.session_service.create_session.side_effect = (
            RuntimeError("DB connection refused")
        )
        response = await test_client.post(
            "/api/v1/itinerary/discover",
            json={"book_title": "1984"},
        )
        assert response.status_code == 200

        events = _parse_sse_response(response.text)
        event_types = [e["event"] for e in events]
        assert "error" in event_types
        assert event_types[-1] == "done"

        error_event = next(e for e in events if e["event"] == "error")
        assert error_event["error_type"] == "SessionError"

    @pytest.mark.asyncio
    async def test_discover_book_not_found(self, test_client, mock_app_state):
        """When Google Books returns no results, stream should emit error + done."""
        with patch(
            "api.streaming.search_books_with_retry", return_value=[]
        ):
            response = await test_client.post(
                "/api/v1/itinerary/discover",
                json={"book_title": "Nonexistent Book XYZ"},
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
    async def test_discover_search_api_error(
        self, test_client, mock_app_state
    ):
        """When Google Books API fails, stream should emit error + done."""
        with patch(
            "api.streaming.search_books_with_retry",
            side_effect=ConnectionError("Network error"),
        ):
            response = await test_client.post(
                "/api/v1/itinerary/discover",
                json={"book_title": "1984"},
            )
            assert response.status_code == 200

            events = _parse_sse_response(response.text)
            event_types = [e["event"] for e in events]
            assert event_types[-1] == "done"

            error_event = next(e for e in events if e["event"] == "error")
            assert "Network error" in error_event["message"]
            assert error_event["error_type"] == "ConnectionError"

    @pytest.mark.asyncio
    async def test_discover_success_streams_events(
        self, test_client, mock_app_state
    ):
        """Successful discover should emit: progress, metadata, progress..., regions, done."""
        from models.book import BookInfo

        mock_book = BookInfo(
            title="1984",
            authors=["George Orwell"],
            description="A dystopian novel",
            published_date="1949",
            categories=["Fiction"],
        )

        mock_session = MagicMock()
        mock_session.state = {}
        mock_app_state.session_service.get_session.return_value = mock_session

        # Mock the ADK Runner to yield a simple event
        mock_event = MagicMock()
        mock_event.author = "region_analyzer"
        mock_event.is_final_response.return_value = False

        async def mock_run_async(*args, **kwargs):
            yield mock_event

        mock_runner_instance = MagicMock()
        mock_runner_instance.run_async = mock_run_async
        mock_runner_instance.__aenter__ = AsyncMock(
            return_value=mock_runner_instance
        )
        mock_runner_instance.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "api.streaming.search_books_with_retry",
                return_value=[mock_book],
            ),
            patch("api.streaming.Runner", return_value=mock_runner_instance),
            patch("api.streaming.create_discovery_workflow"),
            patch("api.streaming.LangfusePlugin") as mock_lf,
        ):
            mock_lf.return_value.enabled = False
            response = await test_client.post(
                "/api/v1/itinerary/discover",
                json={"book_title": "1984", "author": "George Orwell"},
            )
            assert response.status_code == 200

            events = _parse_sse_response(response.text)
            event_types = [e["event"] for e in events]

            # Verify event sequence
            assert event_types[0] == "progress"  # Phase 1 start
            assert "metadata" in event_types
            assert "regions" in event_types
            assert event_types[-1] == "done"

            # Verify metadata content
            metadata = next(e for e in events if e["event"] == "metadata")
            assert metadata["book_title"] == "1984"
            assert metadata["author"] == "George Orwell"

            # Verify done has job_id
            done = next(e for e in events if e["event"] == "done")
            assert "job_id" in done
            assert len(done["job_id"]) > 0


class TestComposeEndpoint:
    """Tests for POST /api/v1/itinerary/{job_id}/compose."""

    @pytest.mark.asyncio
    async def test_compose_session_not_found(
        self, test_client, mock_app_state
    ):
        """When job_id doesn't exist, stream should emit error + done."""
        mock_app_state.session_service.get_session.side_effect = Exception(
            "Not found"
        )
        response = await test_client.post(
            "/api/v1/itinerary/bad-job/compose",
            json={"region_ids": [1]},
        )
        assert response.status_code == 200

        events = _parse_sse_response(response.text)
        event_types = [e["event"] for e in events]
        assert event_types[-1] == "done"

        error_event = next(e for e in events if e["event"] == "error")
        assert "not found" in error_event["message"].lower()
        assert error_event["error_type"] == "JobNotFound"

    @pytest.mark.asyncio
    async def test_compose_session_returns_none(
        self, test_client, mock_app_state
    ):
        """When get_session returns None (no exception), should emit error + done, not crash."""
        mock_app_state.session_service.get_session.return_value = None
        response = await test_client.post(
            "/api/v1/itinerary/bad-job/compose",
            json={"region_ids": [1]},
        )
        assert response.status_code == 200

        events = _parse_sse_response(response.text)
        event_types = [e["event"] for e in events]
        assert event_types[-1] == "done"

        error_event = next(e for e in events if e["event"] == "error")
        assert "not found" in error_event["message"].lower()
        assert error_event["error_type"] == "JobNotFound"

    @pytest.mark.asyncio
    async def test_compose_no_regions_in_session(
        self, test_client, mock_app_state
    ):
        """When session has no region_analysis, stream should emit error + done."""
        mock_session = MagicMock()
        mock_session.state = {"book_metadata": {"book_title": "1984"}}
        mock_app_state.session_service.get_session.return_value = mock_session

        response = await test_client.post(
            "/api/v1/itinerary/job-123/compose",
            json={"region_ids": [1]},
        )
        assert response.status_code == 200

        events = _parse_sse_response(response.text)
        event_types = [e["event"] for e in events]
        assert event_types[-1] == "done"

        error_event = next(e for e in events if e["event"] == "error")
        assert "No regions" in error_event["message"]

    @pytest.mark.asyncio
    async def test_compose_invalid_region_ids(
        self, test_client, mock_app_state
    ):
        """When region_ids don't match any discovered regions, emit error + done."""
        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {"book_title": "1984", "author": "George Orwell"},
            "region_analysis": {
                "regions": [
                    {"region_id": 1, "region_name": "England", "cities": []},
                    {"region_id": 2, "region_name": "Spain", "cities": []},
                ]
            },
        }
        mock_app_state.session_service.get_session.return_value = mock_session

        response = await test_client.post(
            "/api/v1/itinerary/job-123/compose",
            json={"region_ids": [99]},
        )
        assert response.status_code == 200

        events = _parse_sse_response(response.text)
        event_types = [e["event"] for e in events]
        assert "error" in event_types
        assert event_types[-1] == "done"

        error_event = next(e for e in events if e["event"] == "error")
        assert error_event["error_type"] == "InvalidRegionIds"
        assert "99" in error_event["message"]

    @pytest.mark.asyncio
    async def test_compose_malformed_region_ids_in_session(
        self, test_client, mock_app_state
    ):
        """Regions with None/missing region_id should be excluded from valid set."""
        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {"book_title": "1984", "author": "George Orwell"},
            "region_analysis": {
                "regions": [
                    {"region_id": 1, "region_name": "England", "cities": []},
                    {"region_name": "Unknown", "cities": []},  # missing region_id
                    {"region_id": None, "region_name": "Bad", "cities": []},
                ]
            },
        }
        mock_app_state.session_service.get_session.return_value = mock_session

        # region_id=99 is invalid; only region_id=1 is valid (None excluded)
        response = await test_client.post(
            "/api/v1/itinerary/job-123/compose",
            json={"region_ids": [99]},
        )
        assert response.status_code == 200

        events = _parse_sse_response(response.text)
        error_event = next(e for e in events if e["event"] == "error")
        assert error_event["error_type"] == "InvalidRegionIds"
        # Valid IDs should only contain 1, not None
        assert "1" in error_event["message"]

    @pytest.mark.asyncio
    async def test_compose_success(self, test_client, mock_app_state):
        """Successful compose should emit: progress, itinerary, done."""
        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {
                "book_title": "1984",
                "author": "George Orwell",
            },
            "region_analysis": {
                "regions": [
                    {
                        "region_id": 1,
                        "region_name": "England",
                        "cities": [{"name": "London", "country": "UK"}],
                    }
                ]
            },
        }
        mock_session.user_id = "api_user"
        mock_app_state.session_service.get_session.return_value = mock_session

        # Create mock final event with itinerary JSON
        itinerary_json = json.dumps(
            {
                "cities": [
                    {
                        "name": "London",
                        "country": "UK",
                        "days_suggested": 3,
                        "overview": "Explore Orwell's London",
                        "stops": [],
                    }
                ],
                "summary_text": "A journey through 1984",
            }
        )

        mock_part = MagicMock()
        mock_part.text = itinerary_json

        mock_final_event = MagicMock()
        mock_final_event.author = "trip_composer"
        mock_final_event.is_final_response.return_value = True
        mock_final_event.content.parts = [mock_part]

        async def mock_run_async(*args, **kwargs):
            yield mock_final_event

        mock_runner_instance = MagicMock()
        mock_runner_instance.run_async = mock_run_async
        mock_runner_instance.__aenter__ = AsyncMock(
            return_value=mock_runner_instance
        )
        mock_runner_instance.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("api.streaming.Runner", return_value=mock_runner_instance),
            patch("api.streaming.create_composition_workflow"),
            patch("api.streaming.LangfusePlugin") as mock_lf,
        ):
            mock_lf.return_value.enabled = False
            response = await test_client.post(
                "/api/v1/itinerary/job-123/compose",
                json={"region_ids": [1]},
            )
            assert response.status_code == 200

            events = _parse_sse_response(response.text)
            event_types = [e["event"] for e in events]

            assert "progress" in event_types
            assert "itinerary" in event_types
            assert event_types[-1] == "done"

            # Verify itinerary content
            itinerary_event = next(
                e for e in events if e["event"] == "itinerary"
            )
            assert itinerary_event["itinerary"]["cities"][0]["name"] == "London"


# =============================================================================
# Job Status Persistence Tests
# =============================================================================


class TestJobStatusFailedPersistence:
    """Verify _job_status='failed' is set on error paths."""

    @pytest.mark.asyncio
    async def test_discover_book_not_found_sets_failed(
        self, test_client, mock_app_state
    ):
        mock_session = MagicMock()
        mock_session.state = {"_job_status": "searching"}
        mock_app_state.session_service.get_session.return_value = mock_session

        with patch(
            "api.streaming.search_books_with_retry", return_value=[]
        ):
            await test_client.post(
                "/api/v1/itinerary/discover",
                json={"book_title": "Nonexistent"},
            )
        assert mock_session.state["_job_status"] == "failed"

    @pytest.mark.asyncio
    async def test_discover_api_error_sets_failed(
        self, test_client, mock_app_state
    ):
        mock_session = MagicMock()
        mock_session.state = {"_job_status": "searching"}
        mock_app_state.session_service.get_session.return_value = mock_session

        with patch(
            "api.streaming.search_books_with_retry",
            side_effect=ConnectionError("fail"),
        ):
            await test_client.post(
                "/api/v1/itinerary/discover",
                json={"book_title": "1984"},
            )
        assert mock_session.state["_job_status"] == "failed"

    @pytest.mark.asyncio
    async def test_compose_extraction_failure_sets_failed(
        self, test_client, mock_app_state
    ):
        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {"book_title": "1984", "author": "George Orwell"},
            "region_analysis": {
                "regions": [
                    {"region_id": 1, "region_name": "England", "cities": []}
                ]
            },
        }
        mock_app_state.session_service.get_session.return_value = mock_session

        # Final event with no parseable JSON
        mock_part = MagicMock()
        mock_part.text = "Sorry, I cannot do that."

        mock_final_event = MagicMock()
        mock_final_event.author = "trip_composer"
        mock_final_event.is_final_response.return_value = True
        mock_final_event.content.parts = [mock_part]

        async def mock_run_async(*args, **kwargs):
            yield mock_final_event

        mock_runner_instance = MagicMock()
        mock_runner_instance.run_async = mock_run_async
        mock_runner_instance.__aenter__ = AsyncMock(
            return_value=mock_runner_instance
        )
        mock_runner_instance.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("api.streaming.Runner", return_value=mock_runner_instance),
            patch("api.streaming.create_composition_workflow"),
            patch("api.streaming.LangfusePlugin") as mock_lf,
        ):
            mock_lf.return_value.enabled = False
            await test_client.post(
                "/api/v1/itinerary/job-123/compose",
                json={"region_ids": [1]},
            )
        assert mock_session.state["_job_status"] == "failed"


# =============================================================================
# Cancellation Tests
# =============================================================================


class TestCancellationHandling:
    """Verify CancelledError sets _job_status='failed' on disconnect."""

    @pytest.mark.asyncio
    async def test_discover_cancellation_sets_failed(self, mock_app_state):
        """discover_stream should persist failed status on CancelledError."""
        import asyncio
        from api.streaming import discover_stream

        mock_session = MagicMock()
        mock_session.state = {"_job_status": "discovering"}
        mock_app_state.session_service.get_session.return_value = mock_session

        # Runner that raises CancelledError mid-stream
        async def mock_run_async(*args, **kwargs):
            raise asyncio.CancelledError()
            yield  # noqa: unreachable — makes this an async generator

        mock_runner = MagicMock()
        mock_runner.run_async = mock_run_async
        mock_runner.__aenter__ = AsyncMock(return_value=mock_runner)
        mock_runner.__aexit__ = AsyncMock(return_value=False)

        from models.book import BookInfo

        mock_book = BookInfo(
            title="1984", authors=["George Orwell"], description="test"
        )

        with (
            patch(
                "api.streaming.search_books_with_retry",
                return_value=[mock_book],
            ),
            patch("api.streaming.Runner", return_value=mock_runner),
            patch("api.streaming.create_discovery_workflow"),
            patch("api.streaming.LangfusePlugin") as mock_lf,
        ):
            mock_lf.return_value.enabled = False
            gen = discover_stream(
                book_title="1984",
                author="George Orwell",
                preferences=None,
                user_id="api_user",
                app_state=mock_app_state,
            )
            with pytest.raises(asyncio.CancelledError):
                async for _ in gen:
                    pass

        assert mock_session.state["_job_status"] == "failed"

    @pytest.mark.asyncio
    async def test_compose_cancellation_sets_failed(self, mock_app_state):
        """compose_stream should persist failed status on CancelledError."""
        import asyncio
        from api.streaming import compose_stream

        mock_session = MagicMock()
        mock_session.state = {
            "book_metadata": {"book_title": "1984", "author": "George Orwell"},
            "region_analysis": {
                "regions": [
                    {"region_id": 1, "region_name": "England", "cities": []}
                ]
            },
            "_job_status": "composing",
        }
        mock_app_state.session_service.get_session.return_value = mock_session

        async def mock_run_async(*args, **kwargs):
            raise asyncio.CancelledError()
            yield  # noqa: unreachable

        mock_runner = MagicMock()
        mock_runner.run_async = mock_run_async
        mock_runner.__aenter__ = AsyncMock(return_value=mock_runner)
        mock_runner.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("api.streaming.Runner", return_value=mock_runner),
            patch("api.streaming.create_composition_workflow"),
            patch("api.streaming.LangfusePlugin") as mock_lf,
        ):
            mock_lf.return_value.enabled = False
            gen = compose_stream(
                job_id="job-123",
                region_ids=[1],
                user_id="api_user",
                app_state=mock_app_state,
            )
            with pytest.raises(asyncio.CancelledError):
                async for _ in gen:
                    pass

        assert mock_session.state["_job_status"] == "failed"


# =============================================================================
# CORS Configuration Tests
# =============================================================================


class TestCORSConfiguration:
    """Tests for CORS middleware configuration."""

    def test_wildcard_origin_disables_credentials(self):
        """CORS_ORIGINS='*' should set allow_credentials=False."""
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
        """Specific CORS origins should enable allow_credentials."""
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
# Test Helpers
# =============================================================================


def _parse_sse_response(text: str) -> list[dict]:
    """Parse SSE response text into list of event dicts.

    Handles the SSE format:
        event: <type>
        data: <json>

    Returns:
        List of parsed JSON data dicts with 'event' key included
    """
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

    # Handle last event if no trailing newline
    if current_data is not None:
        try:
            parsed = json.loads(current_data)
            if current_event and "event" not in parsed:
                parsed["event"] = current_event
            events.append(parsed)
        except json.JSONDecodeError:
            pass

    return events
