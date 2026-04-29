"""
API request and response models.

Defines Pydantic models for:
- HTTP request bodies (DiscoverRequest, ComposeRequest)
- SSE event payloads (progress, metadata, regions, itinerary, error, done)
- REST response bodies (HealthResponse, JobStatusResponse)
"""

from enum import Enum
from typing import Optional, List, Literal

from pydantic import BaseModel, Field, field_validator


# --- Request Models ---


class DiscoverRequest(BaseModel):
    """Request body for POST /api/v1/itinerary/discover."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "book_title": "1984",
                    "author": "George Orwell",
                },
                {
                    "book_title": "Pride and Prejudice",
                    "author": "Jane Austen",
                    "preferences": {"budget": "luxury", "preferred_pace": "relaxed"},
                },
            ]
        }
    }

    book_title: str = Field(min_length=1, description="Title of the book")
    author: str = Field(min_length=1, description="Author name (required)")
    preferences: Optional[dict] = Field(
        default=None, description="User travel preferences"
    )

    @field_validator("book_title")
    @classmethod
    def validate_book_title(cls, value: str) -> str:
        """Require a non-empty, non-whitespace book title."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("book_title must not be empty")
        return normalized

    @field_validator("author")
    @classmethod
    def validate_author(cls, value: str) -> str:
        """Require a non-empty, non-whitespace author name."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("author must not be empty")
        return normalized


class ComposeRequest(BaseModel):
    """Request body for POST /api/v1/itinerary/{job_id}/compose."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"region_ids": [1]},
                {"region_ids": [1, 2]},
            ]
        }
    }

    region_ids: List[int] = Field(
        min_length=1,
        description="Selected region IDs from the discover response",
    )


class UserLocation(BaseModel):
    """User's current location for the local-atmosphere flow."""

    lat: float = Field(ge=-90, le=90, description="Latitude in decimal degrees")
    lng: float = Field(ge=-180, le=180, description="Longitude in decimal degrees")
    label: str = Field(
        min_length=1,
        max_length=200,
        description="Human-readable location (e.g., 'New York, NY 10013')",
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("label must not be empty")
        return normalized


class LocalAtmosphereRequest(BaseModel):
    """Request body for POST /api/v1/itinerary/local-atmosphere.

    Single-phase flow: caller supplies a book and a current location, the
    workflow returns a TripItinerary of nearby places that match the book's
    atmosphere — no region selection step.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "book_title": "Wuthering Heights",
                    "author": "Emily Brontë",
                    "user_location": {
                        "lat": 40.7128,
                        "lng": -74.0060,
                        "label": "New York, NY 10013",
                    },
                    "radius_km": 80,
                    "preferences": {
                        "budget": "moderate",
                        "preferred_pace": "relaxed",
                    },
                },
            ]
        }
    }

    book_title: str = Field(min_length=1, description="Title of the book")
    author: str = Field(min_length=1, description="Author name (required)")
    user_location: UserLocation = Field(description="User's current location")
    radius_km: int = Field(
        default=80,
        ge=10,
        le=200,
        description="Search radius in km from the user's location",
    )
    preferences: Optional[dict] = Field(
        default=None, description="User travel preferences"
    )

    @field_validator("book_title")
    @classmethod
    def validate_book_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("book_title must not be empty")
        return normalized

    @field_validator("author")
    @classmethod
    def validate_author(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("author must not be empty")
        return normalized


# --- SSE Event Models ---


class SSEProgressEvent(BaseModel):
    """Progress update during workflow execution."""

    event: Literal["progress"] = "progress"
    phase: int = Field(description="Current phase (1, 2, or 3)")
    step: str = Field(description="Current step description")
    detail: Optional[str] = Field(
        default=None, description="Additional detail"
    )


class SSEStartedEvent(BaseModel):
    """Job has been registered with a session and has a stable job_id.

    Emitted as early as possible so a client whose SSE connection drops mid-run
    can recover the job via GET /itinerary/{job_id}/status.
    """

    event: Literal["started"] = "started"
    job_id: str


class SSEMetadataEvent(BaseModel):
    """Book metadata confirmed from the upstream service."""

    event: Literal["metadata"] = "metadata"
    book_title: str
    author: str
    description: Optional[str] = None
    published_date: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    image_url: Optional[str] = None


class SSERegionsEvent(BaseModel):
    """Region analysis results for user selection."""

    event: Literal["regions"] = "regions"
    job_id: str = Field(description="Job ID for the compose endpoint")
    regions: List[dict] = Field(description="Region options from RegionAnalysis")
    analysis_note: str = Field(default="")


class SSEItineraryEvent(BaseModel):
    """Final itinerary result."""

    event: Literal["itinerary"] = "itinerary"
    itinerary: dict = Field(description="TripItinerary as dict")


class SSEErrorEvent(BaseModel):
    """Error during workflow execution."""

    event: Literal["error"] = "error"
    message: str
    error_type: str = Field(default="WorkflowError")
    phase: Optional[int] = Field(
        default=None, description="Phase where error occurred"
    )


class SSEDoneEvent(BaseModel):
    """Stream completion marker."""

    event: Literal["done"] = "done"
    job_id: str
    token_usage: Optional[dict] = Field(
        default=None, description="Token usage stats if available"
    )


# --- REST Response Models ---


class JobStatus(str, Enum):
    """Possible job states derived from session state keys."""

    PENDING = "pending"
    SEARCHING = "searching"
    DISCOVERING = "discovering"
    REGIONS_READY = "regions_ready"
    COMPOSING = "composing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatusResponse(BaseModel):
    """Response for GET /api/v1/itinerary/{job_id}/status."""

    job_id: str
    status: JobStatus
    book_title: Optional[str] = None
    author: Optional[str] = None
    has_regions: bool = False
    has_itinerary: bool = False


class HealthResponse(BaseModel):
    """Response for GET /api/v1/health."""

    status: str = "healthy"
    version: str = "0.1.0"
    model_name: str = ""
