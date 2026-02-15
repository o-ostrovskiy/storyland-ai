"""
API request and response models.

Defines Pydantic models for:
- HTTP request bodies (DiscoverRequest, ComposeRequest)
- SSE event payloads (progress, metadata, regions, itinerary, error, done)
- REST response bodies (HealthResponse, JobStatusResponse)
"""

from enum import Enum
from typing import Optional, List, Literal

from pydantic import BaseModel, Field


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
                    "user_id": "alice",
                },
            ]
        }
    }

    book_title: str = Field(description="Title of the book to search")
    author: Optional[str] = Field(
        default=None, description="Author name for disambiguation"
    )
    preferences: Optional[dict] = Field(
        default=None, description="User travel preferences"
    )
    user_id: str = Field(default="api_user", description="User identifier")


class ComposeRequest(BaseModel):
    """Request body for POST /api/v1/itinerary/{job_id}/compose."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"region_ids": [1]},
                {"region_ids": [1, 2], "user_id": "alice"},
            ]
        }
    }

    region_ids: List[int] = Field(
        description="Selected region IDs from the discover response"
    )
    user_id: str = Field(default="api_user", description="User identifier")


# --- SSE Event Models ---


class SSEProgressEvent(BaseModel):
    """Progress update during workflow execution."""

    event: Literal["progress"] = "progress"
    phase: int = Field(description="Current phase (1, 2, or 3)")
    step: str = Field(description="Current step description")
    detail: Optional[str] = Field(
        default=None, description="Additional detail"
    )


class SSEMetadataEvent(BaseModel):
    """Book metadata resolved from Google Books API."""

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
