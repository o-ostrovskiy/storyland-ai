"""
Transport-agnostic domain events yielded by WorkflowExecutor.

These events represent workflow state changes without any knowledge of
HTTP, SSE, or any specific delivery mechanism. Consumers (API adapter,
CLI, backend) decide how to serialize and deliver them.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import IntEnum


class Phase(IntEnum):
    """Workflow phases."""

    BOOK_SEARCH = 1
    DISCOVERY = 2
    COMPOSITION = 3


@dataclass(frozen=True)
class ProgressEvent:
    """A step started or completed within a phase."""

    phase: Phase
    step: str
    detail: Optional[str] = None


@dataclass(frozen=True)
class JobStarted:
    """A workflow has been registered with a session and has a job_id.

    Emitted as early as possible so a client that disconnects mid-stream can
    still recover the run via /status.
    """

    job_id: str


@dataclass(frozen=True)
class MetadataReady:
    """Book metadata resolved from Google Books API."""

    metadata: dict


@dataclass(frozen=True)
class RegionsReady:
    """Discovery complete, regions available for user selection."""

    job_id: str
    regions: List[dict]
    analysis_note: str


@dataclass(frozen=True)
class ItineraryReady:
    """Composition complete, validated itinerary available."""

    itinerary: dict
    suggestions: List[dict] = field(default_factory=list)
    book_recommendation_chip: Optional[dict] = None


@dataclass(frozen=True)
class ExpansionReady:
    """Expansion complete: new places added to an existing city."""

    parent_city: str
    places: List[dict]
    suggestions: List[dict] = field(default_factory=list)
    book_recommendation_chip: Optional[dict] = None


@dataclass(frozen=True)
class BookRecommendationsReady:
    """Book recommendations ready: 5 books for the reader based on book + destinations."""

    recommendations: List[dict]
    book_recommendation_count: int


@dataclass(frozen=True)
class WorkflowError:
    """An error occurred during workflow execution."""

    message: str
    error_type: str
    phase: Optional[Phase] = None


@dataclass(frozen=True)
class WorkflowComplete:
    """Stream is done. Carries optional token usage."""

    job_id: str
    token_usage: Optional[dict] = None


# Union type for consumers
DomainEvent = (
    ProgressEvent
    | JobStarted
    | MetadataReady
    | RegionsReady
    | ItineraryReady
    | ExpansionReady
    | BookRecommendationsReady
    | WorkflowError
    | WorkflowComplete
)
