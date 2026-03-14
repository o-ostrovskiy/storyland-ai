"""
Transport-agnostic domain events yielded by WorkflowExecutor.

These events represent workflow state changes without any knowledge of
HTTP, SSE, or any specific delivery mechanism. Consumers (API adapter,
CLI, backend) decide how to serialize and deliver them.
"""

from dataclasses import dataclass
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
    | MetadataReady
    | RegionsReady
    | ItineraryReady
    | WorkflowError
    | WorkflowComplete
)
