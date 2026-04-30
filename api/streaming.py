"""
SSE streaming adapter.

Thin adapter that maps core DomainEvents to SSE event dicts for
EventSourceResponse. All business logic lives in core.executor.
"""

from typing import AsyncGenerator, List, Optional

from api.models import (
    SSEProgressEvent,
    SSEStartedEvent,
    SSEMetadataEvent,
    SSERegionsEvent,
    SSEItineraryEvent,
    SSEErrorEvent,
    SSEDoneEvent,
)
from core.events import (
    DomainEvent,
    ProgressEvent,
    JobStarted,
    MetadataReady,
    RegionsReady,
    ItineraryReady,
    WorkflowError,
    WorkflowComplete,
)
from core.executor import WorkflowExecutor


def _sse(event_type: str, data: str) -> dict:
    """Create an SSE event dict for EventSourceResponse."""
    return {"event": event_type, "data": data}


def domain_event_to_sse(event: DomainEvent) -> dict:
    """Map a domain event to an SSE event dict."""
    match event:
        case ProgressEvent(phase=p, step=s, detail=d):
            return _sse(
                "progress",
                SSEProgressEvent(phase=int(p), step=s, detail=d).model_dump_json(),
            )
        case JobStarted(job_id=j):
            return _sse(
                "started",
                SSEStartedEvent(job_id=j).model_dump_json(),
            )
        case MetadataReady(metadata=m):
            return _sse(
                "metadata",
                SSEMetadataEvent(
                    book_title=m.get("book_title", ""),
                    author=m.get("author", ""),
                    description=m.get("description"),
                    published_date=m.get("published_date"),
                    categories=m.get("categories", []),
                    image_url=m.get("image_url"),
                ).model_dump_json(),
            )
        case RegionsReady(job_id=j, regions=r, analysis_note=n):
            return _sse(
                "regions",
                SSERegionsEvent(
                    job_id=j, regions=r, analysis_note=n
                ).model_dump_json(),
            )
        case ItineraryReady(itinerary=i):
            return _sse(
                "itinerary",
                SSEItineraryEvent(itinerary=i).model_dump_json(),
            )
        case WorkflowError(message=m, error_type=t, phase=p):
            return _sse(
                "error",
                SSEErrorEvent(
                    message=m, error_type=t, phase=int(p) if p is not None else None
                ).model_dump_json(),
            )
        case WorkflowComplete(job_id=j, token_usage=u):
            return _sse(
                "done",
                SSEDoneEvent(job_id=j, token_usage=u).model_dump_json(),
            )
        case _:
            return _sse(
                "error",
                SSEErrorEvent(
                    message="Unknown event type", error_type="InternalError"
                ).model_dump_json(),
            )


async def discover_stream(
    book_title: str,
    author: Optional[str],
    preferences: Optional[dict],
    user_id: str,
    executor: WorkflowExecutor,
) -> AsyncGenerator[dict, None]:
    """Run phases 1-2 via executor and yield SSE events."""
    async for event in executor.discover(
        book_title=book_title,
        author=author,
        preferences=preferences,
        user_id=user_id,
    ):
        yield domain_event_to_sse(event)


async def compose_stream(
    job_id: str,
    region_ids: List[int],
    user_id: str,
    executor: WorkflowExecutor,
) -> AsyncGenerator[dict, None]:
    """Run phase 3 via executor and yield SSE events."""
    async for event in executor.compose(
        job_id=job_id,
        region_ids=region_ids,
        user_id=user_id,
    ):
        yield domain_event_to_sse(event)


async def local_atmosphere_stream(
    book_title: str,
    author: str,
    location_label: str,
    lat: float,
    lng: float,
    radius_km: int,
    preferences: Optional[dict],
    user_id: str,
    executor: WorkflowExecutor,
) -> AsyncGenerator[dict, None]:
    """Run the local-atmosphere flow via executor and yield SSE events."""
    async for event in executor.local_atmosphere(
        book_title=book_title,
        author=author,
        location_label=location_label,
        lat=lat,
        lng=lng,
        radius_km=radius_km,
        preferences=preferences,
        user_id=user_id,
    ):
        yield domain_event_to_sse(event)
