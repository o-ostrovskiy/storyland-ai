"""
SSE streaming adapter.

Thin adapter that maps core DomainEvents to SSE event dicts for
EventSourceResponse. All business logic lives in core.executor.
"""

import asyncio
import time
from typing import AsyncGenerator, AsyncIterator, List, Optional

from sse_starlette.sse import AppStatus

from api.models import (
    SSEProgressEvent,
    SSEStartedEvent,
    SSEMetadataEvent,
    SSERegionsEvent,
    SSEItineraryEvent,
    SSEExpansionEvent,
    SSEBookRecommendationsEvent,
    SSEErrorEvent,
    SSEDoneEvent,
)
from common.request_source import RequestSource
from core.events import (
    DomainEvent,
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
from common.logging import get_logger
from core.executor import WorkflowExecutor
from core.run_harness import RESULT_EVENTS
from models.work import WorkRef

logger = get_logger("storyland.api.streaming")


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
                    kind=m.get("kind"),
                    creator=m.get("creator"),
                ).model_dump_json(),
            )
        case RegionsReady(job_id=j, regions=r, analysis_note=n):
            return _sse(
                "regions",
                SSERegionsEvent(
                    job_id=j, regions=r, analysis_note=n
                ).model_dump_json(),
            )
        case ItineraryReady(
            itinerary=i, suggestions=s, book_recommendation_chip=bc
        ):
            return _sse(
                "itinerary",
                SSEItineraryEvent(
                    itinerary=i,
                    suggestions=s,
                    book_recommendation_chip=bc,
                ).model_dump_json(),
            )
        case ExpansionReady(
            parent_city=c, places=p, suggestions=s, book_recommendation_chip=bc
        ):
            return _sse(
                "expansion",
                SSEExpansionEvent(
                    parent_city=c,
                    places=p,
                    suggestions=s,
                    book_recommendation_chip=bc,
                ).model_dump_json(),
            )
        case BookRecommendationsReady(recommendations=r, book_recommendation_count=n):
            return _sse(
                "book_recommendations",
                SSEBookRecommendationsEvent(
                    recommendations=r, book_recommendation_count=n
                ).model_dump_json(),
            )
        case WorkflowError(
            message=m,
            error_type=t,
            phase=p,
            reason=r,
            offending_title=ot,
            suggested_title=st,
            suggested_author=sa,
        ):
            return _sse(
                "error",
                SSEErrorEvent(
                    message=m,
                    error_type=t,
                    phase=int(p) if p is not None else None,
                    reason=r,
                    offending_title=ot,
                    suggested_title=st,
                    suggested_author=sa,
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


async def track_abandonment(
    flow: str, events: AsyncIterator[DomainEvent]
) -> AsyncGenerator[dict, None]:
    """Map domain events to SSE dicts; log once if the client leaves early.

    sse_starlette cancels the stream when the client disconnects (a reader
    closing the page, a superseded search, the MCP server closing discover
    after `regions`) or the server shuts down. That is expected traffic, not
    an error, so it is recorded as ONE info line, `sse_stream_abandoned`,
    carrying where the client left: the last SSE event it was sent, the last
    progress phase, whether the result had already gone out, and the seconds
    since the stream began. Read it in Sentry Logs
    (`message:sse_stream_abandoned`, grouped by `flow` / `last_event` /
    `result_sent`); the bridge in common.logging also counts it as
    `log.events{event=sse_stream_abandoned}`. `result_sent=false` is a
    visitor who gave up during the wait.
    """
    started = time.monotonic()
    last_event = "none"
    phase: Optional[int] = None
    result_sent = False
    events_sent = 0
    job_id: Optional[str] = None
    try:
        async for event in events:
            sse = domain_event_to_sse(event)
            yield sse
            events_sent += 1
            last_event = sse["event"]
            job_id = getattr(event, "job_id", None) or job_id
            if isinstance(event, ProgressEvent) and event.phase is not None:
                # The phase number only: step/detail text can carry a chip
                # label or a place the reader typed, which never goes to Sentry.
                phase = int(event.phase)
            if isinstance(event, RESULT_EVENTS):
                result_sent = True
    except (asyncio.CancelledError, GeneratorExit):
        logger.info(
            "sse_stream_abandoned",
            flow=flow,
            reason="server_shutdown" if AppStatus.should_exit else "client_closed",
            last_event=last_event,
            phase=phase,
            result_sent=result_sent,
            events_sent=events_sent,
            elapsed_s=round(time.monotonic() - started, 1),
            job_id=job_id[:8] if job_id else None,
        )
        raise


async def discover_stream(
    book_title: str,
    author: Optional[str],
    preferences: Optional[dict],
    user_id: str,
    executor: WorkflowExecutor,
    vibe: Optional[str] = None,
    taste_context: Optional[dict] = None,
    work: Optional[WorkRef] = None,
    request_source: Optional[RequestSource] = None,
) -> AsyncGenerator[dict, None]:
    """Run phases 1-2 via executor and yield SSE events."""
    async for sse in track_abandonment(
        "discover",
        executor.discover(
            book_title=book_title,
            author=author,
            preferences=preferences,
            vibe=vibe,
            taste_context=taste_context,
            user_id=user_id,
            work=work,
            request_source=request_source,
        ),
    ):
        yield sse


async def compose_stream(
    job_id: str,
    region_ids: List[int],
    user_id: str,
    executor: WorkflowExecutor,
    request_source: Optional[RequestSource] = None,
) -> AsyncGenerator[dict, None]:
    """Run phase 3 via executor and yield SSE events."""
    async for sse in track_abandonment(
        "compose",
        executor.compose(
            job_id=job_id,
            region_ids=region_ids,
            user_id=user_id,
            request_source=request_source,
        ),
    ):
        yield sse


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
    work: Optional[WorkRef] = None,
) -> AsyncGenerator[dict, None]:
    """Run the local-atmosphere flow via executor and yield SSE events."""
    async for sse in track_abandonment(
        "local_atmosphere",
        executor.local_atmosphere(
            book_title=book_title,
            author=author,
            location_label=location_label,
            lat=lat,
            lng=lng,
            radius_km=radius_km,
            preferences=preferences,
            user_id=user_id,
            work=work,
        ),
    ):
        yield sse


async def expand_stream(
    job_id: str,
    action_id: str,
    action_label: str,
    action_prompt: str,
    user_id: str,
    executor: WorkflowExecutor,
) -> AsyncGenerator[dict, None]:
    """Run the expansion flow via executor and yield SSE events."""
    async for sse in track_abandonment(
        "expand",
        executor.expand(
            job_id=job_id,
            action_id=action_id,
            action_label=action_label,
            action_prompt=action_prompt,
            user_id=user_id,
        ),
    ):
        yield sse


async def recommend_books_stream(
    job_id: str,
    action_id: str,
    action_label: str,
    action_prompt: str,
    user_id: str,
    executor: WorkflowExecutor,
) -> AsyncGenerator[dict, None]:
    """Run the book recommendation flow via executor and yield SSE events."""
    async for sse in track_abandonment(
        "recommend_books",
        executor.recommend_books(
            job_id=job_id,
            action_id=action_id,
            action_label=action_label,
            action_prompt=action_prompt,
            user_id=user_id,
        ),
    ):
        yield sse
