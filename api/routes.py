"""
API route endpoints.

Maps HTTP endpoints to streaming generators and status queries.
"""

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.dependencies import get_app_state
from api.models import (
    DiscoverRequest,
    ComposeRequest,
    HealthResponse,
    JobStatusResponse,
    JobStatus,
)
from api.streaming import discover_stream, compose_stream
from common.logging import get_logger

logger = get_logger("storyland.api.routes")

router = APIRouter(tags=["itinerary"])


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Check API health and model configuration."""
    try:
        app_state = get_app_state()
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            model_name=app_state.config.model_name,
        )
    except RuntimeError:
        return HealthResponse(status="unhealthy", version="0.1.0")


@router.post(
    "/itinerary/discover",
    summary="Discover locations for a book",
    responses={
        200: {
            "description": "SSE event stream (text/event-stream)",
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: progress\ndata: {"event":"progress","phase":1,"step":"Searching Google Books API"}\n\n'
                        'event: metadata\ndata: {"event":"metadata","book_title":"1984","author":"George Orwell"}\n\n'
                        'event: regions\ndata: {"event":"regions","job_id":"abc-123","regions":[...]}\n\n'
                        'event: done\ndata: {"event":"done","job_id":"abc-123"}\n\n'
                    )
                }
            },
        }
    },
)
async def discover(request: DiscoverRequest):
    """
    Run book search and location discovery (phases 1-2).

    Streams SSE events as the workflow progresses:
    - **progress** — Step-by-step updates during each phase
    - **metadata** — Resolved book metadata from Google Books API
    - **regions** — Discovered travel regions for user selection (includes `job_id`)
    - **error** — If something goes wrong during processing
    - **done** — Stream complete (includes `job_id` for the compose endpoint)
    """
    app_state = get_app_state()
    logger.info(
        "discover_request",
        book_title=request.book_title,
        author=request.author,
    )

    generator = discover_stream(
        book_title=request.book_title,
        author=request.author,
        preferences=request.preferences,
        user_id=request.user_id,
        app_state=app_state,
    )

    return EventSourceResponse(generator, media_type="text/event-stream")


@router.post(
    "/itinerary/{job_id}/compose",
    summary="Compose itinerary for selected regions",
    responses={
        200: {
            "description": "SSE event stream (text/event-stream)",
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: progress\ndata: {"event":"progress","phase":3,"step":"Creating personalized itinerary"}\n\n'
                        'event: itinerary\ndata: {"event":"itinerary","itinerary":{...}}\n\n'
                        'event: done\ndata: {"event":"done","job_id":"abc-123"}\n\n'
                    )
                }
            },
        }
    },
)
async def compose(job_id: str, request: ComposeRequest):
    """
    Create a personalized itinerary for selected regions (phase 3).

    Requires a `job_id` from a completed `/discover` call.

    Streams SSE events:
    - **progress** — Composition step updates
    - **itinerary** — Final travel itinerary with cities, stops, and summaries
    - **error** — If the job is not found or composition fails
    - **done** — Stream complete
    """
    app_state = get_app_state()
    logger.info(
        "compose_request", job_id=job_id, region_ids=request.region_ids
    )

    generator = compose_stream(
        job_id=job_id,
        region_ids=request.region_ids,
        user_id=request.user_id,
        app_state=app_state,
    )

    return EventSourceResponse(generator, media_type="text/event-stream")


@router.get(
    "/itinerary/{job_id}/status",
    response_model=JobStatusResponse,
    summary="Check job status",
)
async def get_status(job_id: str, user_id: str = "api_user"):
    """
    Check the current status of a discovery/composition job.

    Useful for reconnecting clients or polling after a disconnect.
    Status progresses through: `pending` -> `searching` -> `discovering` ->
    `regions_ready` -> `composing` -> `completed`.
    """
    app_state = get_app_state()

    try:
        session = await app_state.session_service.get_session(
            app_name="storyland",
            user_id=user_id,
            session_id=job_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if session is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    state = session.state

    # Derive status from _job_status key
    job_status_str = state.get("_job_status", "pending")
    try:
        status = JobStatus(job_status_str)
    except ValueError:
        status = JobStatus.PENDING

    book_metadata = state.get("book_metadata", {})

    return JobStatusResponse(
        job_id=job_id,
        status=status,
        book_title=book_metadata.get("book_title"),
        author=book_metadata.get("author"),
        has_regions=bool(state.get("region_analysis", {}).get("regions")),
        has_itinerary=status == JobStatus.COMPLETED,
    )
