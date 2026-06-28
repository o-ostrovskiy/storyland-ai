"""
API route endpoints.

Maps HTTP endpoints to streaming generators and status queries.
All business logic lives in core.executor — routes are thin wiring.
"""

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.dependencies import (
    enforce_rate_limit,
    get_app_state,
    get_gateway_user_id,
    get_place_to_book_resolver,
    limit_inflight,
    verify_gateway_secret,
)
from api.models import (
    DiscoverRequest,
    ComposeRequest,
    ExpandRequest,
    RecommendBooksRequest,
    LocalAtmosphereRequest,
    HealthResponse,
    JobStatusResponse,
    JobStatus,
    PlaceToBookRequest,
)
from api.streaming import (
    discover_stream,
    compose_stream,
    expand_stream,
    recommend_books_stream,
    local_atmosphere_stream,
)
from common.logging import get_logger
from core.place_to_book import PlaceToBookResolver
from core.session_state import SessionStateAccessor
from models.place_to_book import PlaceToBookResult

logger = get_logger("storyland.api.routes")

# Health endpoint is unauthenticated — must be reachable for readiness/liveness probes
# and for standalone testing without the gateway.
system_router = APIRouter(tags=["system"])

router = APIRouter(
    tags=["itinerary"],
    dependencies=[Depends(verify_gateway_secret), Depends(enforce_rate_limit)],
)


def _derive_job_status(state: dict) -> JobStatus:
    """Derive job status from session state (no _job_status key needed)."""
    accessor = SessionStateAccessor(state)
    if accessor.failed:
        return JobStatus.FAILED
    if accessor.final_itinerary:
        return JobStatus.COMPLETED
    if accessor.selected_regions:
        return JobStatus.COMPOSING
    if accessor.regions:
        return JobStatus.REGIONS_READY
    if accessor.book_metadata:
        return JobStatus.DISCOVERING
    return JobStatus.SEARCHING


@system_router.get("/health", response_model=HealthResponse)
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
    dependencies=[Depends(limit_inflight)],
    summary="Discover locations for a book",
    responses={
        200: {
            "description": "SSE event stream (text/event-stream)",
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: started\ndata: {"event":"started","job_id":"abc-123"}\n\n'
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
async def discover(request: DiscoverRequest, user_id: str = Depends(get_gateway_user_id)):
    """
    Run book search and location discovery (phases 1-2).

    Streams SSE events as the workflow progresses:
    - **started** — Emitted first; carries `job_id` so a client whose connection drops mid-run can recover via `GET /itinerary/{job_id}/status`
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
        vibe=request.vibe,
        user_id=user_id,
        executor=app_state.executor,
    )

    return EventSourceResponse(generator, media_type="text/event-stream")


@router.post(
    "/itinerary/{job_id}/compose",
    dependencies=[Depends(limit_inflight)],
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
async def compose(job_id: str, request: ComposeRequest, user_id: str = Depends(get_gateway_user_id)):
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
        user_id=user_id,
        executor=app_state.executor,
    )

    return EventSourceResponse(generator, media_type="text/event-stream")


@router.post(
    "/itinerary/local-atmosphere",
    dependencies=[Depends(limit_inflight)],
    summary="Build a local-atmosphere itinerary near the user",
    responses={
        200: {
            "description": "SSE event stream (text/event-stream)",
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: started\ndata: {"event":"started","job_id":"abc-123"}\n\n'
                        'event: progress\ndata: {"event":"progress","phase":3,"step":"Building local-atmosphere itinerary"}\n\n'
                        'event: metadata\ndata: {"event":"metadata","book_title":"Wuthering Heights","author":"Emily Brontë"}\n\n'
                        'event: itinerary\ndata: {"event":"itinerary","itinerary":{...}}\n\n'
                        'event: done\ndata: {"event":"done","job_id":"abc-123"}\n\n'
                    )
                }
            },
        }
    },
)
async def local_atmosphere(
    request: LocalAtmosphereRequest,
    user_id: str = Depends(get_gateway_user_id),
):
    """
    Build an itinerary near the user's current location whose mood and sensory
    character evoke the chosen book.

    Single-phase flow — no region selection. Streams SSE events:
    - **started** — Emitted first; carries `job_id` so a client whose connection drops mid-run can recover via `GET /itinerary/{job_id}/status`
    - **progress** — Step-by-step updates
    - **metadata** — Confirmed book metadata
    - **itinerary** — Final TripItinerary of nearby atmospheric stops
    - **error** — If something goes wrong during processing
    - **done** — Stream complete (includes `job_id`)
    """
    app_state = get_app_state()
    logger.info(
        "local_atmosphere_request",
        book_title=request.book_title,
        author=request.author,
        location_label=request.user_location.label,
        radius_km=request.radius_km,
    )

    generator = local_atmosphere_stream(
        book_title=request.book_title,
        author=request.author,
        location_label=request.user_location.label,
        lat=request.user_location.lat,
        lng=request.user_location.lng,
        radius_km=request.radius_km,
        preferences=request.preferences,
        user_id=user_id,
        executor=app_state.executor,
    )

    return EventSourceResponse(generator, media_type="text/event-stream")


@router.post(
    "/itinerary/{job_id}/expand",
    dependencies=[Depends(limit_inflight)],
    summary="Expand itinerary with new places",
    responses={
        200: {
            "description": "SSE event stream (text/event-stream)",
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: progress\ndata: {"event":"progress","phase":3,"step":"Finding places: Add restaurants nearby"}\n\n'
                        'event: expansion\ndata: {"event":"expansion","parent_city":"London","places":[...],"suggestions":[...]}\n\n'
                        'event: done\ndata: {"event":"done","job_id":"abc-123"}\n\n'
                    )
                }
            },
        }
    },
)
async def expand(
    job_id: str,
    request: ExpandRequest,
    user_id: str = Depends(get_gateway_user_id),
):
    """
    Expand the itinerary with 3-5 new places matching the selected suggestion chip.

    Requires a `job_id` from a completed `/compose` or `/local-atmosphere` call.

    Streams SSE events:
    - **progress** — Step updates during expansion
    - **expansion** — New places and follow-up suggestion chips
    - **error** — If the job is not found, action_id is invalid, or expansion fails
    - **done** — Stream complete
    """
    app_state = get_app_state()
    logger.info(
        "expand_request",
        job_id=job_id,
        action_id=request.action_id,
        action_label=request.action_label,
    )

    generator = expand_stream(
        job_id=job_id,
        action_id=request.action_id,
        action_label=request.action_label,
        action_prompt=request.action_prompt,
        user_id=user_id,
        executor=app_state.executor,
    )

    return EventSourceResponse(generator, media_type="text/event-stream")


@router.post(
    "/itinerary/{job_id}/recommend-books",
    dependencies=[Depends(limit_inflight)],
    summary="Get book recommendations based on book and destinations",
    responses={
        200: {
            "description": "SSE event stream (text/event-stream)",
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: progress\ndata: {"event":"progress","phase":3,"step":"Finding books for you"}\n\n'
                        'event: book_recommendations\ndata: {"event":"book_recommendations","recommendations":[...],"book_recommendation_count":1}\n\n'
                        'event: done\ndata: {"event":"done","job_id":"abc-123"}\n\n'
                    )
                }
            },
        }
    },
)
async def recommend_books(
    job_id: str,
    request: RecommendBooksRequest,
    user_id: str = Depends(get_gateway_user_id),
):
    """
    Recommend 5 books based on the current book and destination itinerary.

    Requires a `job_id` from a completed `/compose` or `/local-atmosphere` call.
    The `action_id` must match the 'Find books like this' chip id returned with the itinerary.

    Streams SSE events:
    - **progress** — Step updates during recommendation search
    - **book_recommendations** — 5 recommended books with title, author, reason, and recommendation basis
    - **error** — If the job is not found, action_id is invalid, or the limit is reached
    - **done** — Stream complete
    """
    app_state = get_app_state()
    logger.info(
        "recommend_books_request",
        job_id=job_id,
        action_id=request.action_id,
        action_label=request.action_label,
    )

    generator = recommend_books_stream(
        job_id=job_id,
        action_id=request.action_id,
        action_label=request.action_label,
        action_prompt=request.action_prompt,
        user_id=user_id,
        executor=app_state.executor,
    )

    return EventSourceResponse(generator, media_type="text/event-stream")


@router.get(
    "/itinerary/{job_id}/status",
    response_model=JobStatusResponse,
    summary="Check job status",
)
async def get_status(job_id: str, user_id: str = Depends(get_gateway_user_id)):
    """
    Check the current status of a discovery/composition job.

    Status is derived from session state with the following precedence:
    `failed` (terminal error) > `completed` > `composing` > `regions_ready` >
    `discovering` > `searching`.

    `failed` is set on any terminal error (book not found, timeout, cancellation).
    It takes priority over all other states so that a failed compose retry is
    never masked by a stale itinerary from a prior successful run.
    """
    app_state = get_app_state()

    try:
        session = await app_state.executor.session_service.get_session(
            app_name="storyland",
            user_id=user_id,
            session_id=job_id,
        )
    except Exception as e:
        logger.error(
            "status_session_error",
            job_id=job_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve job status",
        )

    if session is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    state = session.state
    status = _derive_job_status(state)
    book_metadata = state.get("book_metadata", {})

    return JobStatusResponse(
        job_id=job_id,
        status=status,
        book_title=book_metadata.get("book_title"),
        author=book_metadata.get("author"),
        has_regions=bool(state.get("region_analysis", {}).get("regions")),
        has_itinerary=status == JobStatus.COMPLETED,
    )


@router.post(
    "/place-to-book",
    dependencies=[Depends(limit_inflight)],
    response_model=PlaceToBookResult,
    tags=["place-to-book"],
    summary="Resolve a destination to grounded book candidates (reverse discovery)",
)
async def place_to_book(
    request: PlaceToBookRequest,
    resolver: PlaceToBookResolver = Depends(get_place_to_book_resolver),
) -> PlaceToBookResult:
    """
    Reverse-discovery: a free-text destination → grounded, labelled book
    candidates (books *set there* = ``literal``; books that *evoke* the place =
    ``vibe``). Returns a clean ``found=false`` envelope with an empty candidate
    list when the place can't be grounded — never a fabricated list.

    Internal endpoint (gateway secret enforced): the storyland-services gateway
    calls this, then runs the authoritative Google Books existence check and
    decorates each candidate with the user-facing grounding object. Non-streaming
    JSON — the resolver is a single bounded, in-process-cached lookup.
    """
    logger.info("place_to_book_request", place=request.place)
    return await resolver.resolve(request.place)
