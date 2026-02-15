"""
SSE streaming generators for the API.

Contains async generators that run the ADK workflow phases and yield
SSE-formatted events. These generators are the bridge between the
existing workflow code (main.py pattern) and the HTTP SSE endpoints.

Key pattern:
    ADK: async for event in runner.run_async() -> SSE: yield dict for EventSourceResponse

Mirrors main.py lines 286-603 (Runner + event loop).
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator, List, Optional

from async_timeout import timeout
from google.genai import types
from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import LoggingPlugin

from api.dependencies import AppState
from api.models import (
    SSEProgressEvent,
    SSEMetadataEvent,
    SSERegionsEvent,
    SSEItineraryEvent,
    SSEErrorEvent,
    SSEDoneEvent,
)
from common.logging import get_logger
from tools.google_books import search_books_with_retry
from models.book import BookMetadata
from agents.orchestrator import create_discovery_workflow, create_composition_workflow
from plugins.langfuse_plugin import LangfusePlugin

logger = get_logger("storyland.api.streaming")


async def _set_job_failed(app_state: AppState, user_id: str, job_id: str) -> None:
    """Best-effort: persist _job_status='failed' so /status reflects reality."""
    try:
        session = await app_state.session_service.get_session(
            app_name="storyland", user_id=user_id, session_id=job_id
        )
        if session is not None:
            session.state["_job_status"] = "failed"
    except Exception:
        pass  # Don't let status bookkeeping mask the real error


# Map agent names to human-readable progress descriptions
DISCOVERY_AGENT_STEPS = {
    "book_context_researcher": "Researching book setting and themes",
    "book_context_formatter": "Formatting book context",
    "book_context_pipeline": "Analyzing book context",
    "reader_profile_agent": "Reading user preferences",
    "parallel_discovery": "Running parallel location discovery",
    "city_researcher": "Finding cities related to the book",
    "city_formatter": "Formatting city results",
    "city_pipeline": "Discovering cities",
    "landmark_researcher": "Discovering landmarks and key locations",
    "landmark_formatter": "Formatting landmark results",
    "landmark_pipeline": "Discovering landmarks",
    "author_researcher": "Locating author-related sites",
    "author_formatter": "Formatting author sites",
    "author_pipeline": "Discovering author sites",
    "region_analyzer": "Analyzing geographic regions",
}


def _sse(event_type: str, data: str) -> dict:
    """Create an SSE event dict for EventSourceResponse.

    Args:
        event_type: SSE event name (e.g. "progress", "metadata")
        data: JSON string payload

    Returns:
        Dict with 'event' and 'data' keys for sse-starlette
    """
    return {"event": event_type, "data": data}


async def discover_stream(
    book_title: str,
    author: Optional[str],
    preferences: Optional[dict],
    user_id: str,
    app_state: AppState,
) -> AsyncGenerator[dict, None]:
    """
    Run phases 1-2 and yield SSE events.

    Mirrors main.py lines 339-496:
    - Phase 1: search_books_with_retry() (no LLM)
    - Phase 2: create_discovery_workflow() + runner.run_async()

    The job_id IS the session_id.

    Args:
        book_title: Book title to search
        author: Optional author name
        preferences: Optional travel preferences
        user_id: User identifier
        app_state: Shared application state

    Yields:
        SSE event dicts for EventSourceResponse
    """
    job_id = str(uuid.uuid4())
    event_count = 0

    # Build initial session state (mirrors main.py lines 301-303)
    initial_state = {
        "book_title": book_title,
        "author": author or "",
        "_job_status": "searching",
    }
    if preferences:
        initial_state["user:preferences"] = preferences

    # Create session (job_id = session_id)
    try:
        await app_state.session_service.create_session(
            app_name="storyland",
            user_id=user_id,
            session_id=job_id,
            state=initial_state,
        )
    except Exception as e:
        logger.error(
            "session_create_failed", error=str(e), error_type=type(e).__name__
        )
        yield _sse(
            "error",
            SSEErrorEvent(
                message="Failed to initialize session",
                error_type="SessionError",
            ).model_dump_json(),
        )
        yield _sse(
            "done",
            SSEDoneEvent(job_id=job_id).model_dump_json(),
        )
        return

    try:
        async with timeout(app_state.config.workflow_timeout):
            # Phase 1: Book search (mirrors main.py lines 339-408)
            yield _sse(
                "progress",
                SSEProgressEvent(
                    phase=1, step="Searching Google Books API"
                ).model_dump_json(),
            )

            try:
                books = search_books_with_retry(
                    title=book_title, author=author or None
                )
            except Exception as e:
                logger.error("book_search_failed", error=str(e))
                await _set_job_failed(app_state, user_id, job_id)
                yield _sse(
                    "error",
                    SSEErrorEvent(
                        message=f"Could not search Google Books API: {e}",
                        error_type=type(e).__name__,
                        phase=1,
                    ).model_dump_json(),
                )
                yield _sse(
                    "done",
                    SSEDoneEvent(job_id=job_id).model_dump_json(),
                )
                return

            if not books:
                await _set_job_failed(app_state, user_id, job_id)
                yield _sse(
                    "error",
                    SSEErrorEvent(
                        message=f'Could not find "{book_title}" in Google Books.',
                        error_type="BookNotFound",
                        phase=1,
                    ).model_dump_json(),
                )
                yield _sse(
                    "done",
                    SSEDoneEvent(job_id=job_id).model_dump_json(),
                )
                return

            # Auto-select first book (no interactive selection in API)
            selected = books[0]
            exact_author = (
                ", ".join(selected.authors) if selected.authors else "Unknown"
            )
            exact_title = selected.title

            book_metadata = BookMetadata(
                book_title=exact_title,
                author=exact_author,
                description=selected.description,
                published_date=selected.published_date,
                categories=selected.categories or [],
                image_url=selected.image_url,
                book_found=True,
            )

            # Emit metadata event
            yield _sse(
                "metadata",
                SSEMetadataEvent(
                    book_title=exact_title,
                    author=exact_author,
                    description=selected.description,
                    published_date=selected.published_date,
                    categories=selected.categories or [],
                    image_url=selected.image_url,
                ).model_dump_json(),
            )

            # Store in session state (mirrors main.py lines 397-401)
            session = await app_state.session_service.get_session(
                app_name="storyland", user_id=user_id, session_id=job_id
            )
            session.state["book_metadata"] = book_metadata.model_dump()
            session.state["_job_status"] = "discovering"

            # Phase 2: Discovery (mirrors main.py lines 410-493)
            yield _sse(
                "progress",
                SSEProgressEvent(
                    phase=2, step="Starting location discovery"
                ).model_dump_json(),
            )

            discovery_workflow = create_discovery_workflow(
                app_state.model, book_title=exact_title, author=exact_author
            )

            # Fresh LangfusePlugin per request to isolate token tracking
            langfuse_plugin = LangfusePlugin(
                secret_key=app_state.config.langfuse_secret_key,
                public_key=app_state.config.langfuse_public_key,
                host=app_state.config.langfuse_host,
            )

            discovery_runner = Runner(
                agent=discovery_workflow,
                app_name="storyland",
                session_service=app_state.session_service,
                plugins=[LoggingPlugin(), langfuse_plugin],
            )

            discovery_prompt = (
                f'Discover travel locations for "{exact_title}" by {exact_author}.\n\n'
                f"Find cities, landmarks, and author-related sites, "
                f"then group them into practical travel regions."
            )
            discovery_message = types.Content(
                role="user", parts=[types.Part(text=discovery_prompt)]
            )

            # Track which agents we've already reported to avoid duplicate progress events
            reported_agents = set()

            async with discovery_runner:
                async for event in discovery_runner.run_async(
                    user_id=user_id,
                    session_id=job_id,
                    new_message=discovery_message,
                ):
                    event_count += 1
                    if (
                        event.author
                        and event.author in DISCOVERY_AGENT_STEPS
                        and event.author not in reported_agents
                    ):
                        reported_agents.add(event.author)
                        yield _sse(
                            "progress",
                            SSEProgressEvent(
                                phase=2,
                                step=DISCOVERY_AGENT_STEPS[event.author],
                                detail=event.author,
                            ).model_dump_json(),
                        )

            # Get region analysis from session state (mirrors main.py lines 475-478)
            session = await app_state.session_service.get_session(
                app_name="storyland", user_id=user_id, session_id=job_id
            )
            region_analysis = session.state.get("region_analysis", {})
            session.state["_job_status"] = "regions_ready"

            logger.info(
                "regions_discovered",
                job_id=job_id[:8],
                num_regions=len(region_analysis.get("regions", [])),
            )

            # Emit regions event
            yield _sse(
                "regions",
                SSERegionsEvent(
                    job_id=job_id,
                    regions=region_analysis.get("regions", []),
                    analysis_note=region_analysis.get("analysis_note", ""),
                ).model_dump_json(),
            )

            # Token usage
            token_usage = None
            if langfuse_plugin.enabled:
                token_usage = langfuse_plugin.get_session_stats()
                await langfuse_plugin.flush()

            # Done
            yield _sse(
                "done",
                SSEDoneEvent(
                    job_id=job_id, token_usage=token_usage
                ).model_dump_json(),
            )

    except TimeoutError:
        logger.error("discover_timeout", job_id=job_id, events=event_count)
        await _set_job_failed(app_state, user_id, job_id)
        yield _sse(
            "error",
            SSEErrorEvent(
                message=f"Discovery timed out after {app_state.config.workflow_timeout}s",
                error_type="WorkflowTimeoutError",
                phase=2,
            ).model_dump_json(),
        )
        yield _sse(
            "done",
            SSEDoneEvent(job_id=job_id).model_dump_json(),
        )
    except asyncio.CancelledError:
        # Client disconnected — can't yield SSE (client is gone), but persist status
        logger.warning("discover_cancelled", job_id=job_id, events=event_count)
        await _set_job_failed(app_state, user_id, job_id)
        raise
    except Exception as e:
        logger.error(
            "discover_error", error=str(e), error_type=type(e).__name__
        )
        await _set_job_failed(app_state, user_id, job_id)
        yield _sse(
            "error",
            SSEErrorEvent(
                message=str(e),
                error_type=type(e).__name__,
                phase=2,
            ).model_dump_json(),
        )
        yield _sse(
            "done",
            SSEDoneEvent(job_id=job_id).model_dump_json(),
        )


async def compose_stream(
    job_id: str,
    region_ids: List[int],
    user_id: str,
    app_state: AppState,
) -> AsyncGenerator[dict, None]:
    """
    Run phase 3 for selected regions and yield SSE events.

    Mirrors main.py lines 513-603:
    - Retrieves session state from discover phase
    - Stores selected regions
    - Runs create_composition_workflow() + runner.run_async()
    - Extracts JSON from final response

    Args:
        job_id: Session ID from the discover step
        region_ids: Region IDs selected by the user
        user_id: User identifier (must match the discover request)
        app_state: Shared application state

    Yields:
        SSE event dicts for EventSourceResponse
    """
    event_count = 0

    try:
        # Retrieve session from discover phase
        session = await app_state.session_service.get_session(
            app_name="storyland",
            user_id=user_id,
            session_id=job_id,
        )
    except Exception as e:
        logger.error(
            "compose_session_lookup_failed",
            job_id=job_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        yield _sse(
            "error",
            SSEErrorEvent(
                message="Failed to retrieve session",
                error_type="SessionError",
            ).model_dump_json(),
        )
        yield _sse(
            "done",
            SSEDoneEvent(job_id=job_id).model_dump_json(),
        )
        return

    if session is None:
        yield _sse(
            "error",
            SSEErrorEvent(
                message=f"Job {job_id} not found. Run /discover first.",
                error_type="JobNotFound",
            ).model_dump_json(),
        )
        yield _sse(
            "done",
            SSEDoneEvent(job_id=job_id).model_dump_json(),
        )
        return

    # Validate session has region_analysis
    region_analysis = session.state.get("region_analysis", {})
    all_regions = region_analysis.get("regions", [])

    if not all_regions:
        yield _sse(
            "error",
            SSEErrorEvent(
                message="No regions found in session. Discovery may not have completed.",
                error_type="NoRegions",
                phase=3,
            ).model_dump_json(),
        )
        yield _sse(
            "done",
            SSEDoneEvent(job_id=job_id).model_dump_json(),
        )
        return

    # Filter regions by selected IDs (mirrors main.py lines 118-125)
    # Exclude non-int region_ids (malformed agent output) to avoid TypeError in sorted()
    valid_ids = {
        r.get("region_id") for r in all_regions
        if isinstance(r.get("region_id"), int)
    }
    invalid_ids = [rid for rid in region_ids if rid not in valid_ids]

    if invalid_ids:
        yield _sse(
            "error",
            SSEErrorEvent(
                message=f"Invalid region_ids: {invalid_ids}. Valid IDs: {sorted(valid_ids)}",
                error_type="InvalidRegionIds",
                phase=3,
            ).model_dump_json(),
        )
        yield _sse(
            "done",
            SSEDoneEvent(job_id=job_id).model_dump_json(),
        )
        return

    selected_regions = [
        r for r in all_regions if r.get("region_id") in region_ids
    ]

    # Store selected regions (mirrors main.py lines 515-518)
    session.state["selected_regions"] = selected_regions
    session.state["_job_status"] = "composing"

    exact_title = session.state.get("book_metadata", {}).get("book_title", "")
    exact_author = session.state.get("book_metadata", {}).get("author", "")

    try:
        async with timeout(app_state.config.workflow_timeout):
            region_desc = (
                f"{len(selected_regions)} region(s) selected"
            )
            yield _sse(
                "progress",
                SSEProgressEvent(
                    phase=3,
                    step="Creating personalized itinerary",
                    detail=region_desc,
                ).model_dump_json(),
            )

            # Fresh LangfusePlugin per request
            langfuse_plugin = LangfusePlugin(
                secret_key=app_state.config.langfuse_secret_key,
                public_key=app_state.config.langfuse_public_key,
                host=app_state.config.langfuse_host,
            )

            # Create composition workflow (mirrors main.py lines 543-549)
            composition_workflow = create_composition_workflow(app_state.model)
            composition_runner = Runner(
                agent=composition_workflow,
                app_name="storyland",
                session_service=app_state.session_service,
                plugins=[LoggingPlugin(), langfuse_plugin],
            )

            composition_prompt = (
                f'Create a travel itinerary for "{exact_title}" by {exact_author}.\n\n'
                f"Use ONLY the cities from the selected region(s): "
                f"{json.dumps(selected_regions)}\n\n"
                f"Create a personalized itinerary based on user preferences "
                f"and the selected region(s).\n"
                f"Include ALL cities from the selected regions in your itinerary."
            )
            composition_message = types.Content(
                role="user",
                parts=[types.Part(text=composition_prompt)],
            )

            final_response = None
            async with composition_runner:
                async for event in composition_runner.run_async(
                    user_id=user_id,
                    session_id=job_id,
                    new_message=composition_message,
                ):
                    event_count += 1
                    if event.is_final_response():
                        final_response = event

            # Extract itinerary: prefer structured output_key, fall back to text parsing
            refreshed = await app_state.session_service.get_session(
                app_name="storyland", user_id=user_id, session_id=job_id
            )
            if refreshed is not None:
                session = refreshed
            result_data = None

            # Primary: read from session state (set by output_key="final_itinerary")
            state_itinerary = session.state.get("final_itinerary") if session else None
            if isinstance(state_itinerary, dict):
                result_data = state_itinerary
                logger.info("itinerary_from_state", job_id=job_id[:8])
            elif isinstance(state_itinerary, str):
                try:
                    result_data = json.loads(state_itinerary)
                    logger.info("itinerary_from_state_str", job_id=job_id[:8])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Fallback: parse from final response text
            if result_data is None and (
                final_response
                and final_response.content
                and final_response.content.parts
            ):
                for part in final_response.content.parts:
                    if hasattr(part, "text") and part.text:
                        json_start = part.text.find("{")
                        json_end = part.text.rfind("}") + 1
                        if json_start >= 0 and json_end > json_start:
                            try:
                                result_data = json.loads(
                                    part.text[json_start:json_end]
                                )
                                logger.info("itinerary_from_text_fallback", job_id=job_id[:8])
                                break
                            except json.JSONDecodeError as e:
                                logger.error(
                                    "json_parse_error", error=str(e)
                                )

            if result_data:
                session.state["_job_status"] = "completed"

                yield _sse(
                    "itinerary",
                    SSEItineraryEvent(
                        itinerary=result_data
                    ).model_dump_json(),
                )
            else:
                session.state["_job_status"] = "failed"
                yield _sse(
                    "error",
                    SSEErrorEvent(
                        message="Failed to extract itinerary from agent response",
                        error_type="ExtractionError",
                        phase=3,
                    ).model_dump_json(),
                )
                yield _sse(
                    "done",
                    SSEDoneEvent(job_id=job_id).model_dump_json(),
                )
                return

            # Token usage
            token_usage = None
            if langfuse_plugin.enabled:
                token_usage = langfuse_plugin.get_session_stats()
                await langfuse_plugin.flush()

            yield _sse(
                "done",
                SSEDoneEvent(
                    job_id=job_id, token_usage=token_usage
                ).model_dump_json(),
            )

    except TimeoutError:
        logger.error("compose_timeout", job_id=job_id, events=event_count)
        session.state["_job_status"] = "failed"
        yield _sse(
            "error",
            SSEErrorEvent(
                message=f"Composition timed out after {app_state.config.workflow_timeout}s",
                error_type="WorkflowTimeoutError",
                phase=3,
            ).model_dump_json(),
        )
        yield _sse(
            "done",
            SSEDoneEvent(job_id=job_id).model_dump_json(),
        )
    except asyncio.CancelledError:
        # Client disconnected — persist status, re-raise for asyncio
        logger.warning("compose_cancelled", job_id=job_id, events=event_count)
        session.state["_job_status"] = "failed"
        raise
    except Exception as e:
        logger.error(
            "compose_error", error=str(e), error_type=type(e).__name__
        )
        session.state["_job_status"] = "failed"
        yield _sse(
            "error",
            SSEErrorEvent(
                message=str(e),
                error_type=type(e).__name__,
                phase=3,
            ).model_dump_json(),
        )
        yield _sse(
            "done",
            SSEDoneEvent(job_id=job_id).model_dump_json(),
        )
