"""
WorkflowExecutor: transport-agnostic workflow orchestration.

This is the primary interface for the StoryLand AI agentic layer.
It encapsulates all business logic for running the three-phase workflow
and yields domain events that any consumer can process.

Usage (Python SDK — direct import from backend):
    from core import WorkflowExecutor, ExecutorConfig
    from core.events import MetadataReady, RegionsReady, ItineraryReady

    config = ExecutorConfig(model_name="gemini-2.0-flash", google_api_key="...")
    executor = WorkflowExecutor(config)

    async for event in executor.discover(book_title="1984"):
        match event:
            case MetadataReady(metadata=m): handle_metadata(m)
            case RegionsReady(regions=r):   show_regions(r)
            case WorkflowError(message=m):  handle_error(m)

Usage (HTTP adapter):
    The API layer wraps the same executor and maps DomainEvent -> SSE events.
"""

import asyncio
import uuid
from typing import AsyncGenerator, List, Optional

from async_timeout import timeout
from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import LoggingPlugin

from agents.orchestrator import create_discovery_workflow, create_composition_workflow
from common.logging import get_logger
from models.book import BookMetadata
from plugins.langfuse_plugin import LangfusePlugin
from services.session_service import create_session_service
from tools.google_books import search_books_with_retry

from .events import (
    DomainEvent,
    Phase,
    ProgressEvent,
    MetadataReady,
    RegionsReady,
    ItineraryReady,
    WorkflowError,
    WorkflowComplete,
)
from .extraction import extract_itinerary_from_response
from .prompts import build_discovery_prompt, build_composition_prompt
from .regions import get_valid_region_ids, validate_region_selection
from .session_state import SessionStateAccessor, SessionStateKeys
from .types import ExecutorConfig

logger = get_logger("storyland.core.executor")

# Agent name -> human-readable progress descriptions
DISCOVERY_AGENT_STEPS: dict[str, str] = {
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

APP_NAME = "storyland"


class WorkflowExecutor:
    """Orchestrates the 3-phase literary itinerary workflow.

    Yields DomainEvent instances. Has no knowledge of HTTP, SSE, or any
    transport layer. Designed to be consumed by:
    - A Python backend (direct import)
    - An HTTP adapter (maps events to SSE)
    - CLI or Streamlit (maps events to console/UI)
    """

    def __init__(
        self,
        config: ExecutorConfig,
        session_service=None,
        model: Optional[Gemini] = None,
    ):
        self._config = config
        self._session_service = session_service or create_session_service(
            connection_string=config.database_url,
            use_database=config.use_database,
        )
        self._model = model or self._create_model()

    @property
    def session_service(self):
        """Expose session service for status queries."""
        return self._session_service

    @property
    def config(self):
        """Expose config for health checks."""
        return self._config

    def _create_model(self) -> Gemini:
        retry_config = types.HttpRetryOptions(
            attempts=5,
            exp_base=7,
            initial_delay=1,
            http_status_codes=[429, 500, 503, 504],
        )
        return Gemini(
            model=self._config.model_name,
            api_key=self._config.google_api_key,
            retry_options=retry_config,
        )

    def _create_langfuse_plugin(self) -> LangfusePlugin:
        """Fresh plugin per workflow run to isolate token tracking."""
        return LangfusePlugin(
            secret_key=self._config.langfuse_secret_key,
            public_key=self._config.langfuse_public_key,
            host=self._config.langfuse_host,
        )

    async def discover(
        self,
        book_title: str,
        author: Optional[str] = None,
        preferences: Optional[dict] = None,
        user_id: str = "default",
    ) -> AsyncGenerator[DomainEvent, None]:
        """Run phases 1-2: book search + location discovery.

        Yields domain events as work progresses. The final meaningful event
        before WorkflowComplete is always either RegionsReady or WorkflowError.

        Args:
            book_title: Title of the book to search
            author: Optional author name for disambiguation
            preferences: Optional travel preferences dict
            user_id: User identifier for session isolation
        """
        job_id = str(uuid.uuid4())

        # Build initial state
        initial_state = {
            SessionStateKeys.BOOK_TITLE: book_title,
            SessionStateKeys.AUTHOR: author or "",
        }
        if preferences:
            initial_state[SessionStateKeys.USER_PREFERENCES] = preferences

        # Create session
        try:
            await self._session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=job_id,
                state=initial_state,
            )
        except Exception as e:
            logger.error(
                "session_create_failed", error=str(e), error_type=type(e).__name__
            )
            yield WorkflowError(
                message="Failed to initialize session",
                error_type="SessionError",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        try:
            async with timeout(self._config.workflow_timeout):
                # Phase 1: Book search
                yield ProgressEvent(
                    phase=Phase.BOOK_SEARCH, step="Searching Google Books API"
                )

                try:
                    books = search_books_with_retry(
                        title=book_title, author=author or None
                    )
                except Exception as e:
                    logger.error("book_search_failed", error=str(e))
                    yield WorkflowError(
                        message=f"Could not search Google Books API: {e}",
                        error_type=type(e).__name__,
                        phase=Phase.BOOK_SEARCH,
                    )
                    yield WorkflowComplete(job_id=job_id)
                    return

                if not books:
                    yield WorkflowError(
                        message=f'Could not find "{book_title}" in Google Books.',
                        error_type="BookNotFound",
                        phase=Phase.BOOK_SEARCH,
                    )
                    yield WorkflowComplete(job_id=job_id)
                    return

                # Auto-select first result
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

                yield MetadataReady(metadata=book_metadata.model_dump())

                # Store in session state
                session = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=job_id
                )
                state = SessionStateAccessor(session.state)
                state.book_metadata = book_metadata.model_dump()

                # Phase 2: Discovery
                yield ProgressEvent(
                    phase=Phase.DISCOVERY, step="Starting location discovery"
                )

                langfuse_plugin = self._create_langfuse_plugin()
                discovery_workflow = create_discovery_workflow(
                    self._model, book_title=exact_title, author=exact_author
                )
                runner = Runner(
                    agent=discovery_workflow,
                    app_name=APP_NAME,
                    session_service=self._session_service,
                    plugins=[LoggingPlugin(), langfuse_plugin],
                )

                prompt = build_discovery_prompt(exact_title, exact_author)
                message = types.Content(
                    role="user", parts=[types.Part(text=prompt)]
                )

                reported_agents = set()
                async with runner:
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=job_id,
                        new_message=message,
                    ):
                        if (
                            event.author
                            and event.author in DISCOVERY_AGENT_STEPS
                            and event.author not in reported_agents
                        ):
                            reported_agents.add(event.author)
                            yield ProgressEvent(
                                phase=Phase.DISCOVERY,
                                step=DISCOVERY_AGENT_STEPS[event.author],
                                detail=event.author,
                            )

                # Extract regions from session state
                session = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=job_id
                )
                state = SessionStateAccessor(session.state)

                logger.info(
                    "regions_discovered",
                    job_id=job_id[:8],
                    num_regions=len(state.regions),
                )

                yield RegionsReady(
                    job_id=job_id,
                    regions=state.regions,
                    analysis_note=state.analysis_note,
                )

                # Token usage
                token_usage = None
                if langfuse_plugin.enabled:
                    token_usage = langfuse_plugin.get_session_stats()
                    await langfuse_plugin.flush()

                yield WorkflowComplete(job_id=job_id, token_usage=token_usage)

        except TimeoutError:
            logger.error("discover_timeout", job_id=job_id)
            yield WorkflowError(
                message=f"Discovery timed out after {self._config.workflow_timeout}s",
                error_type="WorkflowTimeoutError",
                phase=Phase.DISCOVERY,
            )
            yield WorkflowComplete(job_id=job_id)
        except asyncio.CancelledError:
            logger.warning("discover_cancelled", job_id=job_id)
            raise
        except Exception as e:
            logger.error(
                "discover_error", error=str(e), error_type=type(e).__name__
            )
            yield WorkflowError(
                message=str(e),
                error_type=type(e).__name__,
                phase=Phase.DISCOVERY,
            )
            yield WorkflowComplete(job_id=job_id)

    async def compose(
        self,
        job_id: str,
        region_ids: List[int],
        user_id: str = "default",
    ) -> AsyncGenerator[DomainEvent, None]:
        """Run phase 3: itinerary composition for selected regions.

        Requires a job_id from a previous discover() call.

        Args:
            job_id: Session ID from the discover step
            region_ids: Region IDs selected by the user
            user_id: User identifier (must match the discover request)
        """
        # Session lookup
        try:
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
        except Exception as e:
            logger.error(
                "compose_session_lookup_failed",
                job_id=job_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            yield WorkflowError(
                message="Failed to retrieve session",
                error_type="SessionError",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        if session is None:
            yield WorkflowError(
                message=f"Job {job_id} not found. Run discover first.",
                error_type="JobNotFound",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        state = SessionStateAccessor(session.state)
        all_regions = state.regions

        if not all_regions:
            yield WorkflowError(
                message="No regions found in session. Discovery may not have completed.",
                error_type="NoRegions",
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)
            return

        # Validate regions
        selected_regions, invalid_ids = validate_region_selection(
            region_ids, all_regions
        )
        if invalid_ids:
            valid = sorted(get_valid_region_ids(all_regions))
            yield WorkflowError(
                message=f"Invalid region_ids: {invalid_ids}. Valid IDs: {valid}",
                error_type="InvalidRegionIds",
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)
            return

        state.selected_regions = selected_regions

        exact_title = state.book_title
        exact_author = state.author

        try:
            async with timeout(self._config.workflow_timeout):
                yield ProgressEvent(
                    phase=Phase.COMPOSITION,
                    step="Creating personalized itinerary",
                    detail=f"{len(selected_regions)} region(s) selected",
                )

                langfuse_plugin = self._create_langfuse_plugin()
                composition_workflow = create_composition_workflow(self._model)
                runner = Runner(
                    agent=composition_workflow,
                    app_name=APP_NAME,
                    session_service=self._session_service,
                    plugins=[LoggingPlugin(), langfuse_plugin],
                )

                prompt = build_composition_prompt(
                    exact_title, exact_author, selected_regions
                )
                message = types.Content(
                    role="user", parts=[types.Part(text=prompt)]
                )

                final_response = None
                async with runner:
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=job_id,
                        new_message=message,
                    ):
                        if event.is_final_response():
                            final_response = event

                # Re-fetch session to get output_key data
                refreshed = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=job_id
                )
                if refreshed is not None:
                    session = refreshed

                state = SessionStateAccessor(session.state)
                result_data = extract_itinerary_from_response(
                    final_response, state
                )

                if result_data:
                    yield ItineraryReady(itinerary=result_data)
                else:
                    yield WorkflowError(
                        message="Failed to extract itinerary from agent response",
                        error_type="ExtractionError",
                        phase=Phase.COMPOSITION,
                    )
                    yield WorkflowComplete(job_id=job_id)
                    return

                # Token usage
                token_usage = None
                if langfuse_plugin.enabled:
                    token_usage = langfuse_plugin.get_session_stats()
                    await langfuse_plugin.flush()

                yield WorkflowComplete(job_id=job_id, token_usage=token_usage)

        except TimeoutError:
            logger.error("compose_timeout", job_id=job_id)
            yield WorkflowError(
                message=f"Composition timed out after {self._config.workflow_timeout}s",
                error_type="WorkflowTimeoutError",
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)
        except asyncio.CancelledError:
            logger.warning("compose_cancelled", job_id=job_id)
            raise
        except Exception as e:
            logger.error(
                "compose_error", error=str(e), error_type=type(e).__name__
            )
            yield WorkflowError(
                message=str(e),
                error_type=type(e).__name__,
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)

    async def close(self) -> None:
        """Cleanup resources."""
        pass
