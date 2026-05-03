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
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import LoggingPlugin

from agents.orchestrator import (
    create_discovery_workflow,
    create_composition_workflow,
    create_expansion_workflow,
    create_local_atmosphere_workflow,
)
from google.adk.tools import google_search
from common.logging import get_logger
from models.book import BookMetadata
from plugins.langfuse_plugin import LangfusePlugin
from services.session_service import create_session_service

from .events import (
    DomainEvent,
    Phase,
    ProgressEvent,
    JobStarted,
    MetadataReady,
    RegionsReady,
    ItineraryReady,
    ExpansionReady,
    WorkflowError,
    WorkflowComplete,
)
from .extraction import extract_itinerary_from_response, extract_expansion_from_state
from .prompts import (
    build_discovery_prompt,
    build_composition_prompt,
    build_local_atmosphere_prompt,
)
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

# Agent name -> human-readable progress descriptions for the local-atmosphere flow
LOCAL_ATMOSPHERE_AGENT_STEPS: dict[str, str] = {
    "book_context_researcher": "Researching book mood and themes",
    "book_context_formatter": "Capturing book atmosphere",
    "book_context_pipeline": "Analyzing book atmosphere",
    "reader_profile_agent": "Reading user preferences",
    "local_atmosphere_researcher": "Finding atmospheric places near you",
    "local_atmosphere_formatter": "Composing your local outing",
    "local_atmosphere_pipeline": "Building local-atmosphere itinerary",
}

EXPANSION_AGENT_STEPS: dict[str, str] = {
    "expansion_researcher": "Searching for new places",
    "expansion_formatter": "Curating your additions",
    "expansion_pipeline": "Finding new places to add",
}

SOFT_CHIP_CAP = 6
HARD_EXPANSION_CAP = 20

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
        author: str,
        preferences: Optional[dict] = None,
        user_id: str = "default",
    ) -> AsyncGenerator[DomainEvent, None]:
        """Run phases 1-2: confirm book metadata + location discovery.

        Yields domain events as work progresses. The final meaningful event
        before WorkflowComplete is always either RegionsReady or WorkflowError.

        Args:
            book_title: Title of the book (pre-confirmed by caller)
            author: Author name (pre-confirmed by caller, required)
            preferences: Optional travel preferences dict
            user_id: User identifier for session isolation
        """
        job_id = str(uuid.uuid4())

        # Build initial state
        initial_state = {
            SessionStateKeys.BOOK_TITLE: book_title,
            SessionStateKeys.AUTHOR: author,
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

        yield JobStarted(job_id=job_id)

        try:
            async with timeout(self._config.workflow_timeout):
                # Confirm book metadata from provided title/author (no API lookup needed)
                exact_title = book_title
                exact_author = author

                book_metadata = BookMetadata(
                    book_title=exact_title,
                    author=exact_author,
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
            await self._mark_session_failed(job_id, user_id)
            yield WorkflowError(
                message=f"Discovery timed out after {self._config.workflow_timeout}s",
                error_type="WorkflowTimeoutError",
                phase=Phase.DISCOVERY,
            )
            yield WorkflowComplete(job_id=job_id)
        except asyncio.CancelledError:
            logger.warning("discover_cancelled", job_id=job_id)
            await self._mark_session_failed(job_id, user_id)
            raise
        except Exception as e:
            logger.error(
                "discover_error", error=str(e), error_type=type(e).__name__
            )
            await self._mark_session_failed(job_id, user_id)
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
            await self._mark_session_failed(job_id, user_id)
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
            await self._mark_session_failed(job_id, user_id)
            yield WorkflowError(
                message=f"Invalid region_ids: {invalid_ids}. Valid IDs: {valid}",
                error_type="InvalidRegionIds",
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)
            return

        # Persist retry-clear state: unmark failure, clear stale itinerary, record selection.
        # In-place mutation of session.state does NOT survive get_session(); use append_event.
        clear_event = Event(
            invocation_id="system",
            author="system",
            actions=EventActions(
                state_delta={
                    SessionStateKeys.JOB_FAILED: False,
                    SessionStateKeys.FINAL_ITINERARY: None,
                    SessionStateKeys.COMPOSER_ENVELOPE: None,
                    SessionStateKeys.SELECTED_REGIONS: selected_regions,
                }
            ),
        )
        await self._session_service.append_event(session, clear_event)

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
                result = extract_itinerary_from_response(
                    final_response, state
                )

                if result is not None:
                    itinerary_data, suggestions = result
                    suggestions = self._stamp_suggestion_ids(suggestions)
                    await self._persist_suggestions(job_id, user_id, suggestions, itinerary_data)
                    yield ItineraryReady(itinerary=itinerary_data, suggestions=suggestions)
                else:
                    await self._mark_session_failed(job_id, user_id)
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
            await self._mark_session_failed(job_id, user_id)
            yield WorkflowError(
                message=f"Composition timed out after {self._config.workflow_timeout}s",
                error_type="WorkflowTimeoutError",
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)
        except asyncio.CancelledError:
            logger.warning("compose_cancelled", job_id=job_id)
            await self._mark_session_failed(job_id, user_id)
            raise
        except Exception as e:
            logger.error(
                "compose_error", error=str(e), error_type=type(e).__name__
            )
            await self._mark_session_failed(job_id, user_id)
            yield WorkflowError(
                message=str(e),
                error_type=type(e).__name__,
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)

    async def local_atmosphere(
        self,
        book_title: str,
        author: str,
        location_label: str,
        lat: float,
        lng: float,
        radius_km: int = 80,
        preferences: Optional[dict] = None,
        user_id: str = "default",
    ) -> AsyncGenerator[DomainEvent, None]:
        """Run the single-phase local-atmosphere workflow.

        Yields a ``MetadataReady`` event followed by progress events, an
        ``ItineraryReady`` (or ``WorkflowError``) event, and a final
        ``WorkflowComplete``.

        Args:
            book_title: Exact book title.
            author: Exact author name.
            location_label: Human-readable user location (e.g. "New York, NY").
            lat: Latitude (-90..90).
            lng: Longitude (-180..180).
            radius_km: Travel radius in km (defaults to 80, ≈ 1 hour driving).
            preferences: Optional travel preferences dict.
            user_id: User identifier for session isolation.
        """
        job_id = str(uuid.uuid4())

        initial_state = {
            SessionStateKeys.BOOK_TITLE: book_title,
            SessionStateKeys.AUTHOR: author,
            SessionStateKeys.USER_LOCATION: {
                "label": location_label,
                "lat": lat,
                "lng": lng,
                "radius_km": radius_km,
            },
        }
        if preferences:
            initial_state[SessionStateKeys.USER_PREFERENCES] = preferences

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

        yield JobStarted(job_id=job_id)

        try:
            async with timeout(self._config.workflow_timeout):
                book_metadata = BookMetadata(
                    book_title=book_title,
                    author=author,
                    book_found=True,
                )
                yield MetadataReady(metadata=book_metadata.model_dump())

                session = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=job_id
                )
                state = SessionStateAccessor(session.state)
                state.book_metadata = book_metadata.model_dump()

                yield ProgressEvent(
                    phase=Phase.COMPOSITION,
                    step="Building local-atmosphere itinerary",
                    detail=f"within ~{radius_km} km of {location_label}",
                )

                langfuse_plugin = self._create_langfuse_plugin()
                workflow = create_local_atmosphere_workflow(
                    self._model,
                    book_title=book_title,
                    author=author,
                    location_label=location_label,
                    radius_km=radius_km,
                )
                runner = Runner(
                    agent=workflow,
                    app_name=APP_NAME,
                    session_service=self._session_service,
                    plugins=[LoggingPlugin(), langfuse_plugin],
                )

                prompt = build_local_atmosphere_prompt(
                    book_title, author, location_label, radius_km
                )
                message = types.Content(
                    role="user", parts=[types.Part(text=prompt)]
                )

                final_response = None
                reported_agents = set()
                async with runner:
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=job_id,
                        new_message=message,
                    ):
                        if (
                            event.author
                            and event.author in LOCAL_ATMOSPHERE_AGENT_STEPS
                            and event.author not in reported_agents
                        ):
                            reported_agents.add(event.author)
                            yield ProgressEvent(
                                phase=Phase.COMPOSITION,
                                step=LOCAL_ATMOSPHERE_AGENT_STEPS[event.author],
                                detail=event.author,
                            )
                        if event.is_final_response():
                            final_response = event

                refreshed = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=job_id
                )
                if refreshed is not None:
                    session = refreshed

                state = SessionStateAccessor(session.state)
                result = extract_itinerary_from_response(
                    final_response, state
                )

                if result is not None:
                    itinerary_data, suggestions = result
                    suggestions = self._stamp_suggestion_ids(suggestions)
                    await self._persist_suggestions(job_id, user_id, suggestions, itinerary_data)
                    yield ItineraryReady(itinerary=itinerary_data, suggestions=suggestions)
                else:
                    await self._mark_session_failed(job_id, user_id)
                    yield WorkflowError(
                        message="Failed to extract local-atmosphere itinerary",
                        error_type="ExtractionError",
                        phase=Phase.COMPOSITION,
                    )
                    yield WorkflowComplete(job_id=job_id)
                    return

                token_usage = None
                if langfuse_plugin.enabled:
                    token_usage = langfuse_plugin.get_session_stats()
                    await langfuse_plugin.flush()

                yield WorkflowComplete(job_id=job_id, token_usage=token_usage)

        except TimeoutError:
            logger.error("local_atmosphere_timeout", job_id=job_id)
            await self._mark_session_failed(job_id, user_id)
            yield WorkflowError(
                message=(
                    f"Local-atmosphere flow timed out after "
                    f"{self._config.workflow_timeout}s"
                ),
                error_type="WorkflowTimeoutError",
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)
        except asyncio.CancelledError:
            logger.warning("local_atmosphere_cancelled", job_id=job_id)
            await self._mark_session_failed(job_id, user_id)
            raise
        except Exception as e:
            logger.error(
                "local_atmosphere_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            await self._mark_session_failed(job_id, user_id)
            yield WorkflowError(
                message=str(e),
                error_type=type(e).__name__,
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)

    def _stamp_suggestion_ids(self, suggestions: list) -> list:
        """Overwrite chip ids with fresh server-issued uuid4s."""
        stamped = []
        for chip in suggestions:
            chip = dict(chip)
            chip["id"] = str(uuid.uuid4())
            stamped.append(chip)
        return stamped

    async def _persist_suggestions(
        self,
        job_id: str,
        user_id: str,
        suggestions: list,
        itinerary_data: dict | None = None,
    ) -> None:
        """Persist suggestion chips (and optionally the resolved itinerary) to session state."""
        try:
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            if session is not None:
                delta: dict = {SessionStateKeys.LAST_SUGGESTIONS: suggestions}
                if itinerary_data is not None:
                    delta[SessionStateKeys.FINAL_ITINERARY] = itinerary_data
                event = Event(
                    invocation_id="system",
                    author="system",
                    actions=EventActions(state_delta=delta),
                )
                await self._session_service.append_event(session, event)
        except Exception:
            logger.warning("persist_suggestions_error", job_id=job_id)

    async def expand(
        self,
        job_id: str,
        action_id: str,
        action_label: str,
        action_prompt: str,
        user_id: str = "default",
    ) -> AsyncGenerator[DomainEvent, None]:
        """Expand the itinerary with new places based on a suggestion chip.

        Requires a completed job_id (compose or local-atmosphere). Validates the
        action_id against stored suggestion chips to prevent injection. Yields
        ProgressEvent → ExpansionReady → WorkflowComplete.

        Args:
            job_id: Session ID from a completed compose/local-atmosphere call.
            action_id: ID of the suggestion chip the user clicked.
            action_label: Human-readable chip label (for logging only).
            action_prompt: Expansion instruction carried by the chip.
            user_id: User identifier (must match the original session).
        """
        try:
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
        except Exception as e:
            logger.error(
                "expand_session_lookup_failed",
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
                message=f"Job {job_id} not found.",
                error_type="JobNotFound",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        state = SessionStateAccessor(session.state)

        if state.failed:
            yield WorkflowError(
                message="Job is in a failed state.",
                error_type="JobFailed",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        if state.final_itinerary is None:
            yield WorkflowError(
                message="Itinerary not yet composed. Run compose or local-atmosphere first.",
                error_type="ItineraryNotReady",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        # Hard cap: refuse if already at max expansions
        expansion_count = state.expansion_count
        if expansion_count >= HARD_EXPANSION_CAP:
            yield WorkflowError(
                message=f"Expansion limit reached ({HARD_EXPANSION_CAP} expansions per session).",
                error_type="ExpansionLimitReached",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        # Always validate action_id against stored suggestions.
        # Empty last_suggestions (capped, missing, or legacy session) is also a rejection —
        # the caller must present a server-issued chip id.
        last_suggestions = state.last_suggestions
        valid_ids = {chip.get("id") for chip in last_suggestions if chip.get("id")}
        if action_id not in valid_ids:
            yield WorkflowError(
                message="Invalid action_id. It must match a suggestion chip from the last response.",
                error_type="InvalidActionId",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        # Concurrency guard: block if another expansion is in flight
        if state.expansion_in_progress:
            yield WorkflowError(
                message="An expansion is already in progress for this session.",
                error_type="ExpansionInProgress",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        # Set the lock
        lock_event = Event(
            invocation_id="system",
            author="system",
            actions=EventActions(
                state_delta={SessionStateKeys.EXPANSION_IN_PROGRESS: True}
            ),
        )
        await self._session_service.append_event(session, lock_event)

        try:
            async with timeout(self._config.workflow_timeout):
                yield ProgressEvent(
                    phase=Phase.COMPOSITION,
                    step=f"Finding places: {action_label}",
                )

                # Build dedupe list from current itinerary + prior expansions
                itinerary = state.final_itinerary or {}
                existing_names: list[str] = []
                for city in itinerary.get("cities", []):
                    city_name = city.get("name", "")
                    for stop in city.get("stops", []):
                        stop_name = stop.get("name", "")
                        if stop_name:
                            existing_names.append(f"{stop_name} ({city_name})")

                existing_places = "\n".join(existing_names) if existing_names else "(none)"

                # Determine target city: scan action_prompt for a known city name,
                # fall back to the first city so single-city itineraries always work.
                cities = itinerary.get("cities", [])
                parent_city = cities[0].get("name", "") if cities else ""
                action_prompt_lower = action_prompt.lower()
                for city in cities:
                    city_name = city.get("name", "")
                    if city_name and city_name.lower() in action_prompt_lower:
                        parent_city = city_name
                        break

                book_title = state.book_title
                author = state.author

                langfuse_plugin = self._create_langfuse_plugin()
                expansion_wf = create_expansion_workflow(
                    self._model,
                    google_search,
                    book_title=book_title,
                    author=author,
                    parent_city=parent_city,
                    action_prompt=action_prompt,
                    existing_places=existing_places,
                )
                runner = Runner(
                    agent=expansion_wf,
                    app_name=APP_NAME,
                    session_service=self._session_service,
                    plugins=[LoggingPlugin(), langfuse_plugin],
                )

                prompt = (
                    f"Expand the itinerary for {book_title} by {author}. "
                    f"Find new places in {parent_city} matching: {action_prompt}"
                )
                message = types.Content(
                    role="user", parts=[types.Part(text=prompt)]
                )

                reported_agents: set[str] = set()
                async with runner:
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=job_id,
                        new_message=message,
                    ):
                        if (
                            event.author
                            and event.author in EXPANSION_AGENT_STEPS
                            and event.author not in reported_agents
                        ):
                            reported_agents.add(event.author)
                            yield ProgressEvent(
                                phase=Phase.COMPOSITION,
                                step=EXPANSION_AGENT_STEPS[event.author],
                                detail=event.author,
                            )

                # Re-fetch session after agent run
                refreshed = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=job_id
                )
                if refreshed is not None:
                    session = refreshed

                state = SessionStateAccessor(session.state)
                expansion_data = extract_expansion_from_state(state)

                if expansion_data is None:
                    await self._clear_expansion_lock(job_id, user_id)
                    yield WorkflowError(
                        message="Failed to extract expansion result from agent response",
                        error_type="ExtractionError",
                        phase=Phase.COMPOSITION,
                    )
                    yield WorkflowComplete(job_id=job_id)
                    return

                # Post-filter: remove duplicates by case-insensitive name match
                existing_lower = {n.split(" (")[0].lower() for n in existing_names}
                new_places = [
                    p for p in expansion_data.get("places", [])
                    if p.get("name", "").lower() not in existing_lower
                ]

                # Force source="expansion" on all new stops
                for place in new_places:
                    place["source"] = "expansion"

                # Determine actual parent city (match against itinerary cities)
                returned_city = expansion_data.get("parent_city", parent_city)
                matched_city = parent_city
                for city in cities:
                    if city.get("name", "").lower() == returned_city.lower():
                        matched_city = city["name"]
                        break

                # Merge new places into final_itinerary in session state
                updated_itinerary = dict(itinerary)
                updated_cities = []
                merged = False
                for city in updated_itinerary.get("cities", []):
                    city = dict(city)
                    if city.get("name", "").lower() == matched_city.lower():
                        city["stops"] = list(city.get("stops", [])) + new_places
                        merged = True
                    updated_cities.append(city)

                if not merged and updated_cities:
                    updated_cities[0] = dict(updated_cities[0])
                    updated_cities[0]["stops"] = (
                        list(updated_cities[0].get("stops", [])) + new_places
                    )
                    matched_city = updated_cities[0].get("name", parent_city)

                updated_itinerary["cities"] = updated_cities

                # Stamp new suggestion ids and apply soft cap
                new_suggestions = expansion_data.get("suggestions", [])
                new_expansion_count = expansion_count + 1
                if new_expansion_count >= SOFT_CHIP_CAP:
                    new_suggestions = []
                else:
                    new_suggestions = self._stamp_suggestion_ids(new_suggestions)

                # Accumulate this expansion in session state
                prior_expansions = state.expansions
                this_expansion = {
                    "parent_city": matched_city,
                    "places": new_places,
                    "action_label": action_label,
                    "action_id": action_id,
                }

                persist_event = Event(
                    invocation_id="system",
                    author="system",
                    actions=EventActions(
                        state_delta={
                            SessionStateKeys.FINAL_ITINERARY: updated_itinerary,
                            SessionStateKeys.EXPANSIONS: prior_expansions + [this_expansion],
                            SessionStateKeys.EXPANSION_COUNT: new_expansion_count,
                            SessionStateKeys.LAST_SUGGESTIONS: new_suggestions,
                            SessionStateKeys.EXPANSION_IN_PROGRESS: False,
                        }
                    ),
                )
                await self._session_service.append_event(session, persist_event)

                token_usage = None
                if langfuse_plugin.enabled:
                    token_usage = langfuse_plugin.get_session_stats()
                    await langfuse_plugin.flush()

                logger.info(
                    "expansion_complete",
                    job_id=job_id[:8],
                    matched_city=matched_city,
                    new_places=len(new_places),
                    expansion_count=new_expansion_count,
                )

                yield ExpansionReady(
                    parent_city=matched_city,
                    places=new_places,
                    suggestions=new_suggestions,
                )
                yield WorkflowComplete(job_id=job_id, token_usage=token_usage)

        except TimeoutError:
            logger.error("expand_timeout", job_id=job_id)
            await self._clear_expansion_lock(job_id, user_id)
            await self._mark_session_failed(job_id, user_id)
            yield WorkflowError(
                message=f"Expansion timed out after {self._config.workflow_timeout}s",
                error_type="WorkflowTimeoutError",
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)
        except asyncio.CancelledError:
            logger.warning("expand_cancelled", job_id=job_id)
            await self._clear_expansion_lock(job_id, user_id)
            raise
        except Exception as e:
            logger.error(
                "expand_error", error=str(e), error_type=type(e).__name__
            )
            await self._clear_expansion_lock(job_id, user_id)
            yield WorkflowError(
                message=str(e),
                error_type=type(e).__name__,
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)

    async def _clear_expansion_lock(self, job_id: str, user_id: str) -> None:
        """Clear EXPANSION_IN_PROGRESS flag after error or cancellation."""
        try:
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            if session is not None:
                event = Event(
                    invocation_id="system",
                    author="system",
                    actions=EventActions(
                        state_delta={SessionStateKeys.EXPANSION_IN_PROGRESS: False}
                    ),
                )
                await self._session_service.append_event(session, event)
        except Exception:
            logger.warning("clear_expansion_lock_error", job_id=job_id)

    async def _mark_session_failed(self, job_id: str, user_id: str) -> None:
        """Persist a failure marker to session state so /status returns FAILED."""
        try:
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            if session is not None:
                event = Event(
                    invocation_id="system",
                    author="system",
                    actions=EventActions(state_delta={SessionStateKeys.JOB_FAILED: True}),
                )
                await self._session_service.append_event(session, event)
        except Exception:
            logger.warning("mark_session_failed_error", job_id=job_id)

    async def close(self) -> None:
        """Cleanup resources."""
        pass
