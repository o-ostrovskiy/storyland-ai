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
import hashlib
import uuid
from typing import AsyncGenerator, List, Optional

from async_timeout import timeout
from google.genai import types

from core.retry import build_retry_options
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import LoggingPlugin

from agents.orchestrator import (
    create_book_recommendation_workflow,
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
    BookRecommendationsReady,
    WorkflowError,
    WorkflowComplete,
)
from .extraction import (
    extract_itinerary_from_response,
    extract_expansion_from_state,
    extract_book_recommendations_from_state,
)
from .prompts import (
    build_discovery_prompt,
    build_composition_prompt,
    build_local_atmosphere_prompt,
)
from .cache import TTLCache
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
BOOK_RECOMMENDATION_HARD_CAP = 5

BOOK_RECOMMENDATION_AGENT_STEPS: dict[str, str] = {
    "book_recommendation_researcher": "Searching for book recommendations",
    "book_recommendation_formatter": "Curating your book picks",
    "book_recommendation_pipeline": "Finding books for you",
    "book_recommendation_workflow": "Finding books for you",
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
        # In-process result cache for the Discovery chain (always on;
        # validated in prod 2026-06-17). Replays prior validated discovery
        # results verbatim, so it cannot introduce new hallucinations.
        self._discovery_cache = TTLCache(
            ttl_seconds=config.cache_ttl_seconds,
            max_entries=config.cache_max_entries,
        )

    @property
    def session_service(self):
        """Expose session service for status queries."""
        return self._session_service

    @property
    def config(self):
        """Expose config for health checks."""
        return self._config

    def _create_model(self) -> Gemini:
        retry_config = build_retry_options(
            attempts=self._config.retry_attempts,
            exp_base=self._config.retry_exp_base,
            initial_delay=1,
            max_delay=self._config.retry_max_delay,
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

    async def _emit_cached_discovery(
        self,
        job_id: str,
        user_id: str,
        book_title: str,
        author: str,
        region_analysis: dict,
    ) -> AsyncGenerator[DomainEvent, None]:
        """Replay a cached discovery result without invoking Gemini.

        The session already exists (created by ``discover`` before the cache
        lookup). This writes the confirmed book metadata and the cached
        region analysis into session state via ``append_event`` so the
        downstream discover->compose handoff (which reads ``state.regions``)
        behaves identically to a fresh run.
        """
        book_metadata = BookMetadata(
            book_title=book_title,
            author=author,
            book_found=True,
        )
        yield MetadataReady(metadata=book_metadata.model_dump())

        session = await self._session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=job_id
        )
        cache_event = Event(
            invocation_id="system",
            author="system",
            actions=EventActions(
                state_delta={
                    SessionStateKeys.BOOK_METADATA: book_metadata.model_dump(),
                    SessionStateKeys.REGION_ANALYSIS: region_analysis,
                }
            ),
        )
        await self._session_service.append_event(session, cache_event)

        yield RegionsReady(
            job_id=job_id,
            regions=region_analysis.get("regions", []),
            analysis_note=region_analysis.get("analysis_note", ""),
        )
        yield WorkflowComplete(job_id=job_id)

    @staticmethod
    def _discovery_cache_key(
        book_title: str, author: str, preferences: Optional[dict]
    ) -> str:
        """Build a normalized, versioned cache key for a discovery request.

        Versioned prefix ('v1') lets a logic change invalidate cleanly.
        """
        norm_title = (book_title or "").strip().lower()
        norm_author = (author or "").strip().lower()
        # Stable hash of preferences regardless of key ordering.
        pref_items = sorted((preferences or {}).items(), key=lambda kv: str(kv[0]))
        pref_sig = hashlib.sha1(
            repr(pref_items).encode("utf-8")
        ).hexdigest()
        return f"discover:v1:{norm_title}|{norm_author}|{pref_sig}"

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

        # Cache fast-path: an identical book/author/preferences request reuses
        # the previously computed (already validated) regions and makes ZERO
        # Gemini calls. Only non-empty region sets are ever cached, so a hit
        # cannot introduce a new fabrication; staleness is bounded by the TTL.
        cache_key = self._discovery_cache_key(book_title, author, preferences)
        cached_region_analysis = await self._discovery_cache.get(cache_key)
        if cached_region_analysis:
            logger.info("discovery_cache_hit", job_id=job_id[:8])
            async for ev in self._emit_cached_discovery(
                job_id=job_id,
                user_id=user_id,
                book_title=book_title,
                author=author,
                region_analysis=cached_region_analysis,
            ):
                yield ev
            return

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

                # Store on miss: cache only non-empty, schema-validated region
                # sets so a future hit can safely short-circuit the chain.
                if state.regions:
                    await self._discovery_cache.set(cache_key, state.region_analysis)

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
                    book_recommendation_chip = self._build_book_recommendation_chip()
                    await self._persist_suggestions(
                        job_id, user_id, suggestions, itinerary_data,
                        book_recommendation_chip=book_recommendation_chip,
                    )
                    yield ItineraryReady(
                        itinerary=itinerary_data,
                        suggestions=suggestions,
                        book_recommendation_chip=book_recommendation_chip,
                    )
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
                    book_recommendation_chip = self._build_book_recommendation_chip()
                    await self._persist_suggestions(
                        job_id, user_id, suggestions, itinerary_data,
                        book_recommendation_chip=book_recommendation_chip,
                    )
                    yield ItineraryReady(
                        itinerary=itinerary_data,
                        suggestions=suggestions,
                        book_recommendation_chip=book_recommendation_chip,
                    )
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

    def _build_book_recommendation_chip(self) -> dict:
        """Build the deterministic 'Find books like this' chip.

        The chip is kept separate from `last_suggestions` so the expand
        endpoint never accepts its id and the FE can route on it directly.
        """
        return {
            "id": str(uuid.uuid4()),
            "label": "Find books like this",
            "action_prompt": "",
        }

    async def _persist_suggestions(
        self,
        job_id: str,
        user_id: str,
        suggestions: list,
        itinerary_data: dict | None = None,
        book_recommendation_chip: dict | None = None,
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
                if book_recommendation_chip is not None:
                    delta[SessionStateKeys.BOOK_RECOMMENDATION_CHIP] = book_recommendation_chip
                    delta[SessionStateKeys.BOOK_RECOMMENDATION_CHIP_ID] = book_recommendation_chip["id"]
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
                    book_recommendation_chip=state.book_recommendation_chip,
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

    async def recommend_books(
        self,
        job_id: str,
        action_id: str,
        action_label: str,
        action_prompt: str,
        user_id: str = "default",
    ) -> AsyncGenerator[DomainEvent, None]:
        """Recommend books based on the current book + destination itinerary.

        Requires a completed job_id (compose or local-atmosphere). Validates
        action_id against the stored book-recommendation chip id. Yields
        ProgressEvent → BookRecommendationsReady → WorkflowComplete.

        Args:
            job_id: Session ID from a completed compose/local-atmosphere call.
            action_id: ID of the 'Find books like this' chip.
            action_label: Human-readable chip label (for logging only).
            action_prompt: Unused for book recs; kept for interface symmetry with expand.
            user_id: User identifier (must match the original session).
        """
        try:
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
        except Exception as e:
            logger.error(
                "recommend_books_session_lookup_failed",
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

        # Hard cap
        book_recommendation_count = state.book_recommendation_count
        if book_recommendation_count >= BOOK_RECOMMENDATION_HARD_CAP:
            yield WorkflowError(
                message=f"Book recommendation limit reached ({BOOK_RECOMMENDATION_HARD_CAP} per session).",
                error_type="BookRecommendationLimitReached",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        # Validate action_id against stored books chip id
        stored_chip_id = state.book_recommendation_chip_id
        if not stored_chip_id or action_id != stored_chip_id:
            yield WorkflowError(
                message="Invalid action_id. Use the 'Find books like this' chip id from the itinerary response.",
                error_type="InvalidActionId",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        # Concurrency guard
        if state.book_recs_in_progress:
            yield WorkflowError(
                message="A book recommendation request is already in progress for this session.",
                error_type="BookRecsInProgress",
            )
            yield WorkflowComplete(job_id=job_id)
            return

        # Set lock
        lock_event = Event(
            invocation_id="system",
            author="system",
            actions=EventActions(
                state_delta={SessionStateKeys.BOOK_RECS_IN_PROGRESS: True}
            ),
        )
        await self._session_service.append_event(session, lock_event)

        try:
            async with timeout(self._config.workflow_timeout):
                yield ProgressEvent(
                    phase=Phase.COMPOSITION,
                    step="Finding books like this",
                )

                book_title = state.book_title
                author = state.author

                # Extract destinations from itinerary cities
                itinerary = state.final_itinerary or {}
                destinations = ", ".join(
                    city.get("name", "") for city in itinerary.get("cities", [])
                    if city.get("name")
                ) or "unknown"

                # Extract themes from book context
                book_context = state.book_context or {}
                themes_list = book_context.get("themes", [])
                themes = ", ".join(themes_list) if themes_list else "literary fiction"

                langfuse_plugin = self._create_langfuse_plugin()
                book_rec_wf = create_book_recommendation_workflow(
                    self._model,
                    google_search,
                    book_title=book_title,
                    author=author,
                    destinations=destinations,
                    themes=themes,
                )
                runner = Runner(
                    agent=book_rec_wf,
                    app_name=APP_NAME,
                    session_service=self._session_service,
                    plugins=[LoggingPlugin(), langfuse_plugin],
                )

                prompt = (
                    f"Recommend 5 books for a reader who loved {book_title} by {author} "
                    f"and is travelling to {destinations}."
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
                            and event.author in BOOK_RECOMMENDATION_AGENT_STEPS
                            and event.author not in reported_agents
                        ):
                            reported_agents.add(event.author)
                            yield ProgressEvent(
                                phase=Phase.COMPOSITION,
                                step=BOOK_RECOMMENDATION_AGENT_STEPS[event.author],
                                detail=event.author,
                            )

                # Re-fetch session after agent run
                refreshed = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=job_id
                )
                if refreshed is not None:
                    session = refreshed

                state = SessionStateAccessor(session.state)
                rec_data = extract_book_recommendations_from_state(state)

                if rec_data is None:
                    await self._clear_book_recs_lock(job_id, user_id)
                    yield WorkflowError(
                        message="Failed to extract book recommendations from agent response",
                        error_type="ExtractionError",
                        phase=Phase.COMPOSITION,
                    )
                    yield WorkflowComplete(job_id=job_id)
                    return

                recommendations = rec_data.get("recommendations", [])
                # LLM sometimes hallucinates Amazon image IDs. Strip them so BookCard
                # renders gracefully without a broken img tag; leave other valid URLs.
                for rec in recommendations:
                    url: str = rec.get("image_url") or ""
                    if url and "amazon" in url.lower():
                        rec["image_url"] = None
                new_count = book_recommendation_count + 1

                persist_event = Event(
                    invocation_id="system",
                    author="system",
                    actions=EventActions(
                        state_delta={
                            SessionStateKeys.LAST_BOOK_RECOMMENDATIONS: rec_data,
                            SessionStateKeys.BOOK_RECOMMENDATION_COUNT: new_count,
                            SessionStateKeys.BOOK_RECS_IN_PROGRESS: False,
                        }
                    ),
                )
                await self._session_service.append_event(session, persist_event)

                token_usage = None
                if langfuse_plugin.enabled:
                    token_usage = langfuse_plugin.get_session_stats()
                    await langfuse_plugin.flush()

                logger.info(
                    "book_recommendations_complete",
                    job_id=job_id[:8],
                    count=len(recommendations),
                    book_recommendation_count=new_count,
                )

                yield BookRecommendationsReady(
                    recommendations=recommendations,
                    book_recommendation_count=new_count,
                )
                yield WorkflowComplete(job_id=job_id, token_usage=token_usage)

        except TimeoutError:
            logger.error("recommend_books_timeout", job_id=job_id)
            await self._clear_book_recs_lock(job_id, user_id)
            await self._mark_session_failed(job_id, user_id)
            yield WorkflowError(
                message=f"Book recommendation timed out after {self._config.workflow_timeout}s",
                error_type="WorkflowTimeoutError",
                phase=Phase.COMPOSITION,
            )
            yield WorkflowComplete(job_id=job_id)
        except asyncio.CancelledError:
            logger.warning("recommend_books_cancelled", job_id=job_id)
            await self._clear_book_recs_lock(job_id, user_id)
            raise
        except Exception as e:
            logger.error(
                "recommend_books_error", error=str(e), error_type=type(e).__name__
            )
            await self._clear_book_recs_lock(job_id, user_id)
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

    async def _clear_book_recs_lock(self, job_id: str, user_id: str) -> None:
        """Clear BOOK_RECS_IN_PROGRESS flag after error or cancellation."""
        try:
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            if session is not None:
                event = Event(
                    invocation_id="system",
                    author="system",
                    actions=EventActions(
                        state_delta={SessionStateKeys.BOOK_RECS_IN_PROGRESS: False}
                    ),
                )
                await self._session_service.append_event(session, event)
        except Exception:
            logger.warning("clear_book_recs_lock_error", job_id=job_id)

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
