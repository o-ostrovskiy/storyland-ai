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

Structure note: the ADK-facing scaffolding (event pump, timeout/exception
boundary, token-usage close-out) lives in core/run_harness.py; each flow here
keeps only its business logic (guards, caching, merging, grounding filters)
plus a GuardSpec describing its error-cleanup policy.
"""

import asyncio
import hashlib
import re
import uuid
from collections import OrderedDict
from typing import AsyncGenerator, List, Optional

from google.genai import types

from core.retry import build_retry_options
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import LoggingPlugin

from agents.orchestrator import (
    create_book_recommendation_workflow,
    create_book_to_place_discovery_workflow,
    create_book_to_place_composition_workflow,
    create_expansion_workflow,
    create_local_atmosphere_workflow,
)
from google.adk.tools import google_search
from common.logging import get_logger
from models.book import BookMetadata
from plugins.langfuse_plugin import LangfusePlugin
from services.session_service import create_session_service

from .discovery_errors import classify_discovery_failure
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
    filter_grounded_recommendations,
    validate_trip_itinerary,
    _matches_city_as_standalone_word,
)
from .prompts import (
    build_discovery_prompt,
    build_composition_prompt,
    build_local_atmosphere_prompt,
)
from .cache import TTLCache, build_discovery_cache
from .cache_version import compute_cache_version
from .regions import (
    enrich_region_analysis,
    get_valid_region_ids,
    validate_region_selection,
)
from .run_harness import (
    GuardSpec,
    RunCapture,
    collect_token_usage,
    error_events,
    pump_events,
    run_guarded,
)
from .session_state import SessionStateAccessor, SessionStateKeys
from .types import ExecutorConfig

logger = get_logger("storyland.core.executor")


def _map_phase_exception(
    flow_name: str,
    phase: Optional[Phase],
    e: Exception,
    *,
    taste_context: Optional[dict] = None,
) -> WorkflowError:
    """Classify a phase's terminal exception into a client-safe WorkflowError.

    Shared by every ``GuardSpec.map_exception`` in this module (discover,
    compose, local_atmosphere, expand, recommend_books) so the async-TaskGroup
    boundary -- a child-task failure arrives here as an ``ExceptionGroup``
    whose ``str()`` is the opaque "unhandled errors in a TaskGroup (...)" -- is
    collapsed into the same typed, client-safe error on all five flows, not
    just discover(). ``message`` is never a raw exception / ExceptionGroup
    string; the real exception is logged server-side only (MYS-400).
    """
    compose_error = classify_discovery_failure(e, taste_context=taste_context)
    logger.error(
        f"{flow_name}_error",
        error=str(e),
        error_type=type(e).__name__,
        compose_error_kind=compose_error.kind,
    )
    return WorkflowError(
        message=compose_error.message,
        error_type="DiscoveryComposeError",
        phase=phase,
        reason=compose_error.kind,
        offending_title=compose_error.offending_title,
    )

# Agent name -> human-readable progress descriptions
DISCOVERY_AGENT_STEPS: dict[str, str] = {
    "book_context_researcher": "Researching book setting and themes",
    "book_context_formatter": "Formatting book context",
    "city_researcher": "Finding cities related to the book",
    "city_formatter": "Formatting city results",
    "landmark_researcher": "Discovering landmarks and key locations",
    "landmark_formatter": "Formatting landmark results",
    "author_researcher": "Locating author-related sites",
    "author_formatter": "Formatting author sites",
    "region_analyzer": "Analyzing geographic regions",
}

# Agent name -> human-readable progress descriptions for the local-atmosphere flow
LOCAL_ATMOSPHERE_AGENT_STEPS: dict[str, str] = {
    "book_context_researcher": "Researching book mood and themes",
    "book_context_formatter": "Capturing book atmosphere",
    "local_atmosphere_researcher": "Finding atmospheric places near you",
    "local_atmosphere_formatter": "Composing your local outing",
}

EXPANSION_AGENT_STEPS: dict[str, str] = {
    "expansion_researcher": "Searching for new places",
    "expansion_formatter": "Curating your additions",
}

SOFT_CHIP_CAP = 6
HARD_EXPANSION_CAP = 20
BOOK_RECOMMENDATION_HARD_CAP = 5

# Cap on WorkflowExecutor._session_locks (MYS-401 r4, Codex P2): generous
# relative to realistic concurrent-job volume on a single process -- this
# only bounds an otherwise-unbounded leak, it is never expected to bind in
# practice. See _evict_stale_session_locks for the eviction policy.
_SESSION_LOCK_REGISTRY_CAP = 10_000

BOOK_RECOMMENDATION_AGENT_STEPS: dict[str, str] = {
    "book_recommendation_researcher": "Searching for book recommendations",
    "book_recommendation_formatter": "Curating your book picks",
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
        # In-process result cache for the Discovery chain. Replays prior
        # validated discovery results verbatim, so it cannot introduce new
        # hallucinations. Gated by cache_enabled (default True): when disabled
        # the fast-path and the store are both skipped, so caching is off
        # deliberately (and logged at boot), never silently.
        self._cache_enabled = config.cache_enabled
        # Persistent-or-in-memory Discovery cache, chosen by config.cache_backend
        # (disk in prod: survives restart/redeploy; memory for tests/library use).
        self._discovery_cache = build_discovery_cache(config)
        # Version namespace mixed into every cache key so a model or prompt
        # change auto-invalidates all prior entries (essential now that the disk
        # backend persists them across deploys — no stale grounded recs, and no
        # post-deploy warmup needed).
        self._cache_version = compute_cache_version(config.model_name)
        # Per-session locks serializing the check-then-set concurrency guards
        # in expand()/recommend_books() (MYS-401): without this, two
        # overlapping requests for the same job_id can both read their
        # "in progress" flag as False before either's set is durable, so
        # both proceed -- double-billing Gemini and racing to persist the
        # merge. In-process asyncio.Lock is sufficient because this service
        # runs a single uvicorn worker (Dockerfile has no --workers); a
        # multi-process deployment would need a durable compare-and-set
        # instead (see MYS-168/MYS-169 for that coupling).
        #
        # An OrderedDict, not a plain defaultdict (Codex P2, MYS-401 r4):
        # every expand()/recommend_books() call -- even for a job_id that
        # never resolves to a real session -- used to create an entry that
        # then lived for the lifetime of this process-wide singleton, an
        # unbounded leak under normal session churn. _get_session_lock now
        # bounds this registry; see its docstring for the eviction policy.
        self._session_locks: "OrderedDict[str, asyncio.Lock]" = OrderedDict()

    @property
    def session_service(self):
        """Expose session service for status queries."""
        return self._session_service

    @property
    def config(self):
        """Expose config for health checks."""
        return self._config

    @property
    def model(self):
        """Expose the shared LLM model (e.g. for the place→book resolver)."""
        return self._model

    def _create_model(self) -> Gemini:
        retry_config = build_retry_options(
            attempts=self._config.retry_attempts,
            exp_base=self._config.retry_exp_base,
            initial_delay=1,
            max_delay=self._config.retry_max_delay,
        )
        # ADK 2.x removed the api_key field (a leftover api_key= kwarg is
        # silently DROPPED by pydantic, falling back to env auth) — the key
        # must go through client_kwargs, which Gemini passes to genai.Client.
        return Gemini(
            model=self._config.model_name,
            client_kwargs={"api_key": self._config.google_api_key},
            retry_options=retry_config,
        )

    def _create_langfuse_plugin(self) -> LangfusePlugin:
        """Fresh plugin per workflow run to isolate token tracking."""
        return LangfusePlugin(
            secret_key=self._config.langfuse_secret_key,
            public_key=self._config.langfuse_public_key,
            host=self._config.langfuse_host,
        )

    def _build_runner(self, workflow, langfuse_plugin=None) -> Runner:
        """Build a Runner for one workflow run.

        References the module-level ``Runner`` name so tests can keep
        monkeypatching ``core.executor.Runner`` as the single fake seam.
        """
        plugins = [LoggingPlugin()]
        if langfuse_plugin is not None:
            # Inject the trace-name identity: under Runner(node=...) ADK builds
            # InvocationContext with agent=None, so the plugin cannot recover
            # the root name from the context (see LangfusePlugin.root_name).
            langfuse_plugin.root_name = getattr(workflow, "name", None)
            plugins.append(langfuse_plugin)
        return Runner(
            node=workflow,
            app_name=APP_NAME,
            session_service=self._session_service,
            plugins=plugins,
        )

    async def _emit_cached_discovery(
        self,
        job_id: str,
        user_id: str,
        book_title: str,
        author: str,
        cached: dict,
    ) -> AsyncGenerator[DomainEvent, None]:
        """Replay a cached discovery result without invoking Gemini.

        The session already exists (created by ``discover`` before the cache
        lookup). ``cached`` is the v2 grounding bundle: region_analysis PLUS
        book_context and the three discovery payloads. ALL of it is written
        into session state via ``append_event`` so the downstream
        discover->compose handoff behaves identically to a fresh run —
        compose() reads the grounding from state (ADR #24 graph scoping), so
        a hit that replayed only regions would compose from city names alone
        on precisely the popular, repeated titles the cache serves most.
        """
        region_analysis = cached.get("region_analysis") or {}
        # A cache HIT must never replay a region with no place_key: hits land
        # hardest on popular, repeated titles — exactly the book-club case the
        # combined readaway exists for — so a keyless replay would kill the
        # feature on precisely the books it was built for. The version namespace
        # (core/cache_version.py) already makes a pre-MYS-460 entry unreachable;
        # this is the second lock on the same door, and it costs one call.
        region_analysis = enrich_region_analysis(region_analysis)

        book_metadata = BookMetadata(
            book_title=book_title,
            author=author,
            book_found=True,
        )
        yield MetadataReady(metadata=book_metadata.model_dump())

        session = await self._session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=job_id
        )
        state_delta = {
            SessionStateKeys.BOOK_METADATA: book_metadata.model_dump(),
            SessionStateKeys.REGION_ANALYSIS: region_analysis,
        }
        # Replay the grounding payloads compose() will read from state.
        for key, value in (
            (SessionStateKeys.BOOK_CONTEXT, cached.get("book_context")),
            (SessionStateKeys.CITY_DISCOVERY, cached.get("city_discovery")),
            (SessionStateKeys.LANDMARK_DISCOVERY, cached.get("landmark_discovery")),
            (SessionStateKeys.AUTHOR_SITES, cached.get("author_sites")),
        ):
            if value:
                state_delta[key] = value
        cache_event = Event(
            invocation_id="system",
            author="system",
            actions=EventActions(state_delta=state_delta),
        )
        await self._session_service.append_event(session, cache_event)

        yield RegionsReady(
            job_id=job_id,
            regions=region_analysis.get("regions", []),
            analysis_note=region_analysis.get("analysis_note", ""),
        )
        yield WorkflowComplete(job_id=job_id)

    def _versioned_key(self, base_key: str) -> str:
        """Namespace a logical cache key with the model/prompt version hash.

        Applied at the cache boundary (not inside ``_discovery_cache_key``) so
        the logical key contract is unchanged; a model/prompt change flips the
        namespace and all prior (persisted) entries stop matching.
        """
        return f"dv:{self._cache_version}:{base_key}"

    @staticmethod
    def _discovery_cache_key(
        book_title: str,
        author: str,
        preferences: Optional[dict],
        vibe: Optional[str] = None,
        taste_context: Optional[dict] = None,
    ) -> str:
        """Build a normalized, versioned cache key for a discovery request.

        Versioned prefix ('v1') lets a logic change invalidate cleanly. A
        present ``vibe`` is appended so a vibe-annotated request never reuses an
        unannotated (or differently-vibed) cached region set; an absent vibe
        leaves the key byte-identical to the pre-vibe format.
        """
        norm_title = (book_title or "").strip().lower()
        norm_author = (author or "").strip().lower()
        # Stable hash of preferences regardless of key ordering.
        pref_items = sorted((preferences or {}).items(), key=lambda kv: str(kv[0]))
        pref_sig = hashlib.sha1(
            repr(pref_items).encode("utf-8")
        ).hexdigest()
        # v2: the cached value is the full grounding bundle (region_analysis +
        # book_context + the three discovery payloads), not just region_analysis —
        # a cache HIT must compose with the same grounding as a fresh run.
        key = f"discover:v2:{norm_title}|{norm_author}|{pref_sig}"
        norm_vibe = (vibe or "").strip().lower()
        if norm_vibe:
            key += f"|vibe={norm_vibe}"
        # A present taste_context must not reuse an unannotated (or
        # differently-tasted) cached region set; a stable fingerprint over the
        # normalized titles/moods keeps an absent taste_context byte-identical.
        if taste_context:
            taste_sig = hashlib.sha1(
                repr(
                    (
                        tuple(taste_context.get("titles") or ()),
                        tuple(taste_context.get("moods") or ()),
                    )
                ).encode("utf-8")
            ).hexdigest()[:12]
            key += f"|taste={taste_sig}"
        return key

    async def discover(
        self,
        book_title: str,
        author: str,
        preferences: Optional[dict] = None,
        user_id: str = "default",
        vibe: Optional[str] = None,
        taste_context: Optional[dict] = None,
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
            for ev in error_events(
                job_id, "Failed to initialize session", "SessionError"
            ):
                yield ev
            return

        yield JobStarted(job_id=job_id)

        # Cache fast-path: an identical book/author/preferences request reuses
        # the previously computed (already validated) regions and makes ZERO
        # Gemini calls. Only non-empty region sets are ever cached, so a hit
        # cannot introduce a new fabrication; staleness is bounded by the TTL.
        cache_key = self._versioned_key(
            self._discovery_cache_key(
                book_title, author, preferences, vibe, taste_context
            )
        )
        cached_region_analysis = (
            await self._discovery_cache.get(cache_key)
            if self._cache_enabled
            else None
        )
        if cached_region_analysis:
            logger.info("discovery_cache_hit", job_id=job_id[:8])
            async for ev in self._emit_cached_discovery(
                job_id=job_id,
                user_id=user_id,
                book_title=book_title,
                author=author,
                cached=cached_region_analysis,
            ):
                yield ev
            return

        async def body() -> AsyncGenerator[DomainEvent, None]:
            # Confirm book metadata from provided title/author (no API lookup needed)
            exact_title = book_title
            exact_author = author

            book_metadata = BookMetadata(
                book_title=exact_title,
                author=exact_author,
                book_found=True,
            )

            yield MetadataReady(metadata=book_metadata.model_dump())

            # Persist book metadata via append_event: in-place mutation of
            # session.state does not survive get_session() (MYS-172) -- the
            # accessor's setters were silent no-ops here.
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            metadata_event = Event(
                invocation_id="system",
                author="system",
                actions=EventActions(
                    state_delta={SessionStateKeys.BOOK_METADATA: book_metadata.model_dump()}
                ),
            )
            await self._session_service.append_event(session, metadata_event)

            # Phase 2: Discovery
            yield ProgressEvent(
                phase=Phase.DISCOVERY, step="Starting location discovery"
            )

            langfuse_plugin = self._create_langfuse_plugin()
            # vibe/taste ride the researcher instructions now (ADR #24:
            # graph-scoped context — branch researchers never see the user
            # prompt that used to carry them).
            discovery_workflow = create_book_to_place_discovery_workflow(
                self._model,
                book_title=exact_title,
                author=exact_author,
                vibe=vibe,
                taste_context=taste_context,
            )
            runner = self._build_runner(discovery_workflow, langfuse_plugin)

            prompt = build_discovery_prompt(
                exact_title, exact_author, vibe, taste_context
            )
            message = types.Content(
                role="user", parts=[types.Part(text=prompt)]
            )

            async for ev in pump_events(
                runner,
                user_id=user_id,
                session_id=job_id,
                message=message,
                phase=Phase.DISCOVERY,
                agent_steps=DISCOVERY_AGENT_STEPS,
            ):
                yield ev

            # Extract regions from session state
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            run_state = SessionStateAccessor(session.state)

            logger.info(
                "regions_discovered",
                job_id=job_id[:8],
                num_regions=len(run_state.regions),
            )

            # Empty-discovery guard: when discovery yields zero regions
            # (obscure book, failed/empty google_search, extraction miss),
            # emit a clean NoRegions error instead of a silent, empty-but-
            # "successful" RegionsReady that dead-ends the user at the
            # activation moment and is recorded as a success in the funnel.
            # Mirrors the existing compose() NoRegions guard.
            if not run_state.regions:
                logger.info("discovery_empty_regions", job_id=job_id[:8])
                await self._mark_session_failed(job_id, user_id)
                for ev in error_events(
                    job_id,
                    (
                        "We couldn't map locations for this book. "
                        "Try another title or check the spelling."
                    ),
                    "NoRegions",
                    phase=Phase.DISCOVERY,
                ):
                    yield ev
                return

            # Mint the canonical cross-job place_key (MYS-460) before the
            # payload goes anywhere. Enrich the value we CACHE and the value
            # we EMIT from the same call, so a cached hit and a fresh run are
            # byte-identical on the wire. Session state is deliberately left
            # untouched: its setters are silent no-ops against persisted
            # state (MYS-172), and compose() keys on region_id, not place_key.
            region_analysis = enrich_region_analysis(run_state.region_analysis)
            regions = region_analysis.get("regions", [])

            # Store on miss: cache only non-empty, schema-validated region
            # sets so a future hit can safely short-circuit the chain. The
            # bundle carries the grounding payloads too: on a HIT, compose()
            # reads them from replayed session state (a hit that composed
            # from region names alone was the first PR-3 gate failure).
            if self._cache_enabled and regions:
                await self._discovery_cache.set(
                    cache_key,
                    {
                        "region_analysis": region_analysis,
                        "book_context": run_state.book_context,
                        "city_discovery": run_state.city_discovery,
                        "landmark_discovery": run_state.landmark_discovery,
                        "author_sites": run_state.author_sites,
                    },
                )

            yield RegionsReady(
                job_id=job_id,
                regions=regions,
                analysis_note=run_state.analysis_note,
            )

            token_usage = await collect_token_usage(langfuse_plugin)
            yield WorkflowComplete(job_id=job_id, token_usage=token_usage)

        def map_discover_exception(e: Exception) -> WorkflowError:
            return _map_phase_exception(
                "discover", Phase.DISCOVERY, e, taste_context=taste_context
            )

        spec = GuardSpec(
            flow_name="discover",
            job_id=job_id,
            timeout_seconds=self._config.workflow_timeout,
            timeout_message=(
                f"Discovery timed out after {self._config.workflow_timeout}s"
            ),
            phase=Phase.DISCOVERY,
            on_timeout=lambda: self._mark_session_failed(job_id, user_id),
            on_cancel=lambda: self._mark_session_failed(job_id, user_id),
            on_error=lambda: self._mark_session_failed(job_id, user_id),
            map_exception=map_discover_exception,
        )
        async for ev in run_guarded(body(), spec):
            yield ev

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
            for ev in error_events(
                job_id, "Failed to retrieve session", "SessionError"
            ):
                yield ev
            return

        if session is None:
            for ev in error_events(
                job_id,
                f"Job {job_id} not found. Run discover first.",
                "JobNotFound",
            ):
                yield ev
            return

        state = SessionStateAccessor(session.state)
        all_regions = state.regions

        if not all_regions:
            await self._mark_session_failed(job_id, user_id)
            for ev in error_events(
                job_id,
                "No regions found in session. Discovery may not have completed.",
                "NoRegions",
                phase=Phase.COMPOSITION,
            ):
                yield ev
            return

        # Validate regions
        selected_regions, invalid_ids = validate_region_selection(
            region_ids, all_regions
        )
        if invalid_ids:
            valid = sorted(get_valid_region_ids(all_regions))
            await self._mark_session_failed(job_id, user_id)
            for ev in error_events(
                job_id,
                f"Invalid region_ids: {invalid_ids}. Valid IDs: {valid}",
                "InvalidRegionIds",
                phase=Phase.COMPOSITION,
            ):
                yield ev
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

        async def body() -> AsyncGenerator[DomainEvent, None]:
            yield ProgressEvent(
                phase=Phase.COMPOSITION,
                step="Creating personalized itinerary",
                detail=f"{len(selected_regions)} region(s) selected",
            )

            langfuse_plugin = self._create_langfuse_plugin()
            composition_workflow = create_book_to_place_composition_workflow(self._model)
            runner = self._build_runner(composition_workflow, langfuse_plugin)

            # Graph-runtime history is scoped per invocation (ADR #24): the
            # composer sees none of the discovery conversation, so its
            # grounded inputs are passed explicitly from session state.
            prompt = build_composition_prompt(
                exact_title,
                exact_author,
                selected_regions,
                book_context=state.book_context,
                city_discovery=state.city_discovery,
                landmark_discovery=state.landmark_discovery,
                author_sites=state.author_sites,
                preferences=state.user_preferences,
            )
            message = types.Content(
                role="user", parts=[types.Part(text=prompt)]
            )

            capture = RunCapture()
            async for ev in pump_events(
                runner,
                user_id=user_id,
                session_id=job_id,
                message=message,
                phase=Phase.COMPOSITION,
                agent_steps={},
                capture=capture,
                track_final_response=True,
            ):
                yield ev

            # Re-fetch session to get output_key data
            refreshed = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            active_session = refreshed if refreshed is not None else session

            run_state = SessionStateAccessor(active_session.state)
            result = extract_itinerary_from_response(
                capture.final_response, run_state
            )

            if result is None:
                await self._mark_session_failed(job_id, user_id)
                for ev in error_events(
                    job_id,
                    "Failed to extract itinerary from agent response",
                    "ExtractionError",
                    phase=Phase.COMPOSITION,
                ):
                    yield ev
                return

            itinerary_data, suggestions = result
            suggestions = self._stamp_suggestion_ids(suggestions)
            book_recommendation_chip = self._build_book_recommendation_chip()
            # MYS-494 r1: use the RETURNED (clamped) list for the emitted
            # event too, so what the reader is shown matches what a later
            # expand() call resolves against -- not the pre-clamp local
            # variable, which is what this used to emit.
            suggestions = await self._persist_suggestions(
                job_id, user_id, suggestions, itinerary_data,
                book_recommendation_chip=book_recommendation_chip,
            )
            yield ItineraryReady(
                itinerary=itinerary_data,
                suggestions=suggestions,
                book_recommendation_chip=book_recommendation_chip,
            )

            token_usage = await collect_token_usage(langfuse_plugin)
            yield WorkflowComplete(job_id=job_id, token_usage=token_usage)

        spec = GuardSpec(
            flow_name="compose",
            job_id=job_id,
            timeout_seconds=self._config.workflow_timeout,
            timeout_message=(
                f"Composition timed out after {self._config.workflow_timeout}s"
            ),
            phase=Phase.COMPOSITION,
            on_timeout=lambda: self._mark_session_failed(job_id, user_id),
            on_cancel=lambda: self._mark_session_failed(job_id, user_id),
            on_error=lambda: self._mark_session_failed(job_id, user_id),
            map_exception=lambda e: _map_phase_exception(
                "compose", Phase.COMPOSITION, e
            ),
        )
        async for ev in run_guarded(body(), spec):
            yield ev

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
            for ev in error_events(
                job_id, "Failed to initialize session", "SessionError"
            ):
                yield ev
            return

        yield JobStarted(job_id=job_id)

        async def body() -> AsyncGenerator[DomainEvent, None]:
            book_metadata = BookMetadata(
                book_title=book_title,
                author=author,
                book_found=True,
            )
            yield MetadataReady(metadata=book_metadata.model_dump())

            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            # Persist via append_event: in-place mutation of session.state
            # does not survive get_session() (MYS-172).
            metadata_event = Event(
                invocation_id="system",
                author="system",
                actions=EventActions(
                    state_delta={SessionStateKeys.BOOK_METADATA: book_metadata.model_dump()}
                ),
            )
            await self._session_service.append_event(session, metadata_event)

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
                preferences=preferences,
            )
            runner = self._build_runner(workflow, langfuse_plugin)

            prompt = build_local_atmosphere_prompt(
                book_title, author, location_label, radius_km,
                preferences=preferences,
            )
            message = types.Content(
                role="user", parts=[types.Part(text=prompt)]
            )

            capture = RunCapture()
            async for ev in pump_events(
                runner,
                user_id=user_id,
                session_id=job_id,
                message=message,
                phase=Phase.COMPOSITION,
                agent_steps=LOCAL_ATMOSPHERE_AGENT_STEPS,
                capture=capture,
                track_final_response=True,
            ):
                yield ev

            refreshed = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            active_session = refreshed if refreshed is not None else session

            run_state = SessionStateAccessor(active_session.state)
            result = extract_itinerary_from_response(
                capture.final_response, run_state
            )

            if result is None:
                await self._mark_session_failed(job_id, user_id)
                for ev in error_events(
                    job_id,
                    "Failed to extract local-atmosphere itinerary",
                    "ExtractionError",
                    phase=Phase.COMPOSITION,
                ):
                    yield ev
                return

            itinerary_data, suggestions = result
            suggestions = self._stamp_suggestion_ids(suggestions)
            book_recommendation_chip = self._build_book_recommendation_chip()
            # MYS-494 r1: use the RETURNED (clamped) list for the emitted
            # event too, so what the reader is shown matches what a later
            # expand() call resolves against -- not the pre-clamp local
            # variable, which is what this used to emit.
            suggestions = await self._persist_suggestions(
                job_id, user_id, suggestions, itinerary_data,
                book_recommendation_chip=book_recommendation_chip,
            )
            yield ItineraryReady(
                itinerary=itinerary_data,
                suggestions=suggestions,
                book_recommendation_chip=book_recommendation_chip,
            )

            token_usage = await collect_token_usage(langfuse_plugin)
            yield WorkflowComplete(job_id=job_id, token_usage=token_usage)

        spec = GuardSpec(
            flow_name="local_atmosphere",
            job_id=job_id,
            timeout_seconds=self._config.workflow_timeout,
            timeout_message=(
                f"Local-atmosphere flow timed out after "
                f"{self._config.workflow_timeout}s"
            ),
            phase=Phase.COMPOSITION,
            on_timeout=lambda: self._mark_session_failed(job_id, user_id),
            on_cancel=lambda: self._mark_session_failed(job_id, user_id),
            on_error=lambda: self._mark_session_failed(job_id, user_id),
            map_exception=lambda e: _map_phase_exception(
                "local_atmosphere", Phase.COMPOSITION, e
            ),
        )
        async for ev in run_guarded(body(), spec):
            yield ev

    @staticmethod
    def _resolve_trusted_action_prompt(last_suggestions: list, action_id: str) -> str:
        """Look up the server-stored action_prompt for a validated chip id.

        MYS-167: the client-echoed `action_prompt` request field must never
        reach an agent instruction -- only the value the server itself
        generated and persisted onto the chip (at `_persist_suggestions`
        time, sourced from the composer/formatter's own SuggestionChip
        output) may. Callers must validate `action_id` against
        `last_suggestions` before calling this (expand() already does, via
        `valid_ids`); an id with no matching chip resolves to "" rather than
        raising, since the caller already owns the InvalidActionId rejection
        path and this is never reached for an unresolved id in practice.
        """
        resolved_chip = next(
            (chip for chip in last_suggestions if chip.get("id") == action_id), None
        )
        return (resolved_chip or {}).get("action_prompt", "")

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

    # MYS-494 item 1: mirrors ExpandRequest.action_prompt's bound
    # (api/models.py:249-253, min_length=1, max_length=500).
    _MAX_ACTION_PROMPT_CHARS = 500

    @classmethod
    def _clamp_action_prompt(cls, chip: dict) -> dict:
        """Truncate an overlong composer-generated action_prompt at persist
        time (MYS-494 item 1).

        SuggestionChip.action_prompt (models/itinerary.py:94-96) is a
        field the LLM must fill under a structured-output response
        schema (expansion_formatter's output_schema=ExpansionResult,
        same for the trip composer) -- adding max_length there would
        turn an overlong composer output into a live compose failure
        rather than a cosmetic truncation. The bound lives here instead,
        where it is guaranteed to actually run: ADK writes the raw dict
        into session state, so a Pydantic field_validator/computed_field
        on the response schema would never execute on this path (a
        validator that silently never runs is worse than no validator).

        Clamp, don't reject: an overlong prompt is our own composer's
        tidiness bug, not the reader's fault -- dropping the whole chip
        would degrade their UI to fix our output.
        """
        prompt = chip.get("action_prompt", "")
        if isinstance(prompt, str) and len(prompt) > cls._MAX_ACTION_PROMPT_CHARS:
            logger.warning(
                "suggestion_chip_action_prompt_overlong",
                original_length=len(prompt),
                clamped_to=cls._MAX_ACTION_PROMPT_CHARS,
            )
            chip = dict(chip)
            chip["action_prompt"] = prompt[: cls._MAX_ACTION_PROMPT_CHARS]
        return chip

    async def _persist_suggestions(
        self,
        job_id: str,
        user_id: str,
        suggestions: list,
        itinerary_data: dict | None = None,
        book_recommendation_chip: dict | None = None,
    ) -> list:
        """Persist suggestion chips (and optionally the resolved itinerary) to session state.

        Returns the clamped suggestions list actually written to session
        state (MYS-494 r1). Callers MUST emit this returned list, not the
        one they passed in -- clamping only the copy written here while
        the caller separately emits its own unclamped local variable is
        exactly the divergence this fix closes: the reader is shown (and
        can click) a chip whose action_prompt is longer than what's
        stored, and whose stored copy is what a later expand() actually
        resolves against. On the no-op path (session missing, or an
        exception during the write) the original suggestions are
        returned unclamped, matching prior no-op behaviour.
        """
        clamped = suggestions
        try:
            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            if session is not None:
                clamped = [
                    self._clamp_action_prompt(chip) for chip in suggestions
                ]
                delta: dict = {SessionStateKeys.LAST_SUGGESTIONS: clamped}
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
        return clamped

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
        action_id against stored suggestion chips, then resolves the expansion
        instruction from THAT stored chip -- never from the client-echoed
        action_prompt argument -- so a caller cannot steer the researcher's
        system instruction by sending an arbitrary string alongside a valid
        chip id (MYS-167). Yields ProgressEvent → ExpansionReady →
        WorkflowComplete.

        Args:
            job_id: Session ID from a completed compose/local-atmosphere call.
            action_id: ID of the suggestion chip the user clicked.
            action_label: Human-readable chip label (for logging only).
            action_prompt: Client-echoed chip prompt text. NOT trusted and never
                interpolated into an agent instruction -- display-only, same as
                action_label. The instruction actually used is looked up from
                the server-stored chip matching action_id. Kept as a request
                field for FE wire compatibility.
            user_id: User identifier (must match the original session).
        """
        # Concurrency guard (MYS-401 r4 -- Codex P1): EVERY admission
        # decision -- job lookup, failed/not-ready state, HARD_EXPANSION_CAP,
        # action_id validity, and the in-progress flag itself -- is made
        # from ONE session snapshot fetched UNDER the per-job_id lock. There
        # is no separate pre-lock read at all: the lock only needs job_id to
        # compute its key, so it is acquired first, and the single fetch
        # taken while holding it is what every check below runs against.
        #
        # r3 only adopted the under-lock snapshot as the MERGE base (state/
        # expansion_count/last_suggestions used inside body()) -- the
        # admission checks above it still ran against the stale pre-lock
        # capture, so a delayed second click naming a chip an intervening,
        # already-completed expansion had consumed was still admitted (and
        # still billed a second Gemini call): finding #1 of this ticket's
        # own "Why", still open one hop later. Collapsing to a single
        # fetch removes the second, stale snapshot the bug needs to exist.
        error_message: Optional[str] = None
        error_kind: Optional[str] = None
        state: Optional[SessionStateAccessor] = None
        expansion_count = 0
        last_suggestions: list = []
        trusted_action_prompt = ""
        lock_session = None

        try:
            async with self._get_session_lock(job_id):
                lock_session = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=job_id
                )
                if lock_session is None:
                    error_message = f"Job {job_id} not found."
                    error_kind = "JobNotFound"
                else:
                    state = SessionStateAccessor(lock_session.state)
                    if state.failed:
                        error_message = "Job is in a failed state."
                        error_kind = "JobFailed"
                    elif state.final_itinerary is None:
                        error_message = (
                            "Itinerary not yet composed. Run compose or "
                            "local-atmosphere first."
                        )
                        error_kind = "ItineraryNotReady"
                    else:
                        # Hard cap: refuse if already at max expansions
                        expansion_count = state.expansion_count
                        if expansion_count >= HARD_EXPANSION_CAP:
                            error_message = (
                                f"Expansion limit reached "
                                f"({HARD_EXPANSION_CAP} expansions per "
                                f"session)."
                            )
                            error_kind = "ExpansionLimitReached"
                        else:
                            # Always validate action_id against stored
                            # suggestions. Empty last_suggestions (capped,
                            # missing, or legacy session) is also a
                            # rejection -- the caller must present a
                            # server-issued chip id from THIS fresh
                            # snapshot, not one an earlier response handed
                            # out that a concurrent/prior call has since
                            # consumed.
                            last_suggestions = state.last_suggestions
                            valid_ids = {
                                chip.get("id")
                                for chip in last_suggestions
                                if chip.get("id")
                            }
                            if action_id not in valid_ids:
                                error_message = (
                                    "Invalid action_id. It must match a "
                                    "suggestion chip from the last "
                                    "response."
                                )
                                error_kind = "InvalidActionId"
                            elif state.expansion_in_progress:
                                error_message = (
                                    "An expansion is already in progress "
                                    "for this session."
                                )
                                error_kind = "ExpansionInProgress"
                            else:
                                # MYS-167: resolve the chip's OWN
                                # server-stored action_prompt rather than
                                # trusting the client-echoed
                                # `action_prompt` argument. `action_id` is
                                # already confirmed to be in `valid_ids`
                                # above, so the lookup here always
                                # resolves. From here down `action_prompt`
                                # (the parameter) is display-only, same as
                                # `action_label` -- it must never reach an
                                # agent instruction.
                                trusted_action_prompt = (
                                    self._resolve_trusted_action_prompt(
                                        last_suggestions, action_id
                                    )
                                )
                                lock_event = Event(
                                    invocation_id="system",
                                    author="system",
                                    actions=EventActions(
                                        state_delta={
                                            SessionStateKeys.EXPANSION_IN_PROGRESS: True
                                        }
                                    ),
                                )
                                await self._session_service.append_event(
                                    lock_session, lock_event
                                )
        except Exception as e:
            # A transient session-backend error anywhere in the block above
            # (the fetch or the lock-flag persist) must still emit the same
            # client-safe SessionError + WorkflowComplete pair the initial
            # lookup always has, not propagate raw and truncate the stream
            # (the same client-visible-truncation class MYS-400 closed for
            # the phase executors).
            logger.error(
                "expand_session_lookup_failed",
                job_id=job_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            error_message = "Failed to retrieve session"
            error_kind = "SessionError"

        if error_kind is not None:
            for ev in error_events(job_id, error_message, error_kind):
                yield ev
            return

        async def body() -> AsyncGenerator[DomainEvent, None]:
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

            # Determine target city: scan the TRUSTED action_prompt for a known
            # city name, fall back to the first city so single-city itineraries
            # always work. (MYS-167: was the client-echoed action_prompt.)
            cities = itinerary.get("cities", [])
            parent_city = cities[0].get("name", "") if cities else ""
            trip_city_names = [c.get("name", "") for c in cities if c.get("name")]
            for city in cities:
                city_name = city.get("name", "")
                if city_name and _matches_city_as_standalone_word(
                    city_name, trusted_action_prompt, trip_city_names
                ):
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
                action_prompt=trusted_action_prompt,
                existing_places=existing_places,
            )
            runner = self._build_runner(expansion_wf, langfuse_plugin)

            prompt = (
                f"Expand the itinerary for {book_title} by {author}. "
                f"Find new places in {parent_city} matching: {trusted_action_prompt}"
            )
            message = types.Content(
                role="user", parts=[types.Part(text=prompt)]
            )

            async for ev in pump_events(
                runner,
                user_id=user_id,
                session_id=job_id,
                message=message,
                phase=Phase.COMPOSITION,
                agent_steps=EXPANSION_AGENT_STEPS,
            ):
                yield ev

            # Re-fetch session after agent run
            refreshed = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            active_session = refreshed if refreshed is not None else lock_session

            run_state = SessionStateAccessor(active_session.state)
            expansion_data = extract_expansion_from_state(run_state)

            if expansion_data is None:
                await self._clear_expansion_lock(job_id, user_id)
                for ev in error_events(
                    job_id,
                    "Failed to extract expansion result from agent response",
                    "ExtractionError",
                    phase=Phase.COMPOSITION,
                ):
                    yield ev
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

            # Re-validate the merged itinerary before persisting (MYS-401):
            # the merge above mutates raw dicts with no schema check, so a
            # formatter quirk (a stop missing match_type, a parent_city
            # mismatch yielding empty cities) could otherwise persist a
            # structurally-degraded FINAL_ITINERARY that /status re-serves,
            # bypassing every downstream schema guard. On failure, do NOT
            # persist the degraded merge -- FINAL_ITINERARY stays whatever
            # it already was (the pre-merge itinerary), so /status keeps
            # re-serving a valid result instead of a broken one.
            validated_itinerary = validate_trip_itinerary(updated_itinerary)
            if validated_itinerary is None:
                logger.error(
                    "expansion_merge_validation_failed",
                    job_id=job_id[:8],
                    matched_city=matched_city,
                )
                await self._clear_expansion_lock(job_id, user_id)
                for ev in error_events(
                    job_id,
                    "Failed to merge new places into the itinerary.",
                    "MergeValidationError",
                    phase=Phase.COMPOSITION,
                ):
                    yield ev
                return
            updated_itinerary = validated_itinerary

            # Stamp new suggestion ids and apply soft cap
            new_suggestions = expansion_data.get("suggestions", [])
            new_expansion_count = expansion_count + 1
            if new_expansion_count >= SOFT_CHIP_CAP:
                new_suggestions = []
            else:
                new_suggestions = self._stamp_suggestion_ids(new_suggestions)
                # MYS-494 r1: expand() writes LAST_SUGGESTIONS directly via
                # persist_event below rather than calling
                # _persist_suggestions, so that helper's clamp never ran on
                # this path -- every chip an expansion produces shipped
                # unclamped. Clamp here, at the point these chips are
                # produced, so the SAME list is both what gets persisted
                # and what ExpansionReady emits below.
                new_suggestions = [
                    self._clamp_action_prompt(chip) for chip in new_suggestions
                ]

            # Accumulate this expansion in session state
            prior_expansions = run_state.expansions
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
            await self._session_service.append_event(active_session, persist_event)

            token_usage = await collect_token_usage(langfuse_plugin)

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
                book_recommendation_chip=run_state.book_recommendation_chip,
            )
            yield WorkflowComplete(job_id=job_id, token_usage=token_usage)

        async def timeout_cleanup() -> None:
            await self._clear_expansion_lock(job_id, user_id)
            await self._mark_session_failed(job_id, user_id)

        spec = GuardSpec(
            flow_name="expand",
            job_id=job_id,
            timeout_seconds=self._config.workflow_timeout,
            timeout_message=(
                f"Expansion timed out after {self._config.workflow_timeout}s"
            ),
            phase=Phase.COMPOSITION,
            on_timeout=timeout_cleanup,
            on_cancel=lambda: self._clear_expansion_lock(job_id, user_id),
            on_error=lambda: self._clear_expansion_lock(job_id, user_id),
            map_exception=lambda e: _map_phase_exception(
                "expand", Phase.COMPOSITION, e
            ),
        )
        async for ev in run_guarded(body(), spec):
            yield ev

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
        # Concurrency guard (MYS-401 r4 -- Codex P1): same structural fix as
        # expand() above -- every admission decision (job lookup, failed/
        # not-ready state, BOOK_RECOMMENDATION_HARD_CAP, action_id validity,
        # the in-progress flag) is made from ONE session snapshot fetched
        # UNDER the per-job_id lock, with no separate pre-lock read to go
        # stale. r3 only adopted the under-lock snapshot for the merge base
        # (state/book_recommendation_count used inside body()); the
        # admission checks above it still ran against the stale pre-lock
        # capture, so a delayed second click naming a chip an intervening,
        # already-completed call had consumed was still admitted and still
        # billed a second Gemini call.
        error_message: Optional[str] = None
        error_kind: Optional[str] = None
        state: Optional[SessionStateAccessor] = None
        book_recommendation_count = 0
        lock_session = None

        try:
            async with self._get_session_lock(job_id):
                lock_session = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=job_id
                )
                if lock_session is None:
                    error_message = f"Job {job_id} not found."
                    error_kind = "JobNotFound"
                else:
                    state = SessionStateAccessor(lock_session.state)
                    if state.failed:
                        error_message = "Job is in a failed state."
                        error_kind = "JobFailed"
                    elif state.final_itinerary is None:
                        error_message = (
                            "Itinerary not yet composed. Run compose or "
                            "local-atmosphere first."
                        )
                        error_kind = "ItineraryNotReady"
                    else:
                        # Hard cap
                        book_recommendation_count = state.book_recommendation_count
                        if book_recommendation_count >= BOOK_RECOMMENDATION_HARD_CAP:
                            error_message = (
                                f"Book recommendation limit reached "
                                f"({BOOK_RECOMMENDATION_HARD_CAP} per "
                                f"session)."
                            )
                            error_kind = "BookRecommendationLimitReached"
                        else:
                            # Validate action_id against stored books chip
                            # id from THIS fresh snapshot, not one an
                            # earlier response handed out that a
                            # concurrent/prior call has since consumed.
                            stored_chip_id = state.book_recommendation_chip_id
                            if not stored_chip_id or action_id != stored_chip_id:
                                error_message = (
                                    "Invalid action_id. Use the 'Find "
                                    "books like this' chip id from the "
                                    "itinerary response."
                                )
                                error_kind = "InvalidActionId"
                            elif state.book_recs_in_progress:
                                error_message = (
                                    "A book recommendation request is "
                                    "already in progress for this "
                                    "session."
                                )
                                error_kind = "BookRecsInProgress"
                            else:
                                lock_event = Event(
                                    invocation_id="system",
                                    author="system",
                                    actions=EventActions(
                                        state_delta={
                                            SessionStateKeys.BOOK_RECS_IN_PROGRESS: True
                                        }
                                    ),
                                )
                                await self._session_service.append_event(
                                    lock_session, lock_event
                                )
        except Exception as e:
            # A transient session-backend error anywhere in the block above
            # must still emit the same client-safe SessionError +
            # WorkflowComplete pair the initial lookup always has, not
            # propagate raw and truncate the stream.
            logger.error(
                "recommend_books_session_lookup_failed",
                job_id=job_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            error_message = "Failed to retrieve session"
            error_kind = "SessionError"

        if error_kind is not None:
            for ev in error_events(job_id, error_message, error_kind):
                yield ev
            return

        async def body() -> AsyncGenerator[DomainEvent, None]:
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
            runner = self._build_runner(book_rec_wf, langfuse_plugin)

            prompt = (
                f"Recommend 5 books for a reader who loved {book_title} by {author} "
                f"and is travelling to {destinations}."
            )
            message = types.Content(
                role="user", parts=[types.Part(text=prompt)]
            )

            # Capture the grounded researcher output so we can post-validate
            # the formatter's titles against it.
            capture = RunCapture()
            async for ev in pump_events(
                runner,
                user_id=user_id,
                session_id=job_id,
                message=message,
                phase=Phase.COMPOSITION,
                agent_steps=BOOK_RECOMMENDATION_AGENT_STEPS,
                capture=capture,
                capture_authors=("book_recommendation_researcher",),
            ):
                yield ev

            # Re-fetch session after agent run
            refreshed = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=job_id
            )
            active_session = refreshed if refreshed is not None else lock_session

            run_state = SessionStateAccessor(active_session.state)
            rec_data = extract_book_recommendations_from_state(run_state)

            # Output-side grounding guard: drop any formatter title that
            # is not present in the grounded researcher candidates. Pairs
            # with the relaxed schema floor (min REC_MIN_RESULTS, not a hard
            # 5) so a thin researcher result yields fewer honest picks rather
            # than an invented book. Fail-open (see filter docstring).
            researcher_text = capture.text_for("book_recommendation_researcher")
            rec_data = filter_grounded_recommendations(rec_data, researcher_text)

            if rec_data is None:
                await self._clear_book_recs_lock(job_id, user_id)
                for ev in error_events(
                    job_id,
                    "Failed to extract book recommendations from agent response",
                    "ExtractionError",
                    phase=Phase.COMPOSITION,
                ):
                    yield ev
                return

            recommendations = rec_data.get("recommendations", [])
            floor = getattr(self._config, "rec_min_results", 3)
            if len(recommendations) < floor:
                logger.info(
                    "book_recommendations_below_floor",
                    count=len(recommendations),
                    floor=floor,
                )
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
            await self._session_service.append_event(active_session, persist_event)

            token_usage = await collect_token_usage(langfuse_plugin)

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

        async def timeout_cleanup() -> None:
            await self._clear_book_recs_lock(job_id, user_id)
            await self._mark_session_failed(job_id, user_id)

        spec = GuardSpec(
            flow_name="recommend_books",
            job_id=job_id,
            timeout_seconds=self._config.workflow_timeout,
            timeout_message=(
                f"Book recommendation timed out after {self._config.workflow_timeout}s"
            ),
            phase=Phase.COMPOSITION,
            on_timeout=timeout_cleanup,
            on_cancel=lambda: self._clear_book_recs_lock(job_id, user_id),
            on_error=lambda: self._clear_book_recs_lock(job_id, user_id),
            map_exception=lambda e: _map_phase_exception(
                "recommend_books", Phase.COMPOSITION, e
            ),
        )
        async for ev in run_guarded(body(), spec):
            yield ev

    def _get_session_lock(self, job_id: str) -> asyncio.Lock:
        """Per-job_id lock guarding the concurrency check-then-set in
        expand()/recommend_books() (MYS-401). See the registry's docstring
        in __init__ for why an in-process lock is sufficient today.

        Registry invariant (MYS-401 r6 -- re-owned after r4/r5 each closed
        one hop of the same hole one call site at a time): a lock this
        method returns to a caller is NEVER a candidate for eviction on
        THIS call. That used to be true only because a brand-new entry
        usually sorts after the entries eviction considers first -- true
        on realistic traffic, but not structurally: if every pre-existing
        entry happened to be held or waited-on, eviction would walk past
        all of them and delete the entry this same call had just created,
        one line above. Evicting strictly BEFORE creating the new entry
        (this method, below) removes that failure mode by construction:
        the entry doesn't exist yet when eviction runs, so it cannot be
        iterated over, let alone deleted, regardless of how many older
        entries turn out to be evictable.
        """
        lock = self._session_locks.get(job_id)
        if lock is not None:
            self._session_locks.move_to_end(job_id)
            return lock
        self._evict_stale_session_locks()
        lock = asyncio.Lock()
        self._session_locks[job_id] = lock
        return lock

    def _evict_stale_session_locks(self) -> None:
        """Make room in the per-job_id lock registry (MYS-401 r4-r6).

        Called ONLY from _get_session_lock, and only BEFORE it inserts a
        new entry -- never after. That ordering is what makes the
        invariant in _get_session_lock's docstring structural rather than
        a timing accident: this method only ever sees entries that
        existed before the current call, so it can never evict the lock
        _get_session_lock is about to hand back.

        Evicts least-recently-used entries down to (just under) the cap,
        but ONLY ones that are neither currently held nor have a pending
        waiter. ``asyncio.Lock`` has no public API for "is anyone waiting
        on this", hence the ``_waiters`` check; skipping both conditions
        matters because evicting a lock a genuinely-overlapping second
        caller is about to acquire would hand that caller a DIFFERENT
        ``Lock`` instance from the one an earlier caller (or another
        waiter) holds a reference to -- silently defeating the per-job_id
        serialization this whole registry exists for. If every existing
        entry is held or waited-on, this call evicts nothing and the
        registry is left at (or, after the pending insert, one above) the
        cap -- the cap is a generous leak bound, not a hard limit, and
        reaching this branch needs a genuinely pathological number of
        concurrently in-flight jobs. Ordinary traffic never reaches it.
        """
        if len(self._session_locks) < _SESSION_LOCK_REGISTRY_CAP:
            return
        for job_id, lock in list(self._session_locks.items()):
            if len(self._session_locks) < _SESSION_LOCK_REGISTRY_CAP:
                break
            if lock.locked() or getattr(lock, "_waiters", None):
                continue
            del self._session_locks[job_id]

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
        close = getattr(self._discovery_cache, "close", None)
        if callable(close):
            close()
