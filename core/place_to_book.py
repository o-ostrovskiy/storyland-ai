"""
Place→book reverse-routing capability (AI layer).

Resolves a destination to grounded, literal/vibe-labelled book candidates by
running the place_to_book pipeline, then applying an output-side grounding
filter and label invariants. Results are cached in-process under the key
``place2book:v1:<normalized place>``, mirroring the discovery result cache.

HTTP surface: exposed as the internal ``POST /place-to-book`` endpoint
(``api/routes.py``, gateway secret enforced). The storyland-services gateway
calls it and then runs the authoritative Google Books *existence* check on
each candidate, decorating the user-facing grounding object. The grounding
filter here is a defensive AI-side complement, not the hard gate.

Grounding contract:
  * Every candidate's title must appear in the grounded researcher text
    (output-side guard against the formatter inventing a title).
  * ``literal`` matches must name a real ``maps_to`` location; a literal claim
    without one is dropped (it cannot be a grounded "set there" match).
  * ``vibe`` matches always have ``maps_to=None``.
  * If nothing survives, the resolver returns a clean not-found result with an
    empty candidate list — never a fabricated list.
"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional, Tuple

from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.tools import google_search
from google.genai import types

from agents.orchestrator import create_place_to_book_workflow
from common.logging import get_logger
from models.place_to_book import PlaceBookCandidate, PlaceToBookCandidates, PlaceToBookResult
from plugins.langfuse_plugin import LangfusePlugin
from services.session_service import create_session_service

from .cache import TTLCache
from .extraction import grounding_token_set, is_title_grounded
from .run_harness import RunCapture, pump_events
from .session_state import SessionStateAccessor

logger = get_logger("storyland.core.place_to_book")

# Same app name the executor uses, so sessions live in one namespace.
APP_NAME = "storyland"

# Cache key version; bump when the candidate shape or grounding rules change.
_CACHE_KEY_PREFIX = "place2book:v1:"

# Cache sizing: place→book lookups are cheap to keep and repeat often for the
# seeded/landing destinations. 6h TTL mirrors the BE place→book hit TTL.
_DEFAULT_TTL_SECONDS = 6 * 60 * 60
_DEFAULT_MAX_ENTRIES = 500

_COUNTRY_SUFFIXES = (
    "usa",
    "united states",
    "uk",
    "united kingdom",
    "england",
    "scotland",
    "ireland",
    "france",
    "italy",
    "spain",
    "portugal",
    "japan",
    "russia",
)


def normalize_place(place: object) -> str:
    """Normalize a free-text place into a stable cache key.

    Lowercases, strips surrounding punctuation/whitespace, collapses internal
    whitespace, and removes a single trailing ", <country>" qualifier so
    "Lisbon" and "Lisbon, Portugal" share a key. Intentionally lighter than the
    BE normalizer (which owns the canonical alias map) — this only needs to be a
    consistent in-process cache key on the AI side.
    """
    if not isinstance(place, str):
        return ""
    text = re.sub(r"\s+", " ", place).strip().lower()
    text = text.strip(" ,.;:'\"")
    if "," in text:
        head, _, tail = text.rpartition(",")
        head = head.strip()
        tail = tail.strip()
        if head and tail in _COUNTRY_SUFFIXES:
            text = head
    return text


def cache_key(place: str) -> str:
    """Return the versioned cache key for a (raw) place string."""
    return f"{_CACHE_KEY_PREFIX}{normalize_place(place)}"


# The grounding-match primitive (grounding_token_set / is_title_grounded) is
# shared with core.extraction (imported above). The grounding FILTERS
# themselves stay separate on purpose: this module's filter treats an
# all-dropped result as a valid honest not-found, while the
# book-recommendation filter fails open on that outcome — different contracts,
# documented on each function.


def validate_place_to_book_candidates(value: object) -> Optional[List[dict]]:
    """Validate a PlaceToBookCandidates payload → list of candidate dicts.

    Returns the validated ``candidates`` list (possibly empty) or ``None`` if
    the payload is unparseable/invalid.
    """
    import json

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    try:
        validated = PlaceToBookCandidates.model_validate(value)
    except Exception:
        return None
    return [c.model_dump() for c in validated.candidates]


def extract_place_to_book_from_state(state_accessor) -> Optional[List[dict]]:
    """Extract and validate the formatter's candidate list from session state."""
    return validate_place_to_book_candidates(state_accessor.last_place_to_book)


def filter_grounded_candidates(
    candidates: Optional[List[dict]], researcher_text: str
) -> List[dict]:
    """Drop candidates whose title is not grounded in the researcher text
    (per the shared ``is_title_grounded`` token-overlap rule).

    Output-side complement to the formatter's "never invent" instruction. Unlike
    the book-recommendation filter, dropping every candidate is a VALID outcome
    here (it produces the honest not-found state), so this does not fail open on
    a fully-ungrounded result. It DOES fail open when no researcher text was
    captured at all (a plumbing/capture miss is not proof of fabrication — and
    BE's Google Books existence check remains the hard gate downstream).
    """
    if not candidates:
        return []
    haystack = grounding_token_set(researcher_text)
    if not haystack:
        return list(candidates)
    grounded = [c for c in candidates if is_title_grounded(c.get("title"), haystack)]
    if len(grounded) != len(candidates):
        logger.info(
            "place_to_book_grounding_filtered",
            kept=len(grounded),
            dropped=len(candidates) - len(grounded),
        )
    return grounded


def enforce_label_invariants(candidates: List[dict]) -> List[dict]:
    """Enforce literal/vibe invariants; drop candidates that cannot be honest.

    * ``vibe`` → ``maps_to`` forced to None (it is not set in the place).
    * ``literal`` → kept only if it names a real ``maps_to`` location; a literal
      claim without a location is not a grounded "set there" match, so it is
      dropped rather than surfaced.
    * Any unknown ``match_type`` is dropped.
    """
    cleaned: List[dict] = []
    for c in candidates:
        match_type = c.get("match_type")
        if match_type == "vibe":
            entry = dict(c)
            entry["maps_to"] = None
            cleaned.append(entry)
        elif match_type == "literal":
            if (c.get("maps_to") or "").strip():
                cleaned.append(dict(c))
            else:
                logger.info("place_to_book_dropped_literal_without_location", title=c.get("title"))
    return cleaned


class PlaceToBookResolver:
    """Resolve a destination to grounded place→book candidates.

    Runs the place_to_book pipeline, applies grounding + label invariants, and
    caches the result. Designed to be driven directly (tests / EVAL / a future
    BE call), independent of the HITL discovery/compose session lifecycle.
    """

    def __init__(
        self,
        model,
        session_service=None,
        cache: Optional[TTLCache] = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._model = model
        self._session_service = session_service or create_session_service(
            use_database=False
        )
        self._cache = cache or TTLCache(
            ttl_seconds=ttl_seconds, max_entries=max_entries
        )

    def _build_runner(self, workflow) -> Runner:
        """Build a Runner for one pipeline run.

        Mirrors ``WorkflowExecutor._build_runner`` — references the
        module-level ``Runner`` name so tests keep monkeypatching
        ``core.place_to_book.Runner`` as the single fake seam.
        """
        # MYS-815 r2: the reverse-discovery researcher is configured with
        # google_search, so it belongs in the "did researchers actually
        # search?" measurement -- and with only LoggingPlugin registered it
        # emitted neither search_grounding_captured nor _absent, silently
        # excluding every /place-to-book cache miss from the numbers.
        #
        # Constructed WITHOUT credentials on purpose: this adds the receipts
        # and nothing else. LangfusePlugin with no keys sets enabled=False and
        # opens no client, while _log_search_grounding runs BEFORE that gate
        # (see its docstring) -- so this changes what we can measure, not what
        # this path sends anywhere.
        observer = LangfusePlugin()
        observer.root_name = getattr(workflow, "name", None)
        return Runner(
            node=workflow,
            app_name=APP_NAME,
            session_service=self._session_service,
            plugins=[LoggingPlugin(), observer],
        )

    async def resolve(self, place: str) -> PlaceToBookResult:
        """Resolve ``place`` to a PlaceToBookResult (cached)."""
        normalized = normalize_place(place)
        if not normalized:
            return self._not_found(place, normalized)

        key = f"{_CACHE_KEY_PREFIX}{normalized}"
        cached = await self._cache.get(key)
        if cached is not None:
            logger.info("place_to_book_cache_hit", place=normalized)
            return cached

        raw_candidates, researcher_text = await self._run_pipeline(place)
        grounded = filter_grounded_candidates(raw_candidates, researcher_text)
        cleaned = enforce_label_invariants(grounded)

        if cleaned:
            result = PlaceToBookResult(
                place=place,
                query=normalized,
                found=True,
                message=None,
                candidates=[PlaceBookCandidate(**c) for c in cleaned],
            )
        else:
            result = self._not_found(place, normalized)

        await self._cache.set(key, result)
        logger.info(
            "place_to_book_resolved",
            place=normalized,
            found=result.found,
            count=len(result.candidates),
        )
        return result

    def _not_found(self, place: str, normalized: str) -> PlaceToBookResult:
        return PlaceToBookResult(
            place=place,
            query=normalized,
            found=False,
            message=f"We haven't mapped {place} yet.",
            candidates=[],
        )

    async def _run_pipeline(self, place: str) -> Tuple[List[dict], str]:
        """Run the pipeline once; return (candidate dicts, researcher text)."""
        job_id = str(uuid.uuid4())
        user_id = "place2book"

        await self._session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=job_id, state={}
        )

        workflow = create_place_to_book_workflow(
            self._model, google_search, place=place
        )
        runner = self._build_runner(workflow)

        prompt = f"What should I read before travelling to {place}?"
        message = types.Content(role="user", parts=[types.Part(text=prompt)])

        # Drain via the shared harness pump; no agent_steps means no progress
        # events are produced — we only capture the grounded researcher text.
        capture = RunCapture()
        async for _ in pump_events(
            runner,
            user_id=user_id,
            session_id=job_id,
            message=message,
            agent_steps={},
            capture=capture,
            capture_authors=("place_to_book_researcher",),
        ):
            pass

        refreshed = await self._session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=job_id
        )
        candidates: List[dict] = []
        if refreshed is not None:
            run_state = SessionStateAccessor(refreshed.state)
            candidates = extract_place_to_book_from_state(run_state) or []

        return candidates, capture.text_for("place_to_book_researcher")
