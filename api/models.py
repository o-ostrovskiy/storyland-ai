"""
API request and response models.

Defines Pydantic models for:
- HTTP request bodies (DiscoverRequest, ComposeRequest)
- SSE event payloads (progress, metadata, regions, itinerary, error, done)
- REST response bodies (HealthResponse, JobStatusResponse)
"""

from enum import Enum
from typing import Optional, List, Literal

from pydantic import BaseModel, Field, field_validator


# --- Request Models ---

# Input-size bounds: reject oversized payloads with a 422 BEFORE any prompt is
# built or Gemini is called (cost + prompt-injection-surface guard). 200 matches
# the existing UserLocation.label / ExpandRequest caps.
MAX_TITLE_LENGTH = 200
MAX_AUTHOR_LENGTH = 200
MAX_PREFERENCES_KEYS = 30
MAX_PREFERENCE_KEY_LENGTH = 100
MAX_PREFERENCE_VALUE_LENGTH = 500


def _bound_preferences(value):
    """Bound the free-form preferences dict (key count + per-key/value length)."""
    if value is None:
        return value
    if len(value) > MAX_PREFERENCES_KEYS:
        raise ValueError(
            f"preferences may contain at most {MAX_PREFERENCES_KEYS} keys"
        )
    for key, val in value.items():
        if not isinstance(key, str) or len(key) > MAX_PREFERENCE_KEY_LENGTH:
            raise ValueError(
                "preferences keys must be strings of at most "
                f"{MAX_PREFERENCE_KEY_LENGTH} characters"
            )
        if len(str(val)) > MAX_PREFERENCE_VALUE_LENGTH:
            raise ValueError(
                f"preferences values must be at most {MAX_PREFERENCE_VALUE_LENGTH} characters"
            )
    return value


# Curated mood/vibe set mirrored from the gateway (storyland-services). The
# gateway already validates + normalizes before forwarding, but we re-validate
# here so the AI service never trusts an unbounded value into the discovery
# prompt (defense in depth). Canonical form is lower-case; "slow-burn" keeps its
# hyphen. Absent ⇒ today's behavior exactly; an unrecognized value is a 422.
ALLOWED_VIBES = frozenset(
    {"cozy", "melancholic", "adventurous", "slow-burn", "atmospheric", "hopeful"}
)


def _validate_vibe(value):
    """Normalize + validate an optional mood/vibe token.

    Returns the canonical lower-case token, or None when absent/blank. Raises
    ValueError (→ 422) for a non-string or an out-of-set value so an unknown
    vibe is never silently dropped into the prompt.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("vibe must be a string")
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in ALLOWED_VIBES:
        allowed = ", ".join(sorted(ALLOWED_VIBES))
        raise ValueError(f"vibe must be one of: {allowed}")
    return normalized


# Bounds for the optional imported-reading-history taste signal (StoryGraph /
# Goodreads CSV import, PR1). The gateway (storyland-services) already parses +
# bounds it, but the AI service re-bounds here so an unbounded history can never
# inflate the discovery prompt (cost + prompt-injection surface). We TRUNCATE
# rather than 422: a large legitimate reading history is normal input, not an
# attack, and rejecting it would needlessly fail discovery.
MAX_TASTE_TITLES = 20
MAX_TASTE_MOODS = 12
MAX_TASTE_STRING_LENGTH = 200


def _bound_taste_context(value):
    """Normalize + bound an optional ``{"titles": [...], "moods": [...]}`` block.

    Strings are stripped and length-capped; blanks dropped; each list is
    de-duplicated case-insensitively (first occurrence wins, order preserved)
    and truncated to its cap. Returns a normalized dict, or None when
    absent/empty so the prompt + cache key stay byte-identical to a no-taste
    request. Raises ValueError (→ 422) only for a structurally wrong type.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("taste_context must be an object")

    def _clean(items, cap):
        if items is None:
            return []
        if not isinstance(items, (list, tuple)):
            raise ValueError("taste_context titles/moods must be lists")
        out, seen = [], set()
        for item in items:
            if not isinstance(item, str):
                continue
            normalized = item.strip()[:MAX_TASTE_STRING_LENGTH].strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(normalized)
            if len(out) >= cap:
                break
        return out

    titles = _clean(value.get("titles"), MAX_TASTE_TITLES)
    moods = _clean(value.get("moods"), MAX_TASTE_MOODS)
    if not titles and not moods:
        return None
    result = {}
    if titles:
        result["titles"] = titles
    if moods:
        result["moods"] = moods
    return result


class DiscoverRequest(BaseModel):
    """Request body for POST /api/v1/itinerary/discover."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "book_title": "1984",
                    "author": "George Orwell",
                },
                {
                    "book_title": "Pride and Prejudice",
                    "author": "Jane Austen",
                    "preferences": {"budget": "luxury", "preferred_pace": "relaxed"},
                },
            ]
        }
    }

    book_title: str = Field(
        min_length=1, max_length=MAX_TITLE_LENGTH, description="Title of the book"
    )
    author: str = Field(
        min_length=1, max_length=MAX_AUTHOR_LENGTH, description="Author name (required)"
    )
    preferences: Optional[dict] = Field(
        default=None, description="User travel preferences"
    )
    vibe: Optional[str] = Field(
        default=None,
        description=(
            "Optional explicit mood/vibe to bias discovery toward (curated set: "
            "cozy, melancholic, adventurous, slow-burn, atmospheric, hopeful). "
            "Absent ⇒ unchanged behavior."
        ),
    )
    taste_context: Optional[dict] = Field(
        default=None,
        description=(
            "Optional imported reading-history taste signal "
            '({"titles": [...], "moods": [...]}) used to bias discovery toward '
            "grounded places that resonate with the reader's demonstrated taste. "
            "Bounded + de-duplicated server-side; absent ⇒ unchanged behavior."
        ),
    )

    @field_validator("book_title")
    @classmethod
    def validate_book_title(cls, value: str) -> str:
        """Require a non-empty, non-whitespace book title."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("book_title must not be empty")
        return normalized

    @field_validator("author")
    @classmethod
    def validate_author(cls, value: str) -> str:
        """Require a non-empty, non-whitespace author name."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("author must not be empty")
        return normalized

    @field_validator("preferences")
    @classmethod
    def validate_preferences(cls, value):
        """Bound preferences size before it flows into the prompt."""
        return _bound_preferences(value)

    @field_validator("vibe")
    @classmethod
    def validate_vibe(cls, value):
        """Validate/normalize the optional mood/vibe token before the prompt."""
        return _validate_vibe(value)

    @field_validator("taste_context")
    @classmethod
    def validate_taste_context(cls, value):
        """Normalize + bound the optional imported taste signal before the prompt."""
        return _bound_taste_context(value)


class ComposeRequest(BaseModel):
    """Request body for POST /api/v1/itinerary/{job_id}/compose."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"region_ids": [1]},
                {"region_ids": [1, 2]},
            ]
        }
    }

    region_ids: List[int] = Field(
        min_length=1,
        description="Selected region IDs from the discover response",
    )


class ExpandRequest(BaseModel):
    """Request body for POST /api/v1/itinerary/{job_id}/expand."""

    action_id: str = Field(
        min_length=1,
        description="ID of the suggestion chip that was clicked (server-issued uuid4)",
    )
    action_label: str = Field(
        min_length=1,
        max_length=100,
        description="Human-readable chip label (for logging only)",
    )
    action_prompt: str = Field(
        min_length=1,
        max_length=500,
        description="Expansion instruction carried by the chip",
    )


class RecommendBooksRequest(BaseModel):
    """Request body for POST /api/v1/itinerary/{job_id}/recommend-books."""

    action_id: str = Field(
        min_length=1,
        description="ID of the 'Find books like this' chip (server-issued uuid4)",
    )
    action_label: str = Field(
        min_length=1,
        max_length=100,
        description="Human-readable chip label (for logging only)",
    )
    action_prompt: str = Field(
        default="",
        max_length=500,
        description="Unused for book recommendations; kept for interface symmetry with expand",
    )


class PlaceToBookRequest(BaseModel):
    """Request body for POST /api/v1/place-to-book (internal: gateway → AI).

    Reverse-discovery input: a free-text destination resolved to grounded,
    literal/vibe-labelled book candidates. The gateway (storyland-services)
    calls this endpoint, then runs the authoritative Google Books existence
    check on each returned candidate.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"place": "Lisbon"},
                {"place": "the Scottish Highlands"},
            ]
        }
    }

    place: str = Field(
        min_length=1,
        max_length=120,
        description="Free-text destination (e.g. 'Lisbon', 'Tokyo, Japan')",
    )

    @field_validator("place")
    @classmethod
    def validate_place(cls, value: str) -> str:
        """Require a non-empty, non-whitespace place string."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("place must not be empty")
        return normalized


class UserLocation(BaseModel):
    """User's current location for the local-atmosphere flow."""

    lat: float = Field(ge=-90, le=90, description="Latitude in decimal degrees")
    lng: float = Field(ge=-180, le=180, description="Longitude in decimal degrees")
    label: str = Field(
        min_length=1,
        max_length=200,
        description="Human-readable location (e.g., 'New York, NY 10013')",
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("label must not be empty")
        return normalized


class LocalAtmosphereRequest(BaseModel):
    """Request body for POST /api/v1/itinerary/local-atmosphere.

    Single-phase flow: caller supplies a book and a current location, the
    workflow returns a TripItinerary of nearby places that match the book's
    atmosphere — no region selection step.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "book_title": "Wuthering Heights",
                    "author": "Emily Brontë",
                    "user_location": {
                        "lat": 40.7128,
                        "lng": -74.0060,
                        "label": "New York, NY 10013",
                    },
                    "radius_km": 80,
                    "preferences": {
                        "budget": "moderate",
                        "preferred_pace": "relaxed",
                    },
                },
            ]
        }
    }

    book_title: str = Field(
        min_length=1, max_length=MAX_TITLE_LENGTH, description="Title of the book"
    )
    author: str = Field(
        min_length=1, max_length=MAX_AUTHOR_LENGTH, description="Author name (required)"
    )
    user_location: UserLocation = Field(description="User's current location")
    radius_km: int = Field(
        default=80,
        ge=10,
        le=200,
        description="Search radius in km from the user's location",
    )
    preferences: Optional[dict] = Field(
        default=None, description="User travel preferences"
    )

    @field_validator("book_title")
    @classmethod
    def validate_book_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("book_title must not be empty")
        return normalized

    @field_validator("author")
    @classmethod
    def validate_author(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("author must not be empty")
        return normalized

    @field_validator("preferences")
    @classmethod
    def validate_preferences(cls, value):
        return _bound_preferences(value)


# --- SSE Event Models ---


class SSEProgressEvent(BaseModel):
    """Progress update during workflow execution."""

    event: Literal["progress"] = "progress"
    phase: int = Field(description="Current phase (1, 2, or 3)")
    step: str = Field(description="Current step description")
    detail: Optional[str] = Field(
        default=None, description="Additional detail"
    )


class SSEStartedEvent(BaseModel):
    """Job has been registered with a session and has a stable job_id.

    Emitted as early as possible so a client whose SSE connection drops mid-run
    can recover the job via GET /itinerary/{job_id}/status.
    """

    event: Literal["started"] = "started"
    job_id: str


class SSEMetadataEvent(BaseModel):
    """Book metadata confirmed from the upstream service."""

    event: Literal["metadata"] = "metadata"
    book_title: str
    author: str
    description: Optional[str] = None
    published_date: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    image_url: Optional[str] = None


class SSERegionsEvent(BaseModel):
    """Region analysis results for user selection."""

    event: Literal["regions"] = "regions"
    job_id: str = Field(description="Job ID for the compose endpoint")
    regions: List[dict] = Field(description="Region options from RegionAnalysis")
    analysis_note: str = Field(default="")


class SSEItineraryEvent(BaseModel):
    """Final itinerary result."""

    event: Literal["itinerary"] = "itinerary"
    itinerary: dict = Field(description="TripItinerary as dict")
    suggestions: List[dict] = Field(
        default_factory=list,
        description="Contextual suggestion chips for follow-up expansions",
    )
    book_recommendation_chip: Optional[dict] = Field(
        default=None,
        description="Server-stamped 'Find books like this' chip (separate from expansion chips)",
    )


class SSEExpansionEvent(BaseModel):
    """Expansion result: new places added to an existing city."""

    event: Literal["expansion"] = "expansion"
    parent_city: str = Field(description="City where new places were added")
    places: List[dict] = Field(description="New CityStop entries")
    suggestions: List[dict] = Field(
        default_factory=list,
        description="Fresh contextual suggestion chips",
    )
    book_recommendation_chip: Optional[dict] = Field(
        default=None,
        description="Server-stamped 'Find books like this' chip (separate from expansion chips)",
    )


class SSEBookRecommendationsEvent(BaseModel):
    """Book recommendations result: 5 books related to the current book + destinations."""

    event: Literal["book_recommendations"] = "book_recommendations"
    recommendations: List[dict] = Field(description="BookRecommendation entries")
    book_recommendation_count: int = Field(
        description="Total number of recommendation requests made this session"
    )


class SSEErrorEvent(BaseModel):
    """Error during workflow execution."""

    event: Literal["error"] = "error"
    message: str
    error_type: str = Field(default="WorkflowError")
    phase: Optional[int] = Field(
        default=None, description="Phase where error occurred"
    )
    reason: Optional[str] = Field(
        default=None,
        description=(
            "Machine-readable failure class for recovery UX "
            '("transient" | "taste_validation"). Absent for unclassified errors.'
        ),
    )
    offending_title: Optional[str] = Field(
        default=None,
        description="taste_context title that failed validation (reason=taste_validation)",
    )


class SSEDoneEvent(BaseModel):
    """Stream completion marker."""

    event: Literal["done"] = "done"
    job_id: str
    token_usage: Optional[dict] = Field(
        default=None, description="Token usage stats if available"
    )


# --- REST Response Models ---


class JobStatus(str, Enum):
    """Possible job states derived from session state keys."""

    PENDING = "pending"
    SEARCHING = "searching"
    DISCOVERING = "discovering"
    REGIONS_READY = "regions_ready"
    COMPOSING = "composing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatusResponse(BaseModel):
    """Response for GET /api/v1/itinerary/{job_id}/status."""

    job_id: str
    status: JobStatus
    book_title: Optional[str] = None
    author: Optional[str] = None
    has_regions: bool = False
    has_itinerary: bool = False


class HealthResponse(BaseModel):
    """Response for GET /api/v1/health."""

    status: str = "healthy"
    version: str = "0.1.0"
    model_name: str = ""
