"""A journey starts from a work: book, film or series (ADR #31, ai-private#66).

Pins the compatibility rule: a book-only request is byte-identical to the
pre-work contract (metadata event JSON and discovery cache key, both pinned
as literals captured from main before this change), while a film request is
accepted, surfaces its kind and never shares a cache entry with a book.
"""

import json

import pytest
from pydantic import ValidationError

from api.models import (
    MAX_AUTHOR_LENGTH,
    MAX_TITLE_LENGTH,
    DiscoverRequest,
    LocalAtmosphereRequest,
    SSEMetadataEvent,
)
from api.streaming import domain_event_to_sse
from core.events import MetadataReady
from core.executor import WorkflowExecutor, _metadata_payload
from core.extraction import clear_screen_claims_unless_screen_work
from core.session_state import SessionStateAccessor, SessionStateKeys
from models.itinerary import CityStop, ComposerEnvelope
from models.work import MAX_WORK_CREATOR_LENGTH, MAX_WORK_TITLE_LENGTH, WorkRef

from tests.unit.test_golden_stream import (
    _ENVELOPE,
    _REGIONS,
    _make_executor,
    _state_writing_runner,
)

# Captured from main @301b9a7 before ADR #31 (see PR): what a book request's
# metadata event and cache key were. They must never change.
BOOK_METADATA_SSE_TODAY = (
    '{"event":"metadata","book_title":"Persuasion","author":"Jane Austen",'
    '"description":null,"published_date":null,"categories":[],"image_url":null}'
)
BOOK_CACHE_KEY_TODAY = (
    "discover:v3:persuasion|jane austen|97d170e1550eee4afc0af065b78cda302a97674c"
)
BOOK_CACHE_KEY_ALL_OPTIONS_TODAY = (
    "discover:v3:persuasion|jane austen|8f20cb11d691e4059f5983b620cdfbb57f82488a"
    "|vibe=cozy|taste=f30200283aad"
)

FILM = {"kind": "film", "title": "Persuasion", "creator": "Carrie Cracknell", "year": 2022}


def _location():
    return {"lat": 51.38, "lng": -2.36, "label": "Bath, England"}


class TestWorkRef:
    def test_bounds_match_the_book_fields(self):
        assert MAX_WORK_TITLE_LENGTH == MAX_TITLE_LENGTH
        assert MAX_WORK_CREATOR_LENGTH == MAX_AUTHOR_LENGTH

    def test_kind_defaults_to_book(self):
        assert WorkRef(title="Emma", creator="Jane Austen").kind == "book"

    @pytest.mark.parametrize("kind", ["podcast", "Film", ""])
    def test_unknown_kind_rejected(self, kind):
        with pytest.raises(ValidationError):
            WorkRef(kind=kind, title="X", creator="Y")

    @pytest.mark.parametrize(
        "field,value",
        [
            ("title", "x" * (MAX_TITLE_LENGTH + 1)),
            ("creator", "x" * (MAX_AUTHOR_LENGTH + 1)),
            ("title", "   "),
            ("creator", "   "),
            ("year", 1066),
            ("year", 3000),
        ],
    )
    def test_invalid_members_rejected(self, field, value):
        data = dict(FILM, **{field: value})
        with pytest.raises(ValidationError):
            WorkRef(**data)

    def test_title_and_creator_are_stripped(self):
        ref = WorkRef(kind="film", title="  Persuasion ", creator=" Carrie Cracknell ")
        assert (ref.title, ref.creator) == ("Persuasion", "Carrie Cracknell")


@pytest.mark.parametrize("request_cls,extra", [
    (DiscoverRequest, {}),
    (LocalAtmosphereRequest, {"user_location": _location()}),
])
class TestRequestsAcceptWork:
    def test_book_only_request_unchanged(self, request_cls, extra):
        req = request_cls(book_title="Persuasion", author="Jane Austen", **extra)
        assert req.work is None
        assert req.work_ref == WorkRef(kind="book", title="Persuasion", creator="Jane Austen")

    def test_book_only_request_still_requires_both_fields(self, request_cls, extra):
        with pytest.raises(ValidationError) as exc:
            request_cls(book_title="Persuasion", **extra)
        assert [e["loc"] for e in exc.value.errors()] == [("author",)]

    def test_film_work_alone_is_accepted_and_mirrored(self, request_cls, extra):
        req = request_cls(work=FILM, **extra)
        assert req.work_ref.kind == "film"
        assert (req.book_title, req.author) == ("Persuasion", "Carrie Cracknell")

    def test_matching_book_fields_alongside_work_are_fine(self, request_cls, extra):
        req = request_cls(
            book_title="persuasion", author="Carrie Cracknell", work=FILM, **extra
        )
        assert req.work_ref.kind == "film"

    def test_conflicting_book_fields_and_work_rejected(self, request_cls, extra):
        with pytest.raises(ValidationError, match="must name the same work"):
            request_cls(book_title="Emma", author="Jane Austen", work=FILM, **extra)

    def test_over_long_work_title_rejected(self, request_cls, extra):
        with pytest.raises(ValidationError):
            request_cls(work=dict(FILM, title="x" * (MAX_TITLE_LENGTH + 1)), **extra)

    def test_invalid_kind_rejected(self, request_cls, extra):
        with pytest.raises(ValidationError):
            request_cls(work=dict(FILM, kind="podcast"), **extra)


class TestMetadataEvent:
    def test_book_metadata_event_is_byte_identical(self):
        sse = domain_event_to_sse(
            MetadataReady(metadata=_metadata_payload("Persuasion", "Jane Austen"))
        )
        assert sse["data"] == BOOK_METADATA_SSE_TODAY

    def test_explicit_book_work_is_byte_identical_too(self):
        work = WorkRef(kind="book", title="Persuasion", creator="Jane Austen")
        sse = domain_event_to_sse(
            MetadataReady(metadata=_metadata_payload("Persuasion", "Jane Austen", work))
        )
        assert sse["data"] == BOOK_METADATA_SSE_TODAY

    def test_film_metadata_event_carries_kind_and_creator(self):
        work = WorkRef(**FILM)
        sse = domain_event_to_sse(
            MetadataReady(metadata=_metadata_payload("Persuasion", "Carrie Cracknell", work))
        )
        data = json.loads(sse["data"])
        assert data["kind"] == "film"
        assert data["creator"] == "Carrie Cracknell"
        # The book fields stay on the wire, never renamed.
        assert data["book_title"] == "Persuasion"
        assert data["author"] == "Carrie Cracknell"

    def test_serializer_only_drops_the_new_fields(self):
        dumped = json.loads(SSEMetadataEvent(book_title="A", author="B").model_dump_json())
        assert "kind" not in dumped and "creator" not in dumped
        assert dumped["description"] is None  # older nullable fields still emitted


class TestCacheKey:
    def test_book_key_pinned_to_today(self):
        assert (
            WorkflowExecutor._discovery_cache_key("Persuasion", "Jane Austen", None)
            == BOOK_CACHE_KEY_TODAY
        )
        assert (
            WorkflowExecutor._discovery_cache_key(
                "Persuasion", "Jane Austen", None, kind="book"
            )
            == BOOK_CACHE_KEY_TODAY
        )

    def test_book_key_with_every_option_pinned_to_today(self):
        assert (
            WorkflowExecutor._discovery_cache_key(
                "Persuasion",
                "Jane Austen",
                {"budget": "luxury"},
                "cozy",
                {"titles": ["Emma"]},
            )
            == BOOK_CACHE_KEY_ALL_OPTIONS_TODAY
        )

    def test_film_and_series_keys_are_distinct(self):
        keys = {
            kind: WorkflowExecutor._discovery_cache_key(
                "Persuasion", "Jane Austen", None, kind=kind
            )
            for kind in ("book", "film", "series")
        }
        assert len(set(keys.values())) == 3


class TestDiscoverThreadsKind:
    async def _run_discover(self, monkeypatch, **kwargs):
        executor, ex = _make_executor(monkeypatch)
        monkeypatch.setattr(
            ex,
            "Runner",
            _state_writing_runner(
                lambda: {
                    SessionStateKeys.REGION_ANALYSIS: {
                        "regions": _REGIONS,
                        "analysis_note": "note",
                    }
                }
            ),
        )
        seen_keys = []
        real_key = executor._discovery_cache_key

        def spy(*args, **kw):
            key = real_key(*args, **kw)
            seen_keys.append(key)
            return key

        monkeypatch.setattr(executor, "_discovery_cache_key", spy)
        events = [ev async for ev in executor.discover(**kwargs)]
        return executor, events, seen_keys

    async def test_book_discover_metadata_and_key_unchanged(self, monkeypatch):
        executor, events, keys = await self._run_discover(
            monkeypatch, book_title="Persuasion", author="Jane Austen"
        )
        metadata = [domain_event_to_sse(e) for e in events if isinstance(e, MetadataReady)]
        assert [m["data"] for m in metadata] == [BOOK_METADATA_SSE_TODAY]
        assert keys == [BOOK_CACHE_KEY_TODAY]

    async def test_film_discover_surfaces_kind_and_distinct_key(self, monkeypatch):
        from core.executor import APP_NAME

        executor, events, keys = await self._run_discover(
            monkeypatch,
            book_title="Persuasion",
            author="Carrie Cracknell",
            work=WorkRef(**FILM),
        )
        metadata = [
            json.loads(domain_event_to_sse(e)["data"])
            for e in events
            if isinstance(e, MetadataReady)
        ]
        assert [m["kind"] for m in metadata] == ["film"]
        assert keys and keys[0].endswith("|kind=film")
        job_id = events[0].job_id
        session = await executor._session_service.get_session(
            app_name=APP_NAME, user_id="default", session_id=job_id
        )
        assert SessionStateAccessor(session.state).work_kind == "film"


def _envelope_with_claim(claim):
    envelope = json.loads(json.dumps(_ENVELOPE))
    envelope["itinerary"]["cities"][0]["stops"][0]["screen_claim"] = claim
    return envelope


class TestScreenClaim:
    def test_defaults_to_null_and_filming_scene_unchanged(self):
        stop = CityStop(name="Pump Room", type="landmark", reason="r", time_of_day="morning")
        assert stop.screen_claim is None
        assert stop.filming_scene is None

    @pytest.mark.parametrize("claim", ["filmed_here", "set_here", None])
    def test_round_trips_through_composer_envelope(self, claim):
        envelope = ComposerEnvelope.model_validate(_envelope_with_claim(claim))
        again = ComposerEnvelope.model_validate_json(envelope.model_dump_json())
        assert again.itinerary.cities[0].stops[0].screen_claim == claim

    def test_unknown_claim_rejected(self):
        with pytest.raises(ValidationError):
            ComposerEnvelope.model_validate(_envelope_with_claim("shot_nearby"))

    def test_book_journey_drops_a_composer_claim(self):
        itinerary = _envelope_with_claim("filmed_here")["itinerary"]
        out = clear_screen_claims_unless_screen_work(itinerary, "book")
        assert out["cities"][0]["stops"][0]["screen_claim"] is None

    @pytest.mark.parametrize("kind", ["film", "series"])
    def test_screen_journey_keeps_its_claim(self, kind):
        itinerary = _envelope_with_claim("set_here")["itinerary"]
        out = clear_screen_claims_unless_screen_work(itinerary, kind)
        assert out["cities"][0]["stops"][0]["screen_claim"] == "set_here"

    def test_expansion_places_cleared_for_books(self):
        expansion = {"places": [{"name": "x", "screen_claim": "filmed_here"}]}
        out = clear_screen_claims_unless_screen_work(expansion, "book", stops_path="expansion")
        assert out["places"][0]["screen_claim"] is None

    async def test_compose_of_a_book_journey_ships_no_claim(self, monkeypatch):
        """End to end: the real compose path nulls a claim on a book journey."""
        executor, ex = _make_executor(monkeypatch)
        monkeypatch.setattr(
            ex,
            "Runner",
            _state_writing_runner(
                lambda: {
                    SessionStateKeys.REGION_ANALYSIS: {
                        "regions": _REGIONS,
                        "analysis_note": "note",
                    }
                }
            ),
        )
        job_id = None
        async for ev in executor.discover(book_title="Persuasion", author="Jane Austen"):
            job_id = getattr(ev, "job_id", job_id)
        monkeypatch.setattr(
            ex,
            "Runner",
            _state_writing_runner(
                lambda: {SessionStateKeys.COMPOSER_ENVELOPE: _envelope_with_claim("filmed_here")}
            ),
        )
        itineraries = [
            ev.itinerary
            async for ev in executor.compose(job_id=job_id, region_ids=[1])
            if type(ev).__name__ == "ItineraryReady"
        ]
        assert itineraries
        assert itineraries[0]["cities"][0]["stops"][0]["screen_claim"] is None


class TestPromptSeam:
    def test_book_returns_the_book_prompts_unchanged(self):
        from agents.prompts import load_prompts
        from agents.work_prompts import for_kind

        assert for_kind("book") == load_prompts()

    @pytest.mark.parametrize("kind", ["film", "series"])
    def test_screen_kinds_fall_back_until_story_4(self, kind):
        from agents.prompts import load_prompts
        from agents.work_prompts import for_kind

        assert for_kind(kind) == load_prompts()

    def test_unknown_kind_raises(self):
        from agents.work_prompts import for_kind

        with pytest.raises(ValueError):
            for_kind("podcast")

    def test_seam_module_is_outside_the_cache_fingerprint(self):
        """Adding the seam must not cold-start the book cache (card Risks)."""
        from core.cache_version import _PROMPT_MODULES

        assert "agents.work_prompts" not in _PROMPT_MODULES
