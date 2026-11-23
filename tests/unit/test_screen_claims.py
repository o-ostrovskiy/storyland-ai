"""ai-private#67 part 1: ScreenContext and the screen journey's set_here guardrail.

A ``set_here`` stop labelled literal/historical must carry a ``grounding_source``;
otherwise it leaves the service as ``filmed_here`` + ``vibe``, the weaker claim.
The same happens when the book-path downgrade rejects the stop's evidence.
Book journeys are untouched (their screen claims are nulled elsewhere).
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from core.extraction import extract_expansion_from_state, extract_itinerary_from_response
from core.guardrails.screen_claims import (
    demote_set_here_that_lost_grounding,
    downgrade_unsupported_set_here_claims,
    strong_set_here_stops,
)
from models.screen import MAX_SCREEN_LOCATIONS, ScreenContext
from models.work import WorkRef


def _stop(name, claim, match_type, source=None, scene=None):
    return {
        "name": name,
        "type": "landmark",
        "reason": "r",
        "time_of_day": "morning",
        "screen_claim": claim,
        "match_type": match_type,
        "grounding_source": source,
        "filming_scene": scene,
    }


def _itinerary(*stops, city="Paris", country="France"):
    return {
        "cities": [
            {
                "name": city,
                "country": country,
                "days_suggested": 2,
                "overview": "o",
                "stops": list(stops),
            }
        ],
        "summary_text": "s",
    }


def _claims(itinerary):
    return [
        (s["name"], s["screen_claim"], s["match_type"], s["grounding_source"])
        for c in itinerary["cities"]
        for s in c["stops"]
    ]


# ── ScreenContext ────────────────────────────────────────────────────────────


class TestScreenContext:
    def test_defaults_are_empty_and_real_world(self):
        ctx = ScreenContext()
        assert (ctx.story_locations, ctx.filming_locations, ctx.source_work, ctx.is_fictional_world) == (
            [], [], None, False
        )

    def test_an_invented_world_has_no_real_setting(self):
        ctx = ScreenContext(
            story_locations=["Westeros", "King's Landing"],
            filming_locations=["Dubrovnik", "Northern Ireland"],
            is_fictional_world=True,
        )
        assert ctx.story_locations == []
        assert ctx.filming_locations == ["Dubrovnik", "Northern Ireland"]

    def test_places_are_trimmed_deduplicated_and_capped(self):
        ctx = ScreenContext(
            story_locations=[" Paris ", "paris", "", "Rome"],
            filming_locations=[f"Place {i}" for i in range(MAX_SCREEN_LOCATIONS + 5)],
        )
        assert ctx.story_locations == ["Paris", "Rome"]
        assert len(ctx.filming_locations) == MAX_SCREEN_LOCATIONS

    def test_the_source_work_is_a_book_or_nothing(self):
        book = WorkRef(kind="book", title="Normal People", creator="Sally Rooney")
        film = WorkRef(kind="film", title="Mamma Mia!", creator="Phyllida Lloyd", year=2008)
        assert ScreenContext(source_work=book).source_work == book
        assert ScreenContext(source_work=film).source_work is None

    def test_round_trips_as_json(self):
        ctx = ScreenContext(
            story_locations=["Rome", "Venice"],
            filming_locations=["Ischia", "Procida"],
            source_work={"title": "The Talented Mr. Ripley", "creator": "Patricia Highsmith"},
        )
        assert ScreenContext.model_validate_json(ctx.model_dump_json()) == ctx
        assert ctx.source_work.kind == "book"

    def test_a_non_list_is_rejected(self):
        with pytest.raises(ValidationError):
            ScreenContext(story_locations="Paris")


# ── the guardrail itself ─────────────────────────────────────────────────────


class TestDowngradeUnsupportedSetHere:
    @pytest.mark.parametrize("match_type", ["literal", "historical"])
    def test_a_strong_set_here_without_a_source_becomes_filmed_here_vibe(self, match_type):
        it = _itinerary(_stop("Café de Flore", "set_here", match_type, scene="Emily's breakfast"))
        out = downgrade_unsupported_set_here_claims(it, "series")
        assert _claims(out) == [("Café de Flore", "filmed_here", "vibe", None)]
        # the scene line survives: it is what the reader sees under "Filmed here"
        assert out["cities"][0]["stops"][0]["filming_scene"] == "Emily's breakfast"

    @pytest.mark.parametrize("source", ["", "   "])
    def test_a_blank_source_is_no_source(self, source):
        it = _itinerary(_stop("Café de Flore", "set_here", "literal", source=source))
        assert _claims(downgrade_unsupported_set_here_claims(it, "film"))[0][1:3] == ("filmed_here", "vibe")

    @pytest.mark.parametrize("match_type", ["literal", "historical"])
    def test_a_strong_set_here_with_a_source_stays(self, match_type):
        it = _itinerary(_stop("Place de l'Estrapade", "set_here", match_type, source="Emily's flat, S1E1"))
        assert _claims(downgrade_unsupported_set_here_claims(it, "series")) == [
            ("Place de l'Estrapade", "set_here", match_type, "Emily's flat, S1E1")
        ]

    def test_filmed_here_and_unclaimed_stops_are_left_alone(self):
        stops = [
            _stop("Fort Lovrijenac", "filmed_here", "literal"),
            _stop("Lokrum", None, "literal"),
        ]
        it = _itinerary(*stops, city="Dubrovnik", country="Croatia")
        before = _claims(it)
        assert _claims(downgrade_unsupported_set_here_claims(it, "series")) == before

    @pytest.mark.parametrize("match_type", ["vibe", "thematic", None])
    @pytest.mark.parametrize("source", [None, "Castello Aragonese, Ischia"])
    def test_a_set_here_on_a_weak_or_missing_label_is_demoted_even_with_a_source(self, match_type, source):
        # ai#311 review probe: Castello Aragonese, set_here + vibe, no source came out as set_here.
        it = _itinerary(_stop("Castello Aragonese", "set_here", match_type, source=source), city="Ischia")
        assert _claims(downgrade_unsupported_set_here_claims(it, "film")) == [
            ("Castello Aragonese", "filmed_here", "vibe", None)
        ]

    def test_a_book_journey_is_untouched(self):
        it = _itinerary(_stop("Café de Flore", "set_here", "literal"))
        assert _claims(downgrade_unsupported_set_here_claims(it, "book"))[0][1] == "set_here"

    def test_expansion_places_are_guarded_too(self):
        expansion = {"places": [_stop("Ischia Ponte", "set_here", "historical")]}
        out = downgrade_unsupported_set_here_claims(expansion, "film", stops_path="expansion")
        assert out["places"][0]["screen_claim"] == "filmed_here"

    @pytest.mark.parametrize("holder", [None, {}, {"cities": None}, {"cities": ["x", {"stops": ["y"]}]}])
    def test_odd_shapes_never_raise(self, holder):
        assert downgrade_unsupported_set_here_claims(holder, "film") == holder


class TestLostGrounding:
    def test_a_set_here_whose_evidence_was_rejected_is_demoted(self):
        rome = _stop("Piazza di Spagna", "set_here", "literal", source="opening scene")
        ischia = _stop("Ischia Ponte", "set_here", "historical", source="Mongibello")
        it = _itinerary(rome, ischia, city="Rome", country="Italy")
        strong = strong_set_here_stops(it, "film")
        assert strong == [rome, ischia]
        # what downgrade_ungrounded_match_types does to a stop it can't ground
        ischia["match_type"], ischia["grounding_source"] = "vibe", None
        assert demote_set_here_that_lost_grounding(strong) == 1
        assert _claims(it) == [
            ("Piazza di Spagna", "set_here", "literal", "opening scene"),
            ("Ischia Ponte", "filmed_here", "vibe", None),
        ]

    def test_nothing_is_tracked_on_a_book_journey(self):
        assert strong_set_here_stops(_itinerary(_stop("x", "set_here", "literal", "s")), "book") == []


# ── wired into the real extraction path ──────────────────────────────────────


def _accessor(work_kind, itinerary, grounding_text="", expansion=None):
    return SimpleNamespace(
        grounding_research_text=grounding_text,
        all_discovery_unverified=False,
        work_kind=work_kind,
        composer_envelope={"itinerary": itinerary, "suggestions": []},
        final_itinerary=None,
        last_expansion=expansion,
    )


class TestExtractionPath:
    def test_the_composed_itinerary_leaves_with_the_weaker_claim(self):
        it = _itinerary(
            _stop("Café de Flore", "set_here", "literal"),  # no source → downgraded
            _stop("Place de l'Estrapade", "set_here", "literal", source="Emily's flat"),
        )
        itinerary, _ = extract_itinerary_from_response(
            None, _accessor("series", it, grounding_text="Place de l'Estrapade Paris")
        )
        assert _claims(itinerary) == [
            ("Café de Flore", "filmed_here", "vibe", None),
            ("Place de l'Estrapade", "set_here", "literal", "Emily's flat"),
        ]

    def test_a_sourceless_set_here_is_demoted_even_when_the_research_names_it(self):
        """Only the source rule catches this one: the research found the café, so the
        book-path downgrade keeps it literal, but a set_here with no source is still
        an unsupported claim."""
        it = _itinerary(_stop("Café de Flore", "set_here", "literal"))
        itinerary, _ = extract_itinerary_from_response(
            None, _accessor("series", it, grounding_text="Café de Flore Saint-Germain Paris")
        )
        assert _claims(itinerary) == [("Café de Flore", "filmed_here", "vibe", None)]

    def test_a_cited_set_here_the_research_never_found_is_demoted(self):
        it = _itinerary(
            _stop("Ischia Ponte", "set_here", "literal", source="Mongibello harbour"),
            _stop("Piazza di Spagna", "set_here", "literal", source="opening"),
            city="Rome",
            country="Italy",
        )
        itinerary, _ = extract_itinerary_from_response(
            None, _accessor("film", it, grounding_text="Piazza di Spagna Rome Venice")
        )
        assert _claims(itinerary) == [
            ("Ischia Ponte", "filmed_here", "vibe", None),
            ("Piazza di Spagna", "set_here", "literal", "opening"),
        ]

    def test_a_set_here_typed_vibe_leaves_as_filmed_here(self):
        """ai#311 review probe: set_here + vibe, no source, used to ship as "Set here"."""
        it = _itinerary(_stop("Castello Aragonese", "set_here", "vibe"), city="Ischia", country="Italy")
        itinerary, _ = extract_itinerary_from_response(
            None, _accessor("film", it, grounding_text="Castello Aragonese Ischia")
        )
        assert _claims(itinerary) == [("Castello Aragonese", "filmed_here", "vibe", None)]

    def test_a_book_journey_still_ships_no_screen_claim(self):
        it = _itinerary(_stop("Pump Room", "set_here", "literal", source="ch. 3"), city="Bath", country="UK")
        itinerary, _ = extract_itinerary_from_response(None, _accessor("book", it, grounding_text="Pump Room Bath"))
        assert itinerary["cities"][0]["stops"][0]["screen_claim"] is None
        assert itinerary["cities"][0]["stops"][0]["match_type"] == "literal"

    def test_an_expansion_is_guarded(self):
        expansion = {
            "parent_city": "Naples",
            "places": [_stop("Procida harbour", "set_here", "historical")],
            "suggestions": [],
        }
        out = extract_expansion_from_state(_accessor("film", _itinerary(), expansion=expansion))
        assert out["places"][0]["screen_claim"] == "filmed_here"
        assert out["places"][0]["match_type"] == "vibe"
