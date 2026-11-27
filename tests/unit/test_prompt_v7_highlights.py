"""Prompt v7: the trip composer emits reading highlights (2026-11-25)."""

import json
from pathlib import Path

from agents.prompts import CURRENT_PROMPT_VERSION, load_prompts
from models.itinerary import CityPlan, CityStop, TripItinerary


def test_v7_is_v6_plus_the_composer_and_book_context_changes_only():
    v6 = json.loads(Path("agents/prompts/v6.json").read_text(encoding="utf-8"))
    v7 = json.loads(Path("agents/prompts/v7.json").read_text(encoding="utf-8"))
    assert v6["agents"].keys() == v7["agents"].keys()
    changed = sorted(a for a in v6["agents"] if v6["agents"][a] != v7["agents"][a])
    assert changed == ["book_context_formatter", "book_context_researcher", "trip_composer"], (
        "v7 must not quietly re-tune any other agent"
    )


def test_v7_is_current_and_asks_for_every_highlight_field():
    assert CURRENT_PROMPT_VERSION == "v7"
    composer = load_prompts("v7").trip_composer
    for field in ("headline", "summary_highlights", "overview_highlights", "book_hook", "highlights"):
        assert field in composer, field
    assert "letter for letter" in composer
    assert "at most 1 per field" in composer
    prompts = load_prompts("v7")
    assert "MAIN CHARACTERS" in prompts.book_context_researcher
    assert "main_characters" in prompts.book_context_formatter
    # The researcher template still formats with its two placeholders.
    prompts.book_context_researcher.format(book_ref="BOOK: x", search_hint="")


def test_highlight_fields_are_optional_so_older_payloads_still_validate():
    stop = CityStop(name="Ambassade Hotel", type="hotel", reason="Theo stays here.", time_of_day="morning")
    assert stop.book_hook is None and stop.highlights == []
    city = CityPlan(name="Amsterdam", country="Netherlands", days_suggested=2, overview="Canals.", stops=[stop])
    assert city.overview_highlights == []
    trip = TripItinerary(cities=[city], summary_text="A trip.")
    assert trip.headline is None and trip.summary_highlights == []


def test_highlight_fields_round_trip():
    stop = CityStop(
        name="Mauritshuis", type="museum", time_of_day="morning",
        reason="Home of The Goldfinch by Carel Fabritius.",
        book_hook="The real painting the novel is named after.",
        highlights=["The Goldfinch by Carel Fabritius"],
    )
    dumped = stop.model_dump()
    assert dumped["book_hook"] == "The real painting the novel is named after."
    assert dumped["highlights"] == ["The Goldfinch by Carel Fabritius"]


# ── Eval: deterministic reading-highlight checks ────────────────────────────

from evaluation.tools.run_scheduled_eval import (  # noqa: E402
    aggregate_reading_highlights,
    summarize_reading_highlights,
)

_ITINERARY = {
    "summary_text": "Follow The Goldfinch to the Mauritshuis.",
    "summary_highlights": ["The Goldfinch"],
    "headline": "From Amsterdam's canals to the painting itself.",
    "cities": [
        {
            "overview": "Amsterdam reflects Theo's isolation.",
            "overview_highlights": ["Theo's isolation", "not in the text", "isolation"],
            "stops": [
                {"reason": "Theo Decker stays here.", "highlights": ["Theo Decker"], "book_hook": "Where Theo stays."},
                {"reason": "A museum.", "highlights": "Theo"},
                {"reason": "A cafe."},
            ],
        }
    ],
}


def test_summarize_reading_highlights_counts_verbatim_picks_cap_and_hooks():
    out = summarize_reading_highlights(_ITINERARY)
    # 1 summary + 3 overview + 1 stop; the bare-string list is not a list and is skipped.
    assert out["phrases_picked"] == 5
    assert out["phrases_found_verbatim"] == 4
    # "Theo's isolation", "The Goldfinch", "Theo Decker" hold a name; "isolation" does not.
    assert out["phrases_shown"] == 3
    assert out["fields_over_cap"] == 1
    assert out["stops_total"] == 3 and out["stops_with_book_hook"] == 1
    assert out["has_headline"] is True


def test_summarize_reading_highlights_on_a_legacy_itinerary_and_no_itinerary():
    legacy = {"summary_text": "x", "cities": [{"overview": "y", "stops": [{"reason": "z"}]}]}
    out = summarize_reading_highlights(legacy)
    assert out["phrases_picked"] == 0 and out["has_headline"] is False
    assert summarize_reading_highlights(None) is None


def test_aggregate_reading_highlights_rate_is_none_when_nothing_was_picked():
    legacy = summarize_reading_highlights({"summary_text": "x", "cities": []})
    agg = aggregate_reading_highlights([{"reading_highlights": legacy}])
    assert agg["verbatim_rate"] is None
    agg = aggregate_reading_highlights([{"reading_highlights": summarize_reading_highlights(_ITINERARY)}, {}])
    assert agg["cases"] == 1 and agg["verbatim_rate"] == 0.8
    assert aggregate_reading_highlights([{}]) is None


def test_summarize_reading_highlights_unwraps_the_composer_envelope():
    # The runner saves {"itinerary": {...}, "suggestions": [...]}; the first live
    # v7 run read 0 stops because the counters looked at the wrapper.
    wrapped = {"itinerary": _ITINERARY, "suggestions": []}
    assert summarize_reading_highlights(wrapped) == summarize_reading_highlights(_ITINERARY)
    assert summarize_reading_highlights(wrapped)["stops_total"] == 3


def test_phrase_has_name():
    from evaluation.tools.run_scheduled_eval import phrase_has_name
    assert phrase_has_name("Mikael Blomkvist") and phrase_has_name("Room 14")
    assert not phrase_has_name("central hub") and not phrase_has_name("visual splendor")


def test_v7_composer_keeps_v6_non_urban_block_verbatim():
    # ai#326 review (Codex P2): v7 had dropped v6's trail/wilderness/road-trip
    # instructions, so those books would get a forced city itinerary.
    v6 = json.loads(Path("agents/prompts/v6.json").read_text(encoding="utf-8"))["agents"]["trip_composer"]
    start = v6.index("**Non-urban settings")
    block = v6[start : v6.index("\n\n", start + len("**Non-urban settings (trails, wilderness, road trips):**\n"))]
    assert "scenic_stop" in block and "route_point" in block
    assert block in load_prompts("v7").trip_composer
