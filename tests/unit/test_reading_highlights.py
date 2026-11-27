"""core/reading_highlights.py: known names bolded by code, one model fact kept."""

from core.reading_highlights import (
    apply_reading_highlights,
    known_names,
    names_from_state,
    pick_highlights,
)
from core.session_state import SessionStateAccessor

NAMES = known_names("The Goldfinch", "Donna Tartt", ["Theo Decker", "Boris Pavlikovsky", "Hobie"])


def test_known_names_adds_surname_and_first_last_name_parts():
    for name in ("The Goldfinch", "Donna Tartt", "Tartt", "Theo Decker", "Theo", "Decker", "Boris", "Hobie"):
        assert name in NAMES, name
    # Too short or not a name: dropped.
    assert known_names("It", None, ["  ", 42, "al"]) == []


def test_known_name_bolded_by_whole_word_and_longest_first():
    text = "This is where Theo Decker stays; Theo's friend Boris visits."
    assert pick_highlights(text, [], NAMES) == ["Theo Decker", "Boris"]
    # Not inside another word.
    assert pick_highlights("Theodora sings.", [], ["Theo"]) == []


def test_model_fact_comes_first_and_only_one_is_kept():
    text = "Home of The Goldfinch in Room 14, next to the Vermeer."
    picks = pick_highlights(text, ["Room 14", "Vermeer"], NAMES)
    assert picks == ["The Goldfinch", "Room 14"]


def test_model_pick_without_a_name_or_not_verbatim_is_dropped():
    text = "A central hub of Theo Decker's life."
    assert pick_highlights(text, ["central hub"], NAMES) == ["Theo Decker"]
    assert pick_highlights(text, ["Andy Weir"], NAMES) == ["Theo Decker"]
    assert pick_highlights(text, "Theo", []) == []


def test_at_most_two_and_no_overlap():
    text = "Donna Tartt wrote The Goldfinch about Theo Decker and Boris."
    # The model's pick plus the longest known name found first, in text order.
    assert pick_highlights(text, ["The Goldfinch"], NAMES) == ["Donna Tartt", "The Goldfinch"]


def test_apply_rewrites_every_field_and_tolerates_bad_shapes():
    itinerary = {
        "summary_text": "Follow Theo Decker to The Goldfinch.",
        "summary_highlights": ["moderate pace"],
        "cities": [
            {"overview": "Amsterdam, where Boris returns.", "overview_highlights": None,
             "stops": [{"reason": "Hobie's shop.", "highlights": ["Hobie's shop"]}, "junk"]},
            "junk",
        ],
    }
    out = apply_reading_highlights(itinerary, NAMES)
    assert out["summary_highlights"] == ["Theo Decker", "The Goldfinch"]
    assert out["cities"][0]["overview_highlights"] == ["Boris"]
    assert out["cities"][0]["stops"][0]["highlights"] == ["Hobie's shop"]
    assert apply_reading_highlights(None, NAMES) is None


def test_names_from_state_reads_metadata_and_book_context():
    state = SessionStateAccessor({
        "book_metadata": {"book_title": "The Goldfinch", "author": "Donna Tartt"},
        "book_context": {"primary_locations": [], "themes": [], "main_characters": ["Theo Decker"]},
    })
    names = names_from_state(state)
    assert "Theo Decker" in names and "The Goldfinch" in names and "Tartt" in names
    # Older cached discovery without main_characters still works.
    assert "The Goldfinch" in names_from_state(SessionStateAccessor({"book_metadata": {"book_title": "The Goldfinch"}}))
