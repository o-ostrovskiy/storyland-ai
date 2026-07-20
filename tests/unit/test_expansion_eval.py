"""Unit tests for the expansion eval (run_expansion_eval pure logic).

Pins the deterministic gate the runner applies to ExpansionReady payloads —
the executor contract: source stamping, dedupe vs the base itinerary,
CityStop schema validity, and chip-id stamping — plus chip selection and a
lint of the evalset file.
"""

import json
from pathlib import Path

import pytest

from evaluation.tools.run_expansion_eval import (
    build_mechanism_section,
    check_expansion_case,
    collect_stop_names,
    pick_chip,
)

EVALSET_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "expansion_v1.evalset.json"


def _base_itinerary():
    return {
        "cities": [
            {
                "name": "London",
                "country": "England",
                "days_suggested": 2,
                "overview": "Victorian London",
                "stops": [
                    {"name": "Sherlock Holmes Museum", "type": "museum",
                     "reason": "221B Baker Street", "time_of_day": "morning"},
                    {"name": "The Criterion", "type": "restaurant",
                     "reason": "Where Watson hears of Holmes", "time_of_day": "evening"},
                ],
            },
            {
                "name": "Edinburgh",
                "country": "Scotland",
                "days_suggested": 1,
                "overview": "Doyle's birthplace",
                "stops": [
                    {"name": "Conan Doyle Pub", "type": "cafe",
                     "reason": "Named for the author", "time_of_day": "evening"},
                ],
            },
        ],
        "summary_text": "A Holmes pilgrimage.",
    }


def _new_place(name="Bart's Hospital", source="expansion", **overrides):
    place = {
        "name": name,
        "type": "landmark",
        "reason": "Where Holmes and Watson first meet",
        "time_of_day": "morning",
        "source": source,
    }
    place.update(overrides)
    return place


def _chip(cid="chip-1", label="Add museums", prompt="Find more museums in London"):
    return {"id": cid, "label": label, "action_prompt": prompt}


# --- pick_chip -------------------------------------------------------------

def test_pick_chip_matches_label_case_insensitively():
    chips = [
        _chip("a", "Add cafés", "Find more cafés in London"),
        _chip("b", "More MUSEUMS nearby", "Find galleries in London"),
    ]
    assert pick_chip(chips, "museum")["id"] == "b"


def test_pick_chip_matches_action_prompt():
    chips = [_chip("a", "Nearby gems", "Find hidden tea rooms"), _chip("b", "Other")]
    assert pick_chip(chips, "tea")["id"] == "a"


def test_pick_chip_falls_back_to_first_on_miss_or_none():
    chips = [_chip("a"), _chip("b")]
    assert pick_chip(chips, "zzz-no-match")["id"] == "a"
    assert pick_chip(chips, None)["id"] == "a"


def test_pick_chip_empty_list_returns_none():
    assert pick_chip([], "tea") is None


# --- collect_stop_names ----------------------------------------------------

def test_collect_stop_names_lowercased_across_cities():
    names = collect_stop_names(_base_itinerary())
    assert names == {"sherlock holmes museum", "the criterion", "conan doyle pub"}


# --- check_expansion_case --------------------------------------------------

def _expansion(places=None, suggestions=None, parent_city="London"):
    return {
        "parent_city": parent_city,
        "places": places if places is not None else [_new_place()],
        "suggestions": suggestions if suggestions is not None else [_chip()],
    }


def test_happy_path_passes():
    checks = check_expansion_case(_base_itinerary(), _expansion())
    assert checks["pass"] is True
    assert checks["no_duplicates"] is True
    assert checks["source_stamped"] is True
    assert checks["places_schema_valid"] is True
    assert checks["chip_ids_stamped"] is True
    assert checks["parent_city_known"] is True


def test_duplicate_against_base_fails_case_insensitively():
    checks = check_expansion_case(
        _base_itinerary(),
        _expansion(places=[_new_place(name="THE CRITERION")]),
    )
    assert checks["no_duplicates"] is False
    assert checks["duplicates"] == ["the criterion"]
    assert checks["pass"] is False


def test_missing_or_wrong_source_fails():
    checks = check_expansion_case(
        _base_itinerary(), _expansion(places=[_new_place(source="composed")])
    )
    assert checks["source_stamped"] is False
    assert checks["pass"] is False

    place = _new_place()
    del place["source"]
    checks = check_expansion_case(_base_itinerary(), _expansion(places=[place]))
    assert checks["source_stamped"] is False


def test_zero_places_fails_min_new_places():
    checks = check_expansion_case(_base_itinerary(), _expansion(places=[]))
    assert checks["n_new_places"] == 0
    assert checks["pass"] is False


def test_invalid_city_stop_schema_fails():
    place = _new_place()
    del place["reason"]  # required CityStop field
    checks = check_expansion_case(_base_itinerary(), _expansion(places=[place]))
    assert checks["places_schema_valid"] is False
    assert checks["schema_errors"]
    assert checks["pass"] is False


def test_empty_chip_id_fails_stamping():
    checks = check_expansion_case(
        _base_itinerary(), _expansion(suggestions=[_chip(cid="")])
    )
    assert checks["chip_ids_stamped"] is False
    assert checks["pass"] is False


def test_duplicate_chip_ids_fail_stamping():
    checks = check_expansion_case(
        _base_itinerary(), _expansion(suggestions=[_chip("x"), _chip("x", "Other")])
    )
    assert checks["chip_ids_stamped"] is False


def test_empty_chip_list_is_vacuously_ok():
    # The soft cap legally empties the chip list — must not fail the gate.
    checks = check_expansion_case(_base_itinerary(), _expansion(suggestions=[]))
    assert checks["chip_ids_stamped"] is True
    assert checks["pass"] is True


def test_more_than_four_chips_breaks_cap():
    chips = [_chip(f"c{i}") for i in range(5)]
    checks = check_expansion_case(_base_itinerary(), _expansion(suggestions=chips))
    assert checks["chips_within_cap"] is False
    assert checks["pass"] is False


def test_unknown_parent_city_is_informational_only():
    checks = check_expansion_case(
        _base_itinerary(), _expansion(parent_city="Atlantis")
    )
    assert checks["parent_city_known"] is False
    assert checks["pass"] is True  # never gates


def test_mechanism_section_declares_deterministic_only():
    mechanism = build_mechanism_section()
    assert mechanism["llm_judge"] is None
    assert "not applicable" in mechanism["preference_shapes"]


# --- evalset lint ----------------------------------------------------------

@pytest.fixture(scope="module")
def evalset():
    return json.loads(EVALSET_PATH.read_text())


def test_evalset_flow_and_size(evalset):
    assert evalset["flow"] == "expansion"
    assert len(evalset["eval_cases"]) == 5


def test_evalset_cases_are_preference_free_first_region(evalset):
    for case in evalset["eval_cases"]:
        state = case.get("session_input", {}).get("state", {})
        assert not state.get("user:preferences"), (
            f"{case['eval_id']}: expansion cases are preference-free by design"
        )
        assert case.get("region_selection") == "first"
        assert case.get("min_new_places", 1) >= 1


def test_evalset_exercises_both_chip_paths(evalset):
    with_keyword = [c for c in evalset["eval_cases"] if c.get("chip_keyword")]
    without_keyword = [c for c in evalset["eval_cases"] if not c.get("chip_keyword")]
    assert with_keyword, "need at least one keyword-matched chip case"
    assert without_keyword, "need at least one first-chip fallback case"
