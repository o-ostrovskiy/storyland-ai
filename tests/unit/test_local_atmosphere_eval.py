"""Unit tests for the local-atmosphere eval (run_local_atmosphere_eval pure logic).

Covers the deterministic gate (envelope validation + opportunistic radius
check), the per-preference-shape aggregation required by the eval protocol,
and a lint of the evalset file itself (shape mix, criterion protocol).
"""

import json
from pathlib import Path

import pytest

from evaluation.tools.llm_scorer import SCORING_CRITERIA
from evaluation.tools.run_local_atmosphere_eval import (
    aggregate_by_preference_shape,
    build_mechanism_section,
    check_local_atmosphere_case,
    check_radius,
    haversine_km,
    scan_geo_points,
    validate_envelope,
)

EVALSET_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "local_atmosphere_v1.evalset.json"


def _valid_itinerary(**overrides):
    itinerary = {
        "cities": [
            {
                "name": "New York",
                "country": "USA",
                "days_suggested": 1,
                "overview": "Jazz-age Manhattan",
                "stops": [
                    {
                        "name": "The Plaza Hotel",
                        "type": "landmark",
                        "reason": "Setting of the climactic suite scene",
                        "time_of_day": "afternoon",
                    }
                ],
            }
        ],
        "summary_text": "A Gatsby afternoon in Manhattan.",
    }
    itinerary.update(overrides)
    return itinerary


def _chips():
    return [{"id": "abc", "label": "Add jazz bars", "action_prompt": "Find jazz bars nearby"}]


# --- haversine -------------------------------------------------------------

def test_haversine_zero_for_identical_points():
    assert haversine_km(40.7128, -74.006, 40.7128, -74.006) == 0.0


def test_haversine_nyc_to_newark_about_14km():
    d = haversine_km(40.7128, -74.006, 40.7357, -74.1724)
    assert 12 < d < 18


# --- scan_geo_points -------------------------------------------------------

def test_scan_geo_points_empty_for_plain_itinerary():
    assert scan_geo_points(_valid_itinerary()) == []


def test_scan_geo_points_finds_both_spellings_nested():
    obj = {
        "cities": [
            {"stops": [{"lat": 1.0, "lng": 2.0}]},
            {"meta": {"latitude": 3.0, "longitude": 4.0}},
        ]
    }
    assert set(scan_geo_points(obj)) == {(1.0, 2.0), (3.0, 4.0)}


def test_scan_geo_points_ignores_half_pairs_and_non_numeric():
    assert scan_geo_points({"lat": 1.0}) == []
    assert scan_geo_points({"lat": "1.0", "lng": "2.0"}) == []
    assert scan_geo_points({"lat": True, "lng": 2.0}) == []


# --- check_radius ----------------------------------------------------------

def test_check_radius_no_geo_fields():
    result = check_radius(_valid_itinerary(), 40.7128, -74.006, 15)
    assert result["radius_check"] == "no_geo_fields"
    assert result["geo_points_found"] == 0


def test_check_radius_pass_within_slack():
    # ~14 km straight-line with radius 10 and slack 1.5 → 15 km allowance.
    itinerary = {"stops": [{"lat": 40.7357, "lng": -74.1724}]}
    result = check_radius(itinerary, 40.7128, -74.006, 10, slack=1.5)
    assert result["radius_check"] == "pass"


def test_check_radius_fail_beyond_slack():
    # Boston is ~300 km from NYC — far outside 15 km × 1.5.
    itinerary = {"stops": [{"lat": 42.3601, "lng": -71.0589}]}
    result = check_radius(itinerary, 40.7128, -74.006, 15)
    assert result["radius_check"] == "fail"
    assert result["outside_radius"] == 1


# --- validate_envelope -----------------------------------------------------

def test_validate_envelope_accepts_valid_payload():
    ok, err = validate_envelope(_valid_itinerary(), _chips())
    assert ok is True
    assert err is None


def test_validate_envelope_rejects_missing_summary():
    itinerary = _valid_itinerary()
    del itinerary["summary_text"]
    ok, err = validate_envelope(itinerary, [])
    assert ok is False
    assert "summary_text" in err


def test_validate_envelope_rejects_bad_match_type():
    itinerary = _valid_itinerary()
    itinerary["cities"][0]["stops"][0]["match_type"] = "invented"
    ok, _ = validate_envelope(itinerary, [])
    assert ok is False


# --- check_local_atmosphere_case ------------------------------------------

def test_case_passes_with_valid_envelope_and_no_geo_fields():
    checks = check_local_atmosphere_case(
        _valid_itinerary(), _chips(), 40.7128, -74.006, 15
    )
    assert checks["deterministic_pass"] is True
    assert checks["envelope_valid"] is True
    assert checks["radius_check"] == "no_geo_fields"
    assert checks["num_cities"] == 1
    assert checks["num_stops"] == 1


def test_case_fails_on_invalid_envelope():
    itinerary = _valid_itinerary()
    del itinerary["summary_text"]
    checks = check_local_atmosphere_case(itinerary, [], 40.7128, -74.006, 15)
    assert checks["deterministic_pass"] is False


def test_case_fails_on_radius_violation():
    itinerary = _valid_itinerary()
    itinerary["cities"][0]["stops"][0]["lat"] = 42.3601
    itinerary["cities"][0]["stops"][0]["lng"] = -71.0589
    checks = check_local_atmosphere_case(itinerary, [], 40.7128, -74.006, 15)
    assert checks["radius_check"] == "fail"
    assert checks["deterministic_pass"] is False


# --- aggregate_by_preference_shape ----------------------------------------

def _scored_case(has_prefs, base=4):
    scores = {
        "book_relevance": base,
        "completeness": base,
        "actionability": base,
        "geographical_accuracy": base,
        "engagement": base,
    }
    if has_prefs:
        scores["preference_adherence"] = 5
    return {"has_preferences": has_prefs, "scores": scores}


def test_aggregation_splits_shapes_and_excludes_preference_adherence():
    agg = aggregate_by_preference_shape(
        [_scored_case(True, 4), _scored_case(True, 2), _scored_case(False, 3)]
    )
    with_prefs = agg["with_preferences"]
    without_prefs = agg["without_preferences"]
    assert with_prefs["n"] == 2
    assert with_prefs["avg_scores"]["preference_adherence"] == 5.0
    assert without_prefs["n"] == 1
    assert "preference_adherence" not in without_prefs["avg_scores"]
    assert without_prefs["average"] == 3.0


def test_aggregation_handles_empty_shape_and_unscored_cases():
    agg = aggregate_by_preference_shape(
        [_scored_case(True), {"has_preferences": False, "scores": None}]
    )
    assert agg["with_preferences"]["n"] == 1
    assert agg["without_preferences"]["n"] == 0
    assert agg["without_preferences"]["average"] is None


def test_mechanism_section_present_and_structured():
    mechanism = build_mechanism_section()
    assert mechanism["deterministic"]
    assert mechanism["llm_judge"]["scorer"].endswith("score_itinerary")
    assert "pass_rule" in mechanism


# --- evalset lint (eval protocol) -----------------------------------------

@pytest.fixture(scope="module")
def evalset():
    return json.loads(EVALSET_PATH.read_text())


def test_evalset_flow_and_size(evalset):
    assert evalset["flow"] == "local_atmosphere"
    assert len(evalset["eval_cases"]) == 8


def test_evalset_cases_carry_required_fields(evalset):
    for case in evalset["eval_cases"]:
        for field in ("eval_id", "book_title", "author", "location_label",
                      "lat", "lng", "radius_km"):
            assert field in case, f"{case.get('eval_id')} missing {field}"
        assert -90 <= case["lat"] <= 90
        assert -180 <= case["lng"] <= 180
        assert 10 <= case["radius_km"] <= 200  # LocalAtmosphereRequest bounds


def test_evalset_has_both_preference_shapes(evalset):
    def has_prefs(case):
        return bool(
            case.get("session_input", {}).get("state", {}).get("user:preferences")
        )

    with_prefs = [c for c in evalset["eval_cases"] if has_prefs(c)]
    without_prefs = [c for c in evalset["eval_cases"] if not has_prefs(c)]
    assert len(with_prefs) >= 3
    assert len(without_prefs) >= 3


def test_preference_free_cases_omit_preference_adherence_criterion(evalset):
    for case in evalset["eval_cases"]:
        has_prefs = bool(
            case.get("session_input", {}).get("state", {}).get("user:preferences")
        )
        criteria = case.get("quality_criteria", {})
        if not has_prefs:
            assert "preference_adherence" not in criteria, (
                f"{case['eval_id']}: preference-free case must not carry a "
                "preference_adherence criterion (eval protocol)"
            )


def test_quality_criteria_keys_are_known_scorer_dimensions(evalset):
    valid = set(SCORING_CRITERIA.keys())
    for case in evalset["eval_cases"]:
        unknown = set(case.get("quality_criteria", {}).keys()) - valid
        assert not unknown, f"{case['eval_id']}: unknown criteria {unknown}"
