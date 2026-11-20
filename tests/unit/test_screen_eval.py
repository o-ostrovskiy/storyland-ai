"""ai-private#68: the screen-path eval set and its two scorers, offline.

Hand-written discovery outputs (no live run): a set_here on a filming-only place
lowers claim precision; an invented book drops book-pairing honesty below 1.0;
an empty discovery scores 0, not "pass"; the evalset loads and every title
carries its expected fields; the runner's release gate and the trend-report
columns read the same numbers.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from evaluation.tools.eval_dashboard import build_screen_section
from evaluation.tools.langfuse_eval import LangfuseEvalPipeline
from evaluation.tools.release_gate import verdict
from evaluation.tools.run_screen_eval import (
    HARD_METRICS,
    METRIC_FLOORS,
    MIN_PASS_RATE,
    load_cases,
    main as run_screen_main,
    score_outputs,
)
from evaluation.tools.screen_scorers import (
    aggregate,
    book_pairing_honesty,
    claim_precision,
    extract_stops,
    place_matches,
    score_case,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALSET = REPO_ROOT / "evaluation" / "screen_v1.evalset.json"


def _out(cities, source_work=None):
    """A discovery output: cities = [(city, country, [(stop, claim), ...]), ...]."""
    return {
        "itinerary": {
            "cities": [
                {"name": c, "country": k, "stops": [{"name": s, "screen_claim": claim} for s, claim in stops]}
                for c, k, stops in cities
            ]
        },
        "screen_context": {"source_work": source_work},
    }


GOT = {"title": "A Song of Ice and Fire", "author": "George R. R. Martin", "also_titles": ["A Game of Thrones"]}


def _expected(eval_id):
    return next(c for c in load_cases(str(EVALSET)) if c["eval_id"] == eval_id)["expected_output"]


# ── the evalset itself ───────────────────────────────────────────────────────


def test_the_evalset_has_twelve_titles_covering_the_claim_matrix():
    evalset = json.loads(EVALSET.read_text())
    assert evalset["eval_set_id"] == "screen_v1"
    assert evalset["flow"] == "screen"
    cases = evalset["eval_cases"]
    assert len(cases) == 12
    assert len({c["eval_id"] for c in cases}) == 12
    exp = {c["eval_id"]: c["expected_output"] for c in cases}
    # fictional worlds carry no story location
    assert exp["screen_game_of_thrones"]["story_locations"] == []
    assert exp["screen_lord_of_the_rings"]["story_locations"] == []
    # series with a source book / film with no book
    assert exp["screen_normal_people"]["source_work"]["author"] == "Sally Rooney"
    assert exp["screen_before_sunrise"]["source_work"] is None
    assert exp["screen_bad_input"]["expect"] == "not_found"


@pytest.mark.parametrize("case", json.loads(EVALSET.read_text())["eval_cases"], ids=lambda c: c["eval_id"])
def test_every_title_has_its_expected_fields(case):
    assert case["screen_title"].strip()
    assert case["screen_kind"] in ("film", "series")
    exp = case["expected_output"]
    assert exp["expect"] in ("found", "not_found")
    for key in ("story_locations", "filming_locations", "required_claims"):
        assert isinstance(exp[key], list), key
    assert "source_work" in exp and "is_fictional_world" in exp
    if exp["expect"] == "found":
        assert exp["filming_locations"], "a real title was filmed somewhere"
        assert 1 <= len(exp["required_claims"]) <= 4
    for claim in exp["required_claims"]:
        assert claim["screen_claim"] in ("filmed_here", "set_here")
        # a required set_here must be a real setting of the work, or the set contradicts itself
        if claim["screen_claim"] == "set_here":
            assert any(place_matches(claim["place"], s) for s in exp["story_locations"]), claim
    if exp["is_fictional_world"]:
        assert exp["story_locations"] == []


def test_the_dataset_sync_stores_the_screen_input_shape():
    case = load_cases(str(EVALSET))[0]
    assert LangfuseEvalPipeline._extract_input_from_case(case) == {
        "title": "Emily in Paris", "kind": "series", "year": 2020,
    }


# ── place matching ───────────────────────────────────────────────────────────


def test_place_matching_is_whole_words_accent_and_case_free():
    assert place_matches("Café des Deux Moulins, Montmartre, Paris, France", "paris")
    assert place_matches("Belfast, Northern Ireland, United Kingdom", "Northern Ireland")
    assert place_matches("Amélie's café, Montmartre", "Amelie")
    assert not place_matches("Romeo's Diner, Verona", "Rome")
    assert not place_matches("Ireland", "Northern Ireland")
    assert not place_matches("Paris", "")


def test_stops_are_read_from_an_itinerary_or_a_composer_envelope():
    out = _out([("Dubrovnik", "Croatia", [("Fort Lovrijenac", "filmed_here")])])
    envelope = {"itinerary": {"itinerary": out["itinerary"], "suggestions": []}}
    for shape in (out, envelope):
        assert extract_stops(shape) == [
            {"place": "Fort Lovrijenac, Dubrovnik, Croatia", "screen_claim": "filmed_here"}
        ]


# ── claim precision ──────────────────────────────────────────────────────────


def test_a_set_here_on_a_filming_only_place_lowers_claim_precision():
    story = _expected("screen_talented_mr_ripley")["story_locations"]
    honest = extract_stops(_out([
        ("Rome", "Italy", [("Piazza di Spagna", "set_here")]),
        ("Ischia", "Italy", [("Ischia Ponte", "filmed_here")]),
    ]))
    conflated = extract_stops(_out([
        ("Rome", "Italy", [("Piazza di Spagna", "set_here")]),
        ("Ischia", "Italy", [("Ischia Ponte", "set_here")]),  # Mongibello is invented
    ]))
    assert claim_precision(honest, story) == 1.0
    assert claim_precision(conflated, story) == 0.5


def test_any_set_here_in_a_fictional_world_is_a_false_claim():
    assert claim_precision(extract_stops(_out([("Dubrovnik", "Croatia", [("Old Town", "set_here")])])), []) == 0.0


def test_a_journey_with_no_set_here_claim_makes_no_false_claim():
    stops = extract_stops(_out([("Dubrovnik", "Croatia", [("Old Town", "filmed_here")])]))
    assert claim_precision(stops, []) == 1.0


# ── book-pairing honesty ─────────────────────────────────────────────────────


def test_an_invented_book_drops_book_pairing_honesty():
    assert book_pairing_honesty({"title": "Before Sunrise", "author": "Richard Linklater"}, None) == 0.0
    assert book_pairing_honesty({"title": "Normal Lives", "author": "Sally Rooney"},
                                {"title": "Normal People", "author": "Sally Rooney"}) == 0.0
    assert book_pairing_honesty({"title": "Normal People", "author": "Someone Else"},
                                {"title": "Normal People", "author": "Sally Rooney"}) == 0.0


def test_the_right_book_or_no_book_is_honest():
    assert book_pairing_honesty({"title": "Normal People", "author": "Sally Rooney"},
                                {"title": "Normal People", "author": "Sally Rooney"}) == 1.0
    assert book_pairing_honesty({"title": "Game of Thrones", "author": "George R.R. Martin"}, GOT) == 1.0
    assert book_pairing_honesty(None, GOT) == 1.0  # a gap, not a lie
    assert book_pairing_honesty(None, None) == 1.0
    assert book_pairing_honesty({"title": "  "}, None) == 1.0


# ── whole cases ──────────────────────────────────────────────────────────────


def test_an_empty_discovery_scores_zero_not_pass():
    scored = score_case({"itinerary": {"cities": []}, "screen_context": None}, _expected("screen_in_bruges"))
    assert scored["pass"] is False
    assert scored["claim_precision"] == 0.0
    assert scored["book_pairing_honesty"] == 0.0
    assert score_case(None, _expected("screen_in_bruges"))["pass"] is False


def test_game_of_thrones_done_right_passes():
    out = _out([
        ("Dubrovnik", "Croatia", [("Fort Lovrijenac", "filmed_here"), ("Jesuit Stairs", "filmed_here")]),
        ("Belfast", "Northern Ireland", [("Dark Hedges", "filmed_here")]),
    ], source_work={"title": "A Game of Thrones", "author": "George R. R. Martin"})
    scored = score_case(out, _expected("screen_game_of_thrones"))
    assert scored["pass"] is True
    assert (scored["claim_precision"], scored["book_pairing_honesty"], scored["named_claims"]) == (1.0, 1.0, 1.0)


def test_a_named_stop_with_the_wrong_claim_or_missing_fails_the_case():
    exp = _expected("screen_game_of_thrones")
    wrong = _out([("Dubrovnik", "Croatia", [("Old Town", "set_here")]),
                  ("Belfast", "Northern Ireland", [("Dark Hedges", "filmed_here")])])
    missing = _out([("Dubrovnik", "Croatia", [("Old Town", "filmed_here")])])
    assert score_case(wrong, exp)["named_claims"] == 0.5
    assert score_case(missing, exp)["named_claims"] == 0.5
    assert not score_case(missing, exp)["pass"]


def test_the_bad_input_passes_only_when_nothing_is_composed_or_paired():
    exp = _expected("screen_bad_input")
    assert score_case({"itinerary": {"cities": []}, "screen_context": None}, exp)["pass"] is True
    invented = score_case(_out([("Paris", "France", [("Louvre", "filmed_here")])]), exp)
    assert invented["pass"] is False and invented["book_pairing_honesty"] == 0.0
    assert score_case(_out([], source_work={"title": "Xyzzy", "author": "Nobody"}), exp)["pass"] is False


def test_aggregate_skips_not_applicable_metrics():
    cases = [
        {"claim_precision": 1.0, "book_pairing_honesty": 1.0, "named_claims": 1.0},
        {"claim_precision": 0.5, "book_pairing_honesty": 0.0, "named_claims": 0.5},
        {"claim_precision": None, "book_pairing_honesty": 1.0, "named_claims": None},
    ]
    assert aggregate(cases) == {"claim_precision": 0.75, "book_pairing_honesty": 0.6667, "named_claims": 0.75}


# ── the runner + release gate ────────────────────────────────────────────────


def _perfect_outputs():
    """An output per case that satisfies its own expectations (built from the evalset)."""
    outputs = {}
    for case in load_cases(str(EVALSET)):
        exp = case["expected_output"]
        if exp["expect"] == "not_found":
            outputs[case["eval_id"]] = {"itinerary": {"cities": []}, "screen_context": None}
            continue
        stops = [(c["place"], c["screen_claim"]) for c in exp["required_claims"]]
        outputs[case["eval_id"]] = _out([("", "", stops)], source_work=exp["source_work"])
    return outputs


def test_a_clean_run_is_allowed_by_the_gate():
    summary = score_outputs(load_cases(str(EVALSET)), _perfect_outputs(), run_name="t")
    assert summary["evaluated"] == summary["total"] == summary["passed"] == 12
    assert summary["metrics"]["claim_precision"] == 1.0
    assert verdict(summary, MIN_PASS_RATE, HARD_METRICS, metric_floors=METRIC_FLOORS) == (True, [])


def test_an_invented_book_blocks_the_release():
    outputs = _perfect_outputs()
    outputs["screen_before_sunrise"]["screen_context"]["source_work"] = {"title": "Before Sunrise", "author": "R. L."}
    summary = score_outputs(load_cases(str(EVALSET)), outputs, run_name="t")
    allowed, reasons = verdict(summary, MIN_PASS_RATE, HARD_METRICS, metric_floors=METRIC_FLOORS)
    assert not allowed
    assert any("book_pairing_honesty" in r for r in reasons)


def test_claim_precision_under_the_floor_blocks_the_release():
    outputs = _perfect_outputs()
    for eval_id in ("screen_game_of_thrones", "screen_lord_of_the_rings"):
        for stop in outputs[eval_id]["itinerary"]["cities"][0]["stops"]:
            stop["screen_claim"] = "set_here"  # a filming location sold as the setting
    summary = score_outputs(load_cases(str(EVALSET)), outputs, run_name="t")
    assert summary["metrics"]["claim_precision"] < 0.9
    allowed, reasons = verdict(summary, MIN_PASS_RATE, HARD_METRICS, metric_floors=METRIC_FLOORS)
    assert not allowed
    assert any("claim_precision" in r for r in reasons)


def test_a_missing_output_is_unscored_and_blocks():
    outputs = _perfect_outputs()
    del outputs["screen_amelie"]
    summary = score_outputs(load_cases(str(EVALSET)), outputs, run_name="t")
    assert summary["evaluated"] == 11
    assert not verdict(summary, MIN_PASS_RATE, HARD_METRICS, metric_floors=METRIC_FLOORS)[0]


def test_the_cli_writes_results_and_exits_non_zero_when_blocked(tmp_path, capsys):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(_perfect_outputs()))
    bad.write_text(json.dumps({}))
    assert run_screen_main(["--outputs", str(good), "--evalset", str(EVALSET), "--output-dir", str(tmp_path)]) == 0
    assert run_screen_main(["--outputs", str(bad), "--evalset", str(EVALSET), "--output-dir", str(tmp_path)]) == 1
    assert "RELEASE: BLOCKED" in capsys.readouterr().out
    written = sorted(tmp_path.glob("screen_*.json"))
    assert written and "release_gate" in json.loads(written[-1].read_text())


# ── trend report columns ─────────────────────────────────────────────────────


def test_the_trend_report_gets_the_two_screen_columns(tmp_path):
    assert build_screen_section(str(tmp_path)) == []
    (tmp_path / "screen_20261120_000000.json").write_text(json.dumps({
        "run_name": "screen_eval_x", "timestamp": datetime.now().isoformat(), "total": 12, "evaluated": 12,
        "metrics": {"claim_precision": 0.9167, "book_pairing_honesty": 1.0},
        "release_gate": {"allowed": True},
    }))
    lines = build_screen_section(str(tmp_path))
    assert "| Date | Run | Cases | Claim precision | Book-pairing honesty | Release |" in lines
    assert lines[-1].endswith("| screen_eval_x | 12/12 | 0.92 | 1.00 | ✅ allowed |")
