"""Unit tests for eval dataset routing.

Pins the fix for the auto-discovery trap: the scheduled-eval CI workflow
regenerates evaluation/langfuse_datasets.json from a glob of ALL
*.evalset.json files, and run_scheduled_eval's auto-discovery used to feed
every registered dataset through the itinerary workflow — place→book cases
would all skip (failing CI) and local-atmosphere/expansion cases would run
through the WRONG flow, producing plausible-but-bogus judge scores silently.
Routing is now explicit via each evalset's top-level ``flow`` field.
"""

import json
from pathlib import Path

from evaluation.tools.langfuse_eval import LangfuseEvalPipeline
from evaluation.tools.run_scheduled_eval import (
    DEDICATED_RUNNERS,
    select_itinerary_datasets,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _registry(entries):
    return {"datasets": entries}


def test_ci_shaped_registry_selects_only_itinerary_datasets():
    # The exact registry the CI sync step produces once all five evalsets
    # exist in evaluation/ — the trap's regression test.
    registry = _registry([
        {"dataset_name": "books_v1", "flow": "itinerary"},
        {"dataset_name": "expansion_v1", "flow": "expansion"},
        {"dataset_name": "local_atmosphere_v1", "flow": "local_atmosphere"},
        {"dataset_name": "place_to_book_v1", "flow": "place_to_book"},
        {"dataset_name": "storyland_eval", "flow": "itinerary"},
    ])
    selected, routed = select_itinerary_datasets(registry)
    assert selected == ["books_v1", "storyland_eval"]
    assert {r["dataset_name"] for r in routed} == {
        "expansion_v1", "local_atmosphere_v1", "place_to_book_v1"
    }
    # Every routed flow names a dedicated runner for the log message.
    for r in routed:
        assert r["flow"] in DEDICATED_RUNNERS


def test_missing_flow_defaults_to_itinerary():
    # Registries written before the flow field existed contain only
    # itinerary datasets — they must keep working unchanged.
    selected, routed = select_itinerary_datasets(_registry([
        {"dataset_name": "books_v1"},
        {"dataset_name": "storyland_eval"},
    ]))
    assert selected == ["books_v1", "storyland_eval"]
    assert routed == []


def test_empty_registry_yields_empty_split():
    assert select_itinerary_datasets(_registry([])) == ([], [])
    assert select_itinerary_datasets({}) == ([], [])


def test_all_committed_evalsets_declare_expected_flows():
    """Every non-itinerary evalset in evaluation/ must carry its flow field —
    a new evalset without one silently lands in the scheduled itinerary run."""
    expected = {
        "place_to_book_v1": "place_to_book",
        "local_atmosphere_v1": "local_atmosphere",
        "expansion_v1": "expansion",
    }
    for path in (REPO_ROOT / "evaluation").glob("*.evalset.json"):
        evalset = json.loads(path.read_text())
        eval_set_id = evalset.get("eval_set_id")
        if eval_set_id in expected:
            assert evalset.get("flow") == expected[eval_set_id], path.name
        else:
            assert evalset.get("flow", "itinerary") == "itinerary", path.name


# --- input extraction shapes (dataset sync) --------------------------------

def test_extract_input_place_shape():
    case = {"place": "Lisbon", "eval_id": "x"}
    assert LangfuseEvalPipeline._extract_input_from_case(case) == {"place": "Lisbon"}


def test_extract_input_local_atmosphere_shape():
    case = {
        "book_title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "location_label": "New York, NY",
        "lat": 40.7128,
        "lng": -74.006,
        "radius_km": 15,
    }
    assert LangfuseEvalPipeline._extract_input_from_case(case) == {
        "book_title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "location_label": "New York, NY",
        "lat": 40.7128,
        "lng": -74.006,
        "radius_km": 15,
    }


def test_extract_input_local_atmosphere_radius_defaults_to_80():
    case = {"book_title": "T", "author": "A", "location_label": "X", "lat": 1, "lng": 2}
    assert LangfuseEvalPipeline._extract_input_from_case(case)["radius_km"] == 80


def test_extract_input_itinerary_and_expansion_share_book_shape():
    case = {"book_title": "The Hobbit", "author": "J.R.R. Tolkien",
            "chip_keyword": "tea"}
    assert LangfuseEvalPipeline._extract_input_from_case(case) == {
        "book_title": "The Hobbit", "author": "J.R.R. Tolkien"
    }


def test_extract_input_place_takes_precedence_over_location_label():
    case = {"place": "Paris", "location_label": "Paris, France", "lat": 1, "lng": 2}
    assert LangfuseEvalPipeline._extract_input_from_case(case) == {"place": "Paris"}
