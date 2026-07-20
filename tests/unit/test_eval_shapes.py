"""Unit tests for the both-shapes eval machinery (PR-4 step zero).

The books evalset carries two request shapes: preference-carrying cases (the
documented API-contract path) and preference-free cases (the shape 100% of
prod traffic takes since MYS-392 — issue #221). These pin: the 5-dimension
average for unscored preference_adherence, the per-shape aggregation the
gates read, and the evalset's shape split itself.
"""

import json
from pathlib import Path

import pytest

from evaluation.tools.llm_scorer import ItineraryScores
from evaluation.tools.run_scheduled_eval import _summarize_by_shape


class TestOptionalPreferenceDimension:
    def test_average_over_six_when_scored(self):
        scores = ItineraryScores(
            book_relevance=4, preference_adherence=2, completeness=4,
            actionability=4, geographical_accuracy=4, engagement=4,
        )
        assert scores.average_score() == pytest.approx(22 / 6)

    def test_average_over_five_when_not_scored(self):
        scores = ItineraryScores(
            book_relevance=4, preference_adherence=None, completeness=4,
            actionability=4, geographical_accuracy=4, engagement=4,
        )
        assert scores.average_score() == pytest.approx(4.0)

    def test_none_is_default(self):
        scores = ItineraryScores(
            book_relevance=3, completeness=3, actionability=3,
            geographical_accuracy=3, engagement=3,
        )
        assert scores.preference_adherence is None


class TestSummarizeByShape:
    def _case(self, has_prefs, avg, tokens=1000):
        return {
            "has_preferences": has_prefs,
            "scores": {"average": avg},
            "token_usage": {"total_tokens": tokens},
        }

    def test_segments_and_never_blends(self):
        results = [
            self._case(True, 4.0, 2000),
            self._case(True, 3.0, 1000),
            self._case(False, 2.0, 500),
            {"has_preferences": False},  # unscored (failed) case: excluded
        ]
        shapes = _summarize_by_shape(results)
        assert shapes["with_preferences"] == {
            "n": 2, "mean_average": 3.5, "mean_total_tokens": 1500,
        }
        assert shapes["without_preferences"]["n"] == 1
        assert shapes["without_preferences"]["mean_average"] == 2.0

    def test_empty_shape_reports_zero(self):
        shapes = _summarize_by_shape([self._case(True, 4.0)])
        assert shapes["without_preferences"] == {"n": 0}


class TestEvalsetShapeSplit:
    def test_books_v1_carries_both_shapes(self):
        """The gate's validity depends on this split existing — pin it."""
        d = json.loads(
            (Path("evaluation") / "books_v1.evalset.json").read_text()
        )
        shapes = {True: 0, False: 0}
        for case in d["eval_cases"]:
            state = case.get("session_input", {}).get("state", {}) or {}
            has = bool(state.get("user:preferences"))
            shapes[has] += 1
            # A preference-free case must not carry a preference criterion
            # (the judge would grade adherence to nothing).
            if not has:
                assert "preference_adherence" not in (
                    case.get("quality_criteria") or {}
                ), case["eval_id"]
        assert shapes[True] >= 3, "API-contract shape underrepresented"
        assert shapes[False] >= 3, "prod shape underrepresented"
