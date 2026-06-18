"""Tests for the relaxed book-recommendation floor + grounding post-validation.

Covers:
  * BookRecommendationsResult accepts 3-5 entries (floor relaxed from a hard 5).
  * REC_MIN_RESULTS env override + clamping via models.book._rec_min_results.
  * core.extraction.filter_grounded_recommendations drops ungrounded titles,
    is fail-open on missing/total-mismatch evidence, and flags limited_matches.
"""

import importlib

import pytest
from pydantic import ValidationError

import models.book as book_models
from models.book import BookRecommendationsResult, _rec_min_results
from core.extraction import filter_grounded_recommendations


def _rec(title="Book", author="Author", basis="themes"):
    return {
        "title": title,
        "author": author,
        "reason": "A reason.",
        "recommendation_basis": basis,
    }


class TestRelaxedFloor:
    def test_three_is_valid(self):
        model = BookRecommendationsResult(recommendations=[_rec(f"B{i}") for i in range(3)])
        assert len(model.recommendations) == 3

    def test_five_is_valid(self):
        model = BookRecommendationsResult(recommendations=[_rec(f"B{i}") for i in range(5)])
        assert len(model.recommendations) == 5

    def test_two_is_rejected(self):
        with pytest.raises(ValidationError):
            BookRecommendationsResult(recommendations=[_rec(f"B{i}") for i in range(2)])

    def test_six_is_rejected(self):
        with pytest.raises(ValidationError):
            BookRecommendationsResult(recommendations=[_rec(f"B{i}") for i in range(6)])

    def test_default_floor_is_three(self):
        assert book_models.REC_MIN_RESULTS == 3


class TestRecMinResultsResolver:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("REC_MIN_RESULTS", raising=False)
        assert _rec_min_results() == 3

    def test_override(self, monkeypatch):
        monkeypatch.setenv("REC_MIN_RESULTS", "4")
        assert _rec_min_results() == 4

    def test_clamped_high(self, monkeypatch):
        monkeypatch.setenv("REC_MIN_RESULTS", "9")
        assert _rec_min_results() == 5

    def test_clamped_low(self, monkeypatch):
        monkeypatch.setenv("REC_MIN_RESULTS", "0")
        assert _rec_min_results() == 1

    def test_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv("REC_MIN_RESULTS", "not-an-int")
        assert _rec_min_results() == 3


class TestFilterGroundedRecommendations:
    def _data(self, *titles):
        return {"recommendations": [_rec(t) for t in titles]}

    def test_drops_ungrounded_title(self):
        data = self._data("Beloved", "Invented Phantom Book")
        researcher = "Candidate: Beloved by Toni Morrison. A grounded result."
        out = filter_grounded_recommendations(data, researcher)
        titles = [r["title"] for r in out["recommendations"]]
        assert titles == ["Beloved"]
        assert out["limited_matches"] is True

    def test_keeps_all_when_grounded(self):
        data = self._data("Beloved", "Sula", "Jazz")
        researcher = "Found Beloved, Sula, and Jazz by Toni Morrison."
        out = filter_grounded_recommendations(data, researcher)
        assert len(out["recommendations"]) == 3
        assert "limited_matches" not in out  # unchanged object

    def test_fail_open_when_no_researcher_text(self):
        data = self._data("Beloved", "Whatever Book")
        out = filter_grounded_recommendations(data, "")
        assert out is data  # unchanged, nothing dropped

    def test_fail_open_when_all_dropped(self):
        data = self._data("Alpha", "Beta")
        out = filter_grounded_recommendations(data, "totally unrelated grounded text")
        # never surface an empty result; keep original
        assert out is data

    def test_case_and_whitespace_tolerant(self):
        data = self._data("The   Great Gatsby", "Phantom")
        researcher = "the great gatsby by f scott fitzgerald appeared in search."
        out = filter_grounded_recommendations(data, researcher)
        titles = [r["title"] for r in out["recommendations"]]
        assert titles == ["The   Great Gatsby"]

    def test_none_input_returns_none(self):
        assert filter_grounded_recommendations(None, "anything") is None
