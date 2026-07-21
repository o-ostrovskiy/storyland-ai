"""
Unit tests for the weekly human spot-check flow.

Covers the selection logic in run_scheduled_eval, the non-fatal annotation
enqueue, and the trend-report section that keeps skipped reviews visible.
"""

from datetime import datetime
from unittest.mock import MagicMock

from evaluation.tools.run_scheduled_eval import (
    select_spot_check_cases,
    enqueue_for_human_review,
)
from evaluation.tools.eval_dashboard import build_spot_check_section


def _dataset_result(dataset_name, cases):
    return {"dataset_name": dataset_name, "case_results": cases}


def _scored_case(item_id, book="Dracula", trace_id=None):
    return {
        "item_id": item_id,
        "run_name": "eval_run_20260720_120000_v3",
        "trace_id": trace_id or f"trace-{item_id}",
        "status": "evaluated",
        "book_title": book,
        "scores": {"average": 3.5},
    }


class TestSelectSpotCheckCases:
    def test_deterministic_per_seed(self):
        results = [
            _dataset_result("books_v1", [_scored_case(f"q{i}") for i in range(8)])
        ]
        first = select_spot_check_cases(results, k=2, seed="20260720")
        second = select_spot_check_cases(results, k=2, seed="20260720")
        assert first == second
        assert len(first) == 2

    def test_different_seeds_can_differ(self):
        results = [
            _dataset_result("books_v1", [_scored_case(f"q{i}") for i in range(20)])
        ]
        picks = {
            tuple(c["item_id"] for c in select_spot_check_cases(results, k=2, seed=s))
            for s in ("a", "b", "c", "d", "e")
        }
        assert len(picks) > 1

    def test_k_caps_at_available_scored_cases(self):
        results = [_dataset_result("books_v1", [_scored_case("q1")])]
        selected = select_spot_check_cases(results, k=2, seed="x")
        assert len(selected) == 1

    def test_only_scored_evaluated_cases_eligible(self):
        cases = [
            _scored_case("q1"),
            {"item_id": "q2", "status": "failed", "error": "boom"},
            {"item_id": "q3", "status": "evaluated"},  # no scores (judge failed)
            {"item_id": "q4", "status": "skipped"},
        ]
        results = [_dataset_result("books_v1", cases)]
        selected = select_spot_check_cases(results, k=4, seed="x")
        assert [c["item_id"] for c in selected] == ["q1"]

    def test_no_candidates_returns_empty(self):
        assert select_spot_check_cases([], k=2, seed="x") == []
        assert select_spot_check_cases(
            [_dataset_result("books_v1", [])], k=2, seed="x"
        ) == []

    def test_selection_spans_datasets(self):
        results = [
            _dataset_result("storyland_eval", [_scored_case(f"s{i}") for i in range(3)]),
            _dataset_result("books_v1", [_scored_case(f"b{i}") for i in range(3)]),
        ]
        selected = select_spot_check_cases(results, k=6, seed="x")
        assert {c["dataset"] for c in selected} == {"storyland_eval", "books_v1"}

    def test_descriptor_shape(self):
        results = [_dataset_result("books_v1", [_scored_case("q1", book="Wild")])]
        (case,) = select_spot_check_cases(results, k=1, seed="x")
        assert case == {
            "dataset": "books_v1",
            "item_id": "q1",
            "run_name": "eval_run_20260720_120000_v3",
            "trace_id": "trace-q1",
            "book_title": "Wild",
        }


class TestEnqueueForHumanReview:
    def _selected(self):
        return [
            {"item_id": "q1", "trace_id": "t1"},
            {"item_id": "q2", "trace_id": "t2"},
        ]

    def test_enqueues_each_trace(self):
        langfuse = MagicMock()
        assert enqueue_for_human_review(langfuse, "queue-1", self._selected()) == 2
        langfuse.api.annotation_queues.create_queue_item.assert_any_call(
            "queue-1", object_id="t1", object_type="TRACE"
        )

    def test_failure_is_non_fatal_and_partial(self):
        langfuse = MagicMock()
        langfuse.api.annotation_queues.create_queue_item.side_effect = [
            Exception("api down"),
            MagicMock(),
        ]
        assert enqueue_for_human_review(langfuse, "queue-1", self._selected()) == 1

    def test_missing_trace_id_skipped(self):
        langfuse = MagicMock()
        selected = [{"item_id": "q1", "trace_id": None}]
        assert enqueue_for_human_review(langfuse, "queue-1", selected) == 0
        langfuse.api.annotation_queues.create_queue_item.assert_not_called()


class TestBuildSpotCheckSection:
    def _results(self):
        return [{
            "timestamp": "2026-07-13T10:00:00",
            "results": [],
            "human_spot_check": {
                "selected": [
                    {
                        "dataset": "books_v1",
                        "item_id": "q1",
                        "trace_id": "t1",
                        "book_title": "Dracula",
                    },
                    {
                        "dataset": "storyland_eval",
                        "item_id": "s1",
                        "trace_id": "t2",
                        "book_title": "Wild",
                    },
                ],
                "status": "pending_review",
                "selected_at": "2026-07-13T10:00:00",
            },
        }]

    def test_pending_rows_visible_with_age(self):
        lines = build_spot_check_section(
            self._results(),
            annotation_checker=lambda trace_id: False,
            now=datetime(2026, 7, 20, 10, 0, 0),
        )
        text = "\n".join(lines)
        assert "## Human spot-checks" in text
        assert text.count("⏳ PENDING, 7d old") == 2
        assert "2 spot-check(s) awaiting human review" in text

    def test_reviewed_and_pending_mixed(self):
        lines = build_spot_check_section(
            self._results(),
            annotation_checker=lambda trace_id: trace_id == "t1",
            now=datetime(2026, 7, 20),
        )
        text = "\n".join(lines)
        assert "✅ Reviewed" in text
        assert "⏳ PENDING" in text
        assert "1 spot-check(s) awaiting human review" in text

    def test_no_checker_reports_unknown(self):
        text = "\n".join(build_spot_check_section(self._results(), annotation_checker=None))
        assert "❓ Unknown (no Langfuse credentials)" in text
        assert "awaiting human review" not in text

    def test_no_flagged_cases_returns_empty(self):
        assert build_spot_check_section([{"results": []}]) == []
        assert build_spot_check_section([]) == []
        no_selection = [{
            "human_spot_check": {"selected": [], "status": "no_scored_cases"}
        }]
        assert build_spot_check_section(no_selection) == []
