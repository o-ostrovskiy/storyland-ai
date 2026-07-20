"""
Unit tests for the judge-calibration tooling.

Pure-logic coverage (selection, interleaving, agreement math) plus the
Langfuse-facing helpers with mocked clients — no live API.
"""

from unittest.mock import MagicMock

from evaluation.tools.judge_calibration import (
    DIMENSIONS,
    compute_agreement,
    fetch_human_scores,
    get_enqueued_trace_ids,
    hydrate_candidate,
    interleave_by_run,
    merge_manifest_items,
    select_candidates,
)
from evaluation.tools.llm_scorer import SCORING_CRITERIA


def _candidate(dataset, item_id, run_name):
    return {
        "dataset": dataset,
        "item_id": item_id,
        "run_name": run_name,
        "trace_id": f"trace-{dataset}-{item_id}-{run_name}",
    }


class TestInterleaveByRun:
    def test_alternates_datasets_newest_run_first(self):
        per_dataset = {
            "a": [
                [_candidate("a", "1", "run2")],
                [_candidate("a", "1", "run1")],
            ],
            "b": [
                [_candidate("b", "1", "run9")],
            ],
        }
        ordered = interleave_by_run(per_dataset)
        assert [(c["dataset"], c["run_name"]) for c in ordered] == [
            ("a", "run2"), ("b", "run9"), ("a", "run1"),
        ]

    def test_empty(self):
        assert interleave_by_run({}) == []
        assert interleave_by_run({"a": []}) == []


class TestSelectCandidates:
    def test_caps_generations_per_case(self):
        candidates = [
            _candidate("d", "q1", f"run{i}") for i in range(4)
        ] + [_candidate("d", "q2", "run0")]
        selected = select_candidates(candidates, target=30, max_per_case=2)
        q1_count = sum(1 for c in selected if c["item_id"] == "q1")
        assert q1_count == 2
        assert any(c["item_id"] == "q2" for c in selected)

    def test_stops_at_target(self):
        candidates = [_candidate("d", f"q{i}", "run0") for i in range(50)]
        assert len(select_candidates(candidates, target=30)) == 30

    def test_preserves_order(self):
        candidates = [_candidate("d", f"q{i}", "run0") for i in range(5)]
        selected = select_candidates(candidates, target=3)
        assert [c["item_id"] for c in selected] == ["q0", "q1", "q2"]


class TestComputeAgreement:
    def test_mad_and_bias_hand_computed(self):
        items = [
            {
                "judge_scores": {"book_relevance": 4},
                "human_scores": {"book_relevance": 2},
            },
            {
                "judge_scores": {"book_relevance": 3},
                "human_scores": {"book_relevance": 4},
            },
        ]
        agreement = compute_agreement(items)
        stats = agreement["per_dimension"]["book_relevance"]
        # deltas: +2, -1 -> MAD 1.5, bias +0.5, one large disagreement
        assert stats == {
            "n": 2,
            "mean_abs_diff": 1.5,
            "bias": 0.5,
            "large_disagreements": 1,
        }
        assert agreement["n_labeled_items"] == 2
        assert len(agreement["large_disagreements"]) == 1

    def test_missing_dimensions_excluded(self):
        # No-preference cases carry no preference_adherence judge score.
        items = [{
            "judge_scores": {"book_relevance": 4, "preference_adherence": None},
            "human_scores": {"book_relevance": 4},
        }]
        agreement = compute_agreement(items)
        assert agreement["per_dimension"]["preference_adherence"] == {"n": 0}
        assert agreement["per_dimension"]["book_relevance"]["n"] == 1
        assert agreement["per_dimension"]["book_relevance"]["mean_abs_diff"] == 0

    def test_unlabeled_items_counted_but_contribute_nothing(self):
        items = [
            {"judge_scores": {d: 3 for d in DIMENSIONS}, "human_scores": {}},
        ]
        agreement = compute_agreement(items)
        assert agreement["n_manifest_items"] == 1
        assert agreement["n_labeled_items"] == 0
        assert all(s["n"] == 0 for s in agreement["per_dimension"].values())

    def test_all_dimensions_reported(self):
        agreement = compute_agreement([])
        assert set(agreement["per_dimension"].keys()) == set(SCORING_CRITERIA.keys())


class TestFetchHumanScores:
    def _score(self, name, value):
        score = MagicMock()
        score.name = name
        score.value = value
        return score

    def test_maps_prefixed_names_to_dimensions(self):
        langfuse = MagicMock()
        langfuse.api.scores_v3.get_many_v3.return_value = MagicMock(
            data=[
                self._score("human_book_relevance", 4.0),
                self._score("human_engagement", 2.0),
                self._score("book_relevance", 5.0),  # judge score: ignored
                self._score("human_unknown_dim", 1.0),  # not a dimension
            ]
        )
        scores = fetch_human_scores(langfuse, "t1")
        assert scores == {"book_relevance": 4, "engagement": 2}
        langfuse.api.scores_v3.get_many_v3.assert_called_once_with(
            trace_id="t1", source="ANNOTATION", limit=100
        )

    def test_latest_value_wins(self):
        langfuse = MagicMock()
        # API returns newest first; a re-label should shadow the older value.
        langfuse.api.scores_v3.get_many_v3.return_value = MagicMock(
            data=[
                self._score("human_book_relevance", 3.0),
                self._score("human_book_relevance", 5.0),
            ]
        )
        assert fetch_human_scores(langfuse, "t1") == {"book_relevance": 3}

    def test_api_error_returns_empty(self):
        langfuse = MagicMock()
        langfuse.api.scores_v3.get_many_v3.side_effect = Exception("down")
        assert fetch_human_scores(langfuse, "t1") == {}


class TestHydrateCandidate:
    def _trace(self, output, input_data=None, scores=None, html_path="/project/p/traces/t1"):
        trace = MagicMock()
        trace.output = output
        trace.input = input_data or {}
        trace.scores = scores or []
        trace.html_path = html_path
        return trace

    def _judge_score(self, name, value):
        score = MagicMock()
        score.name = name
        score.value = value
        score.source = "API"
        return score

    def test_builds_manifest_entry(self):
        langfuse = MagicMock()
        langfuse.api.trace.get.return_value = self._trace(
            output={"summary": "an itinerary"},
            input_data={
                "book_title": "Dracula",
                "author": "Bram Stoker",
                "preferences": {"budget": "low"},
            },
            scores=[self._judge_score("book_relevance", 4.0)],
        )
        entry = hydrate_candidate(
            langfuse, "https://lf.example", _candidate("books_v1", "q1", "run1")
        )
        assert entry["book_title"] == "Dracula"
        assert entry["has_preferences"] is True
        assert entry["judge_scores"]["book_relevance"] == 4
        assert entry["judge_scores"]["engagement"] is None
        assert entry["trace_url"] == "https://lf.example/project/p/traces/t1"

    def test_no_preferences_shape(self):
        langfuse = MagicMock()
        langfuse.api.trace.get.return_value = self._trace(
            output={"summary": "x"},
            input_data={"book_title": "Wild", "preferences": {}},
            scores=[self._judge_score("engagement", 3.0)],
        )
        entry = hydrate_candidate(
            langfuse, "https://lf.example", _candidate("books_v1", "q1", "run1")
        )
        assert entry["has_preferences"] is False

    def test_drops_trace_without_output(self):
        langfuse = MagicMock()
        langfuse.api.trace.get.return_value = self._trace(output=None)
        assert hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1")
        ) is None

    def test_drops_trace_without_judge_scores(self):
        langfuse = MagicMock()
        langfuse.api.trace.get.return_value = self._trace(
            output={"summary": "x"}, scores=[]
        )
        assert hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1")
        ) is None

    def test_annotation_scores_never_taken_as_judge_scores(self):
        langfuse = MagicMock()
        human = self._judge_score("book_relevance", 1.0)
        human.source = "ANNOTATION"
        langfuse.api.trace.get.return_value = self._trace(
            output={"summary": "x"}, scores=[human]
        )
        assert hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1")
        ) is None

    def test_fetch_error_drops_candidate_after_retries(self):
        langfuse = MagicMock()
        langfuse.api.trace.get.side_effect = Exception("timeout")
        assert hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1"),
            attempts=3, retry_delays=(0,),
        ) is None
        assert langfuse.api.trace.get.call_count == 3

    def test_transient_fetch_error_recovers(self):
        langfuse = MagicMock()
        langfuse.api.trace.get.side_effect = [
            Exception("timeout"),
            self._trace(
                output={"summary": "x"},
                scores=[self._judge_score("engagement", 3.0)],
            ),
        ]
        entry = hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1"),
            attempts=3, retry_delays=(0,),
        )
        assert entry is not None


class TestMergeManifestItems:
    def test_new_wins_and_old_preserved(self):
        existing = [
            {"trace_id": "t1", "book_title": "Old"},
            {"trace_id": "t2", "book_title": "Kept"},
        ]
        new = [
            {"trace_id": "t1", "book_title": "New"},
            {"trace_id": "t3", "book_title": "Added"},
        ]
        merged = {i["trace_id"]: i["book_title"] for i in merge_manifest_items(existing, new)}
        assert merged == {"t1": "New", "t2": "Kept", "t3": "Added"}


class TestGetEnqueuedTraceIds:
    def test_paginates_until_short_page(self):
        langfuse = MagicMock()
        full_page = MagicMock(data=[MagicMock(object_id=f"t{i}") for i in range(100)])
        short_page = MagicMock(data=[MagicMock(object_id="t100")])
        langfuse.api.annotation_queues.list_queue_items.side_effect = [
            full_page, short_page
        ]
        trace_ids = get_enqueued_trace_ids(langfuse, "q1")
        assert len(trace_ids) == 101
        assert langfuse.api.annotation_queues.list_queue_items.call_count == 2

    def test_list_failure_returns_partial(self):
        langfuse = MagicMock()
        langfuse.api.annotation_queues.list_queue_items.side_effect = Exception("down")
        assert get_enqueued_trace_ids(langfuse, "q1") == set()
