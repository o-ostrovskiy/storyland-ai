"""
Unit tests for the judge-calibration tooling.

Pure-logic coverage (selection, interleaving, agreement math) plus the
Langfuse-facing helpers with mocked clients — no live API.
"""

import enum
from unittest.mock import MagicMock

import pytest

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


class _StrMixinSource(str, enum.Enum):
    """The shape ``langfuse.api.core.enum.StrEnum`` degrades to below py3.11.

    ``str(member)`` is "_StrMixinSource.ANNOTATION" here and "ANNOTATION" on
    the native ``enum.StrEnum`` — which is why the source guard reads
    ``.value`` instead of stringifying the member.
    """

    ANNOTATION = "ANNOTATION"
    API = "API"


class _PlainSource(enum.Enum):
    ANNOTATION = "ANNOTATION"
    API = "API"


def _source_spellings():
    """Every spelling of a score source hydrate_candidate can be handed.

    The legacy API returned a plain string; v3 returns an enum member whose
    ``str()`` depends on the interpreter. Pinning only the string is what let
    the old guard pass on CI's 3.12 while inverting on the 3.10 this package
    still supports, so the rows drive real members — including the pinned
    SDK's own ``ScoreSource`` when it is importable.
    """
    cases = [
        pytest.param("ANNOTATION", "API", id="plain-str"),
        pytest.param(
            _StrMixinSource.ANNOTATION, _StrMixinSource.API, id="str-enum-py310-shape"
        ),
        pytest.param(_PlainSource.ANNOTATION, _PlainSource.API, id="plain-enum"),
    ]
    if hasattr(enum, "StrEnum"):  # py>=3.11 shape, what CI actually runs
        native = enum.StrEnum("_NativeSource", {"ANNOTATION": "ANNOTATION", "API": "API"})
        cases.append(
            pytest.param(native.ANNOTATION, native.API, id="native-strenum-py311")
        )
    try:  # the real member off the pinned SDK, so a v5 retype reds here
        from langfuse.api.commons.types.score_source import ScoreSource

        cases.append(
            pytest.param(ScoreSource.ANNOTATION, ScoreSource.API, id="sdk-scoresource")
        )
    except ImportError:  # pragma: no cover - SDK layout moved; other rows still hold
        pass
    return cases


SOURCE_SPELLINGS = _source_spellings()


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
        # No source filter: UI labels are ANNOTATION, API-entered labels are
        # API — the human_ prefix is the contract.
        langfuse.api.scores_v3.get_many_v3.assert_called_once_with(
            trace_id="t1", limit=100
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
    """hydrate_candidate reads the v2 Observations API (root observation
    I/O arrives as raw JSON strings) plus the v3 Scores API — the legacy
    GET /traces/{id} it replaced is served only until 2026-11-16."""

    def _obs_page(self, output, input_data=None, parent=None):
        import json as _json

        obs = MagicMock()
        obs.parent_observation_id = parent
        obs.output = _json.dumps(output) if output is not None else None
        obs.input = _json.dumps(input_data) if input_data is not None else None
        page = MagicMock()
        page.data = [obs]
        page.meta.cursor = None
        return page

    def _judge_score(self, name, value):
        score = MagicMock()
        score.name = name
        score.value = value
        score.source = "API"
        return score

    def _wire(self, langfuse, obs_page, scores):
        langfuse.api.observations.get_many.return_value = obs_page
        langfuse.api.scores_v3.get_many_v3.return_value = MagicMock(data=scores)

    def test_builds_manifest_entry(self):
        langfuse = MagicMock()
        self._wire(
            langfuse,
            self._obs_page(
                output={"summary": "an itinerary"},
                input_data={
                    "book_title": "Dracula",
                    "author": "Bram Stoker",
                    "preferences": {"budget": "low"},
                },
            ),
            scores=[self._judge_score("book_relevance", 4.0)],
        )
        entry = hydrate_candidate(
            langfuse, "https://lf.example", _candidate("books_v1", "q1", "run1"),
            project_id="p",
        )
        assert entry["book_title"] == "Dracula"
        assert entry["has_preferences"] is True
        assert entry["judge_scores"]["book_relevance"] == 4
        assert entry["judge_scores"]["engagement"] is None
        assert entry["trace_url"] == (
            "https://lf.example/project/p/traces/trace-books_v1-q1-run1"
        )

    def test_no_preferences_shape(self):
        langfuse = MagicMock()
        self._wire(
            langfuse,
            self._obs_page(
                output={"summary": "x"},
                input_data={"book_title": "Wild", "preferences": {}},
            ),
            scores=[self._judge_score("engagement", 3.0)],
        )
        entry = hydrate_candidate(
            langfuse, "https://lf.example", _candidate("books_v1", "q1", "run1")
        )
        assert entry["has_preferences"] is False
        # No project id -> no URL, but the entry still hydrates.
        assert entry["trace_url"] is None

    def test_drops_trace_without_output(self):
        langfuse = MagicMock()
        self._wire(langfuse, self._obs_page(output=None), scores=[])
        assert hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1")
        ) is None

    def test_drops_trace_without_root_observation(self):
        # Every observation has a parent: the root never arrived (partial
        # ingest). Must drop, not crash on root=None.
        langfuse = MagicMock()
        self._wire(
            langfuse,
            self._obs_page(output={"summary": "x"}, parent="some-parent"),
            scores=[self._judge_score("engagement", 3.0)],
        )
        assert hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1")
        ) is None

    def test_drops_trace_without_judge_scores(self):
        langfuse = MagicMock()
        self._wire(langfuse, self._obs_page(output={"summary": "x"}), scores=[])
        assert hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1")
        ) is None

    @pytest.mark.parametrize("annotation_source,api_source", SOURCE_SPELLINGS)
    def test_annotation_scores_never_taken_as_judge_scores(
        self, annotation_source, api_source
    ):
        # Human labels must never be read back as judge scores: that silently
        # poisons the calibration pack with the very labels it is measured
        # against. Driven for every spelling `source` can arrive in, because
        # str(member) differs by interpreter version (see _source_spellings).
        langfuse = MagicMock()
        human = self._judge_score("book_relevance", 1.0)
        human.source = annotation_source
        self._wire(langfuse, self._obs_page(output={"summary": "x"}), scores=[human])
        assert hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1")
        ) is None

    @pytest.mark.parametrize("annotation_source,api_source", SOURCE_SPELLINGS)
    def test_judge_scores_survive_every_source_spelling(
        self, annotation_source, api_source
    ):
        # Converse of the row above: the guard must drop ANNOTATION, not
        # everything. A non-annotation source in the same spelling hydrates.
        langfuse = MagicMock()
        judged = self._judge_score("book_relevance", 4.0)
        judged.source = api_source
        self._wire(langfuse, self._obs_page(output={"summary": "x"}), scores=[judged])
        entry = hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1")
        )
        assert entry is not None
        assert entry["judge_scores"]["book_relevance"] == 4

    def test_rescored_trace_keeps_the_newest_judge_score(self):
        # Scores API v3 returns newest first. A rescored trace carries several
        # judge scores per dimension, and taking the last one calibrates the
        # human labels against a superseded judge result -- silently, since
        # every value in the list is a real score. fetch_human_scores() has
        # kept this guard since it was written; this loop did not.
        langfuse = MagicMock()
        newest = self._judge_score("book_relevance", 5.0)
        oldest = self._judge_score("book_relevance", 2.0)
        self._wire(
            langfuse,
            self._obs_page(output={"summary": "x"}),
            scores=[newest, oldest],
        )
        entry = hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1")
        )
        assert entry["judge_scores"]["book_relevance"] == 5

    def test_an_annotation_never_shadows_a_later_judge_score(self):
        # The converse the newest-first guard could break: an ANNOTATION
        # arriving first must not occupy the dimension and lock the judge
        # score out. It is dropped before the slot is taken, so the judge
        # score behind it still lands.
        langfuse = MagicMock()
        human = self._judge_score("book_relevance", 1.0)
        human.source = "ANNOTATION"
        judged = self._judge_score("book_relevance", 4.0)
        self._wire(
            langfuse,
            self._obs_page(output={"summary": "x"}),
            scores=[human, judged],
        )
        entry = hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1")
        )
        assert entry["judge_scores"]["book_relevance"] == 4

    def test_root_found_on_later_page(self):
        # Pages sort newest-first, so the root span (started first) lands on
        # the LAST page of a large trace — pagination must reach it.
        langfuse = MagicMock()
        first_page = self._obs_page(output={"leaf": True}, parent="the-root")
        first_page.meta.cursor = "next"
        last_page = self._obs_page(output={"summary": "x"})
        langfuse.api.observations.get_many.side_effect = [first_page, last_page]
        langfuse.api.scores_v3.get_many_v3.return_value = MagicMock(
            data=[self._judge_score("engagement", 3.0)]
        )
        entry = hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1")
        )
        assert entry is not None
        assert langfuse.api.observations.get_many.call_count == 2

    def test_fetch_error_drops_candidate_after_retries(self):
        langfuse = MagicMock()
        langfuse.api.observations.get_many.side_effect = Exception("timeout")
        assert hydrate_candidate(
            langfuse, "h", _candidate("books_v1", "q1", "run1"),
            attempts=3, retry_delays=(0,),
        ) is None
        assert langfuse.api.observations.get_many.call_count == 3

    def test_transient_fetch_error_recovers(self):
        langfuse = MagicMock()
        langfuse.api.observations.get_many.side_effect = [
            Exception("timeout"),
            self._obs_page(output={"summary": "x"}),
        ]
        langfuse.api.scores_v3.get_many_v3.return_value = MagicMock(
            data=[self._judge_score("engagement", 3.0)]
        )
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
