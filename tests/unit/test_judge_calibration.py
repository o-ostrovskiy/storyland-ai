"""
Unit tests for the judge-calibration tooling.

Pure-logic coverage (selection, interleaving, agreement math) plus the
Langfuse-facing helpers with mocked clients — no live API.
"""

import enum
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from evaluation.tools.judge_calibration import (
    collect_candidates,
    collect_experiment_candidates,
    experiment_item_case_id,
    resolve_dataset_id,
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


# --- MYS-909 PR1: the union read -------------------------------------------


class _Obj:
    """A plain attribute bag.

    Deliberately NOT a MagicMock: a MagicMock auto-creates every attribute, so
    a row asserting `item.metadata["eval_id"]` passes against a mock even when
    the SDK has no such field. This is the trap that stopped the first attempt
    at this card, one level down.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _legacy_run(name, items):
    return _Obj(name=name, created_at=name)


def _legacy_details(items):
    return _Obj(dataset_run_items=[
        _Obj(dataset_item_id=case, trace_id=trace) for case, trace in items
    ])


def _experiment(name, exp_id):
    return _Obj(name=name, id=exp_id, start_time=name)


def _experiment_item(case, trace, group="metadata", item_id="ei-1"):
    """An experiment item carrying eval_id in exactly ONE metadata group."""
    bag = {"metadata": None, "experiment_item_metadata": None,
           "experiment_metadata": None}
    bag[group] = {"eval_id": case}
    return _Obj(id=item_id, trace_id=trace, experiment_item_id=f"scoped-{trace}",
                **bag)


def _client(legacy=None, experiments=None, dataset_id="ds-1"):
    """A langfuse double whose api surface is exactly what the code may call."""
    legacy = legacy or {}
    experiments = experiments or {}

    api = _Obj()
    api.datasets = _Obj(
        get=lambda name: _Obj(id=dataset_id) if dataset_id else _Obj(id=None),
        get_runs=lambda name, limit=50: _Obj(
            data=[_legacy_run(n, i) for n, i in legacy.items()]
        ),
        get_run=lambda name, run_name: _legacy_details(legacy[run_name]),
    )
    api.experiments = _Obj(
        list=lambda **kw: _Obj(
            data=[_experiment(n, f"exp-{n}") for n in experiments]
        ),
        list_items=lambda **kw: _Obj(
            data=experiments[kw["experiment_id"].removeprefix("exp-")]
        ),
    )
    return _Obj(api=api)


class TestSymbolPresence:
    """The pinned SDK really has these, asserted against the imported client."""

    def test_experiments_client_exposes_list_and_list_items(self):
        from langfuse.api.experiments.client import ExperimentsClient

        for method in ("list", "list_items"):
            assert callable(getattr(ExperimentsClient, method, None)), method

    def test_from_start_time_is_required_on_both(self):
        import inspect

        from langfuse.api.experiments.client import ExperimentsClient

        for method in ("list", "list_items"):
            param = inspect.signature(
                getattr(ExperimentsClient, method)
            ).parameters["from_start_time"]
            assert param.default is inspect.Parameter.empty, method

    def test_experiment_item_has_no_dataset_item_id(self):
        """The finding this PR's shape exists for, asserted rather than recalled."""
        from langfuse.api.experiments import ExperimentItem

        fields = set(ExperimentItem.model_fields)
        assert "dataset_item_id" not in fields
        assert {"metadata", "experiment_item_metadata",
                "experiment_metadata"} <= fields

    def test_datasets_client_exposes_get(self):
        from langfuse.api.datasets.client import DatasetsClient

        assert callable(getattr(DatasetsClient, "get", None))


class TestResolveDatasetId:
    def test_returns_the_id(self):
        assert resolve_dataset_id(_client(), "books_v1") == "ds-1"

    def test_raises_on_a_miss_rather_than_returning_falsy(self):
        with pytest.raises(LookupError, match="did not resolve to an id"):
            resolve_dataset_id(_client(dataset_id=None), "nope")


class TestExperimentItemCaseId:
    @pytest.mark.parametrize(
        "group", ["metadata", "experiment_item_metadata", "experiment_metadata"]
    )
    def test_reads_eval_id_from_any_group(self, group):
        item = _experiment_item("case-a", "t1", group=group)
        assert experiment_item_case_id(item) == "case-a"

    def test_raises_when_no_group_carries_it(self):
        item = _Obj(id="ei-9", metadata={}, experiment_item_metadata=None,
                    experiment_metadata=None, experiment_item_id="scoped-9")
        with pytest.raises(LookupError, match="no eval_id"):
            experiment_item_case_id(item)

    def test_does_not_substitute_the_run_scoped_id(self):
        """experiment_item_id is present and must NOT be used as the case id."""
        item = _Obj(id="ei-9", metadata={}, experiment_item_metadata=None,
                    experiment_metadata=None, experiment_item_id="scoped-9")
        with pytest.raises(LookupError):
            experiment_item_case_id(item)


class TestUnionIsTotal:
    def test_legacy_only(self):
        """The row that proves this PR ships safely TODAY, before any write moves."""
        client = _client(legacy={"run-a": [("case-1", "t1"), ("case-2", "t2")]})
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1", "t2"]
        assert [c["item_id"] for c in got] == ["case-1", "case-2"]

    def test_experiment_only(self):
        client = _client(experiments={"run-b": [
            _experiment_item("case-1", "t3"), _experiment_item("case-2", "t4")]})
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t3", "t4"]
        assert [c["item_id"] for c in got] == ["case-1", "case-2"]

    def test_both_sides(self):
        client = _client(
            legacy={"run-a": [("case-1", "t1")]},
            experiments={"run-b": [_experiment_item("case-1", "t3")]},
        )
        got = collect_candidates(client, ["books_v1"], "run-")
        assert sorted(c["trace_id"] for c in got) == ["t1", "t3"]

    def test_same_run_name_on_both_sides_yields_one_candidate(self):
        """Caught by the trace net, not by a name comparison -- see the sibling
        row below for why there is no longer a name net to catch it."""
        client = _client(
            legacy={"run-a": [("case-1", "t1")]},
            experiments={"run-a": [_experiment_item("case-1", "t1")]},
        )
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1"]

    def test_same_trace_under_DIFFERENT_run_names_yields_one_candidate(self):
        """\U0001f534 The row the trace net actually needs, and the one the row
        above does not carry: with matching names the name net drops the
        duplicate first, so removing the trace de-dup left the suite green.
        A generation is ONE trace whichever endpoint described it -- a second
        copy silently doubles its weight in the calibration pack."""
        client = _client(
            legacy={"run-a": [("case-1", "t1")]},
            experiments={"run-b": [_experiment_item("case-1", "t1")]},
        )
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1"]


class TestCapSurvivesTheUnion:
    """\U0001f534 The row a golden-shape contract row cannot see.

    The keys are byte-identical whichever id space fills them; only their
    MEANING moves. So the falsification is two runs of the SAME case.
    """

    @staticmethod
    def _cap(client):
        return select_candidates(
            collect_candidates(client, ["books_v1"], "run-"),
            target=30, max_per_case=2,
        )

    def test_two_legacy_runs_of_one_case_admit_two(self):
        client = _client(legacy={
            "run-a": [("case-1", "t1")], "run-b": [("case-1", "t2")]})
        assert len(self._cap(client)) == 2

    def test_three_experiment_runs_of_one_case_admit_two_not_three(self):
        client = _client(experiments={
            "run-a": [_experiment_item("case-1", "t1")],
            "run-b": [_experiment_item("case-1", "t2")],
            "run-c": [_experiment_item("case-1", "t3")],
        })
        assert len(self._cap(client)) == 2

    def test_mixed_legacy_and_experiment_share_one_case_key(self):
        """The case that matters: a mapping self-consistent WITHIN a leg passes
        the two rows above and still puts two id spaces in one column."""
        client = _client(
            legacy={"run-a": [("case-1", "t1")]},
            experiments={"run-b": [_experiment_item("case-1", "t2")],
                         "run-c": [_experiment_item("case-1", "t3")]},
        )
        selected = self._cap(client)
        assert len(selected) == 2
        assert {(c["dataset"], c["item_id"]) for c in selected} == {
            ("books_v1", "case-1")
        }

    def test_converse_two_different_cases_both_survive(self):
        """Or the cap rows above are satisfied by a cap that admits nothing."""
        client = _client(
            legacy={"run-a": [("case-1", "t1")]},
            experiments={"run-b": [_experiment_item("case-2", "t2")]},
        )
        assert len(self._cap(client)) == 2


class TestExperimentLegDetails:
    def test_run_prefix_filters_experiments(self):
        client = _client(experiments={
            "run-a": [_experiment_item("case-1", "t1")],
            "probe-x": [_experiment_item("case-2", "t2")],
        })
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1"]

    def test_items_without_a_trace_are_dropped(self):
        client = _client(experiments={"run-a": [
            _experiment_item("case-1", "t1"),
            _experiment_item("case-2", None),
        ]})
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1"]

    def test_from_start_time_is_passed_and_is_not_a_selector(self):
        seen = {}

        client = _client(experiments={"run-a": [_experiment_item("c", "t1")]})
        original = client.api.experiments.list

        def spy(**kw):
            seen.update(kw)
            return original(**kw)

        client.api.experiments.list = spy
        collect_candidates(client, ["books_v1"], "run-")
        floor = seen["from_start_time"]
        assert floor.year <= 2020, "the floor must never be able to filter a run"
        assert seen["dataset_id"] == "ds-1", "the ID, not the name"

    def test_a_failing_experiment_leg_does_not_lose_the_legacy_leg(self):
        # 🔴 MYS-914 Codex P1 (discussion_r3827349867): this used to drive the
        # failure via `dataset_id=None`, i.e. a `LookupError` out of
        # `resolve_dataset_id` -- but that is now a fail-closed signal that
        # must PROPAGATE (see TestExperimentLookupErrorsPropagate below), not
        # a degrade-gracefully case. A TRANSPORT failure is what this row is
        # actually about: the experiments leg answering with an error the
        # legacy leg should still survive.
        client = _client(legacy={"run-a": [("case-1", "t1")]})
        client.api.experiments.list = _raise
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1"]


# --- MYS-914 Codex P1 fixes --------------------------------------------------


def _raise(*_a, **_kw):
    raise RuntimeError("boom")


class TestLegacyFailureDoesNotBlockExperiments:
    """discussion_r3825860886: the legacy leg's `continue` used to skip the
    WHOLE dataset, experiments leg included -- exactly the moment
    `datasets.get_runs` retires and every dataset takes this branch."""

    def test_legacy_raises_experiments_still_collected(self):
        client = _client(experiments={"run-a": [_experiment_item("case-1", "t1")]})
        client.api.datasets.get_runs = _raise
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1"]

    def test_both_legs_raising_returns_empty_not_raise(self):
        client = _client()
        client.api.datasets.get_runs = _raise
        client.api.datasets.get = _raise
        assert collect_candidates(client, ["books_v1"], "run-") == []


# --- MYS-914 Codex P2 fixes (discussion_r3827349867, discussion_r3827349872) -


class TestExperimentLookupErrorsPropagate:
    """discussion_r3827349867: a bare `except Exception` around the
    experiments leg re-opened the two fail-closed guards it wraps
    (`resolve_dataset_id`, `experiment_item_case_id`) -- `LookupError` is an
    `Exception`, so a malformed write or an unresolvable dataset name
    degraded to an empty experiment leg behind a warning log, exactly the
    silence those raises exist to prevent."""

    def test_missing_eval_id_aborts_rather_than_falling_back_to_legacy(self):
        client = _client(
            legacy={"run-a": [("case-1", "t-legacy")]},
            experiments={"run-a": [_Obj(
                id="ei-1", trace_id="t-exp",
                metadata=None, experiment_item_metadata=None,
                experiment_metadata=None,
            )]},
        )
        with pytest.raises(LookupError, match="carries no eval_id"):
            collect_candidates(client, ["books_v1"], "run-")

    def test_unresolvable_dataset_name_aborts(self):
        client = _client(
            legacy={"run-a": [("case-1", "t-legacy")]}, dataset_id=None
        )
        with pytest.raises(LookupError, match="did not resolve to an id"):
            collect_candidates(client, ["books_v1"], "run-")

    def test_transport_failure_on_experiments_list_still_degrades(self):
        """🔴 The converse, and it is not optional: a plain transport error
        (not a LookupError) must still degrade to an empty experiment leg and
        return the legacy candidates, rather than raising. Without this row
        the fix trades a fail-open for a fail-closed that cannot survive a
        flaky endpoint. Identical in shape to
        TestLegacyFailureDoesNotBlockExperiments's own row above, asserted
        here too so this class states its own converse rather than pointing
        at one two classes away."""
        client = _client(legacy={"run-a": [("case-1", "t1")]})
        client.api.experiments.list = _raise
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1"]


class TestDedupeBeforeCap:
    """discussion_r3827349872: the cap used to count RUN SLOTS before
    `_dedupe_by_trace` removed overlapping traces, so a run sharing its
    trace with an earlier one still consumed a slot and then emptied it --
    the pack under-filled and lost run diversity silently."""

    def test_a_duplicated_trace_does_not_evict_a_distinct_run(self):
        newer = datetime(2026, 6, 3, tzinfo=timezone.utc)
        dup = datetime(2026, 6, 2, tzinfo=timezone.utc)
        older = datetime(2026, 6, 1, tzinfo=timezone.utc)
        client = _client(
            legacy={
                "run-newest": [("case-1", "t-shared")],
                "run-third": [("case-3", "t-distinct")],
            },
            experiments={"run-dup": [_experiment_item("case-2", "t-shared")]},
        )
        client.api.datasets.get_runs = lambda name, limit=50: _Obj(data=[
            _Obj(name="run-newest", created_at=newer),
            _Obj(name="run-third", created_at=older),
        ])
        client.api.experiments.list = lambda **kw: _Obj(data=[
            _Obj(name="run-dup", id="exp-run-dup", start_time=dup),
        ])
        got = collect_candidates(client, ["books_v1"], "run-", max_runs_per_dataset=2)
        traces = {c["trace_id"] for c in got}
        assert traces == {"t-shared", "t-distinct"}, (
            f"a fully-duplicate run must not consume a cap slot and evict a "
            f"distinct run: got {traces}"
        )

    def test_cap_still_caps_when_nothing_overlaps(self):
        """The converse, already covered by
        TestCapAppliesOnceAfterMerge::test_eight_plus_eight_survives_as_eight_not_sixteen
        above -- eight legacy plus eight experiment runs with no overlapping
        traces still caps to exactly eight. Restated here as its own row so
        this class's converse is not merely a cross-reference."""
        legacy = {f"run-l{i}": [(f"case-l{i}", f"t-l{i}")] for i in range(8)}
        experiments = {f"run-e{i}": [_experiment_item(f"case-e{i}", f"t-e{i}")]
                       for i in range(8)}
        client = _client(legacy=legacy, experiments=experiments)
        got = collect_candidates(client, ["books_v1"], "run-", max_runs_per_dataset=8)
        assert len(got) == 8


class TestRecencyAcrossLegs:
    """discussion_r3825860891: legacy always sorted first by PROVENANCE, not
    by time -- a genuinely newer experiment run could sit behind up to
    max_runs_per_dataset older legacy runs and be dropped by the cap."""

    def test_a_newer_experiment_run_outranks_an_older_legacy_run(self):
        old = datetime(2026, 1, 1, tzinfo=timezone.utc)
        new = datetime(2026, 6, 1, tzinfo=timezone.utc)
        client = _client(
            legacy={"run-old": [("case-1", "t-old")]},
            experiments={"run-new": [_experiment_item("case-1", "t-new")]},
        )
        client.api.datasets.get_runs = lambda name, limit=50: _Obj(
            data=[_Obj(name="run-old", created_at=old)]
        )
        client.api.experiments.list = lambda **kw: _Obj(
            data=[_Obj(name="run-new", id="exp-run-new", start_time=new)]
        )
        got = collect_candidates(client, ["books_v1"], "run-")
        selected = select_candidates(got, target=30, max_per_case=1)
        assert [c["trace_id"] for c in selected] == ["t-new"]

    def test_naive_and_aware_timestamps_do_not_raise(self):
        """A naive datetime from one leg must not TypeError against an aware
        one from the other -- normalize explicitly, per the finding's note,
        rather than papering over it with a try."""
        naive_old = datetime(2026, 1, 1)
        aware_new = datetime(2026, 6, 1, tzinfo=timezone.utc)
        client = _client(
            legacy={"run-old": [("case-1", "t-old")]},
            experiments={"run-new": [_experiment_item("case-1", "t-new")]},
        )
        client.api.datasets.get_runs = lambda name, limit=50: _Obj(
            data=[_Obj(name="run-old", created_at=naive_old)]
        )
        client.api.experiments.list = lambda **kw: _Obj(
            data=[_Obj(name="run-new", id="exp-run-new", start_time=aware_new)]
        )
        got = collect_candidates(client, ["books_v1"], "run-")
        assert sorted(c["trace_id"] for c in got) == ["t-new", "t-old"]


class TestCapAppliesOnceAfterMerge:
    def test_eight_plus_eight_survives_as_eight_not_sixteen(self):
        legacy = {f"run-l{i}": [(f"case-l{i}", f"t-l{i}")] for i in range(8)}
        experiments = {f"run-e{i}": [_experiment_item(f"case-e{i}", f"t-e{i}")]
                       for i in range(8)}
        client = _client(legacy=legacy, experiments=experiments)
        got = collect_candidates(client, ["books_v1"], "run-", max_runs_per_dataset=8)
        assert len(got) == 8


class TestDedupePrecedenceSurvivesTheReSort:
    def test_a_tied_timestamp_keeps_the_legacy_description(self):
        tied = datetime(2026, 3, 1, tzinfo=timezone.utc)
        client = _client(
            legacy={"run-legacy": [("case-1", "t1")]},
            experiments={"run-exp": [_experiment_item("case-1", "t1")]},
        )
        client.api.datasets.get_runs = lambda name, limit=50: _Obj(
            data=[_Obj(name="run-legacy", created_at=tied)]
        )
        client.api.experiments.list = lambda **kw: _Obj(
            data=[_Obj(name="run-exp", id="exp-run-exp", start_time=tied)]
        )
        got = collect_candidates(client, ["books_v1"], "run-")
        assert len(got) == 1
        assert got[0]["run_name"] == "run-legacy"
