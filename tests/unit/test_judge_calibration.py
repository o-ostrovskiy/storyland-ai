"""
Unit tests for the judge-calibration tooling.

Pure-logic coverage (selection, interleaving, agreement math) plus the
Langfuse-facing helpers with mocked clients — no live API.
"""

import enum
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from langfuse.api.core import ApiError

from evaluation.tools.judge_calibration import (
    CalibrationDataError,
    CalibrationTruncatedError,
    _EVAL_ID_GROUPS,
    _EVAL_ID_GROUP_OBSERVED,
    _EVAL_ID_GROUP_REQUEST_FIELDS,
    _EXPERIMENT_ITEM_FIELDS,
    collect_candidates,
    _experiment_fault_is_carved_out,
    _experiment_fault_is_systematic,
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


def _paging_client(experiment_id, pages, dataset_id="ds-1"):
    """A langfuse double whose `experiments.list_items` hands back `pages` in
    order, one per call — `pages` is a list of `(items, next_cursor)` pairs.
    Returns `(client, calls)`; `calls` records each call's kwargs so a test
    can assert call count and the `cursor` actually threaded through."""
    calls = []

    def list_items(**kw):
        calls.append(kw)
        items, cursor = pages[len(calls) - 1]
        return _Obj(data=items, meta=_Obj(cursor=cursor))

    api = _Obj()
    api.datasets = _Obj(get=lambda name: _Obj(id=dataset_id))
    api.experiments = _Obj(
        list=lambda **kw: _Obj(data=[_experiment("run-a", experiment_id)]),
        list_items=list_items,
    )
    return _Obj(api=api), calls


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

    def test_list_items_accepts_a_pagination_cursor(self):
        """The pinned SDK's own pagination handle, asserted rather than
        assumed -- same discipline as the `from_start_time` rows above."""
        import inspect

        from langfuse.api.experiments.client import ExperimentsClient

        assert "cursor" in inspect.signature(ExperimentsClient.list_items).parameters

    def test_list_items_response_carries_the_cursor_at_meta_cursor(self):
        """🔴 @el r3 FL-1 — the RESPONSE half of that same contract.

        The row above asserts `cursor` is a PARAMETER. Nothing asserted where
        the reply puts the next one, and `_experiment_items` reads
        `items_page.meta.cursor` — a shape carried over from
        `_find_root_observation`'s read of `observations.get_many`, a
        DIFFERENT endpoint with a different response model. If the real name
        were `next_cursor`, or top-level, or `meta` were absent, the double
        `getattr` yields `None`, the loop breaks on page 1 and
        `CalibrationTruncatedError` never fires — and every row in
        `TestExperimentItemPaging` stays green, because `_paging_client`
        supplies `meta.cursor` by construction: the double defines the
        contract it is testing. Only the pinned SDK can refute it, so this
        row reads the SDK.

        Deliberately strict: no unwrapping of `Optional`, no `getattr`
        fallback. A shape that stops being exactly this one must red loudly
        here rather than degrade quietly in the loop."""
        import inspect

        from langfuse.api.experiments.client import ExperimentsClient

        response = inspect.signature(ExperimentsClient.list_items).return_annotation
        assert "meta" in response.model_fields, response
        meta = response.model_fields["meta"].annotation
        assert "cursor" in meta.model_fields, meta

    def test_observation_metadata_is_flattened_per_key_not_stored_whole(self):
        """🔴 Codex `r3834098109` (P1) — the finding is REFUTED here, and the
        row exists because refuting it in a reply would leave nothing behind.

        Codex read `run_scheduled_eval.py`'s four `root_span.update(metadata=…)`
        calls and argued that "Langfuse v4 stores observation metadata in a
        single OpenTelemetry span attribute", so each update REPLACES the
        mapping set at creation and erases the `eval_id` this PR adds — which
        would make `experiment_item_case_id` abort every build once those runs
        are read. The premise is checkable against the pinned SDK and it is
        false: `_flatten_and_serialize_metadata` emits ONE attribute PER KEY
        (`langfuse.observation.metadata.<key>`) for a dict, and OTel's
        `set_attributes` merges by key, so an update naming `current_phase`
        cannot touch `eval_id`.

        ⚠️ Why it earns a row rather than a reply. The probe recorded in ADR #27
        (54 runs / 248 items / 0 missing `eval_id`) cannot speak to this: all
        three sibling writers pass only `input`/`output` to `root_span.update`,
        never `metadata`, so no run in that population ever exercised the path.
        `run_scheduled_eval.py` is the FIRST writer that does. The claim
        therefore rests entirely on the SDK's flattening, exactly as r3 FL-1's
        cursor shape did — and if a langfuse bump ever serialises the dict
        whole, `eval_id` starts disappearing silently on the one writer that
        matters. That must red HERE, loudly, not in a calibration run months
        later."""
        from langfuse._client.attributes import (
            LangfuseOtelSpanAttributes,
            _flatten_and_serialize_metadata,
        )

        prefix = LangfuseOtelSpanAttributes.OBSERVATION_METADATA
        at_creation = _flatten_and_serialize_metadata(
            {"prompt_version": "v3", "eval_id": "item-1"}, "observation"
        )
        on_update = _flatten_and_serialize_metadata(
            {"current_phase": "discovery"}, "observation"
        )

        # Per-key, not whole: the single-attribute shape Codex assumed would
        # put `prefix` itself in the mapping, and it does not.
        assert f"{prefix}.eval_id" in at_creation
        assert prefix not in at_creation

        # And the update cannot collide with it, which is the actual claim.
        assert f"{prefix}.eval_id" not in on_update
        merged = {**at_creation, **on_update}  # OTel set_attributes semantics
        assert merged[f"{prefix}.eval_id"] == "item-1"

    def test_the_fields_parameter_exposes_no_valid_value_set_to_ground_against(self):
        """⚠️ @el r4 FL-2 — this is the "say so in a line" branch, written as a
        row so it cannot go stale.

        FL-2 is right that `_EVAL_ID_GROUP_REQUEST_FIELDS` is self-consistent
        rather than grounded: every row checks it against `_EVAL_ID_GROUPS` and
        `_EXPERIMENT_ITEM_FIELDS`, all three of which are ours, and the doubles
        ignore `fields=` entirely — so a wrong camelCase spelling is green
        everywhere in this suite and only fails against the live API. The ask
        was to ground the values in the pinned SDK IF it exposes them.

        It does not: `fields` is annotated `Optional[str]`, a bare string with
        no `Literal` and no enum, so the 2026-08-21 probe (which sent the
        snake_case spelling and read the valid values back off a 400) remains
        the strongest authority available and stays recorded in a comment.
        This row asserts the ABSENCE, so that a langfuse bump introducing a
        `Literal` reds here and tells the next reader the stronger grounding
        has become available — rather than the note quietly staying true-looking
        forever."""
        import typing

        from langfuse.api.experiments.client import ExperimentsClient

        hint = typing.get_type_hints(ExperimentsClient.list_items).get("fields")
        assert hint == typing.Optional[str], (
            "the pinned SDK now constrains `fields` — ground "
            "_EVAL_ID_GROUP_REQUEST_FIELDS against it and delete this row"
        )


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


# --- MYS-914 PR1 FIX 1: `list_items` paging (discussion_r3829698246) --------
# 🔴 One request, no paging: an experiment run with more than 100 linked
# items silently contributed only its first 100 traces, unlike the legacy
# leg's `datasets.get_run`, which returns `dataset_run_items` whole. These
# rows drive the fix the way `_find_root_observation` already proves its own
# paging loop -- from a double, not a live run.


class TestExperimentItemPaging:
    def test_an_experiment_run_longer_than_one_page_contributes_from_every_page(self):
        page1 = [_experiment_item(f"case-{i}", f"t{i}") for i in range(100)]
        page2 = [_experiment_item(f"case-{i}", f"t{i}") for i in range(100, 150)]
        client, calls = _paging_client(
            "exp-run-a", [(page1, "cursor-1"), (page2, None)]
        )
        (start_time, candidates) = collect_experiment_candidates(
            client, "books_v1", "run-", 8
        )[0]
        assert len(candidates) == 150
        assert {c["trace_id"] for c in candidates} == {f"t{i}" for i in range(150)}
        assert len(calls) == 2
        assert calls[1]["cursor"] == "cursor-1"

    def test_a_single_page_run_issues_no_second_request(self):
        """The converse, not optional: a run that fits in one page still
        yields exactly its items and makes ONE call. Without this row the
        fix is satisfiable by a loop that pages forever or re-reads page 1."""
        page1 = [_experiment_item("case-0", "t0")]
        client, calls = _paging_client("exp-run-a", [(page1, None)])
        (_, candidates) = collect_experiment_candidates(
            client, "books_v1", "run-", 8
        )[0]
        assert [c["trace_id"] for c in candidates] == ["t0"]
        assert len(calls) == 1

    def test_item_paging_raises_at_its_bound_rather_than_truncating(self):
        """🔴 Codex `r3832093213`. This row previously read
        `test_item_paging_stops_at_its_bound` and asserted
        `len(candidates) == 20` — it pinned SILENT TRUNCATION as the expected
        behaviour, which is the shortfall this whole card exists to remove,
        reintroduced by the fix for it. The bound must still bound (the loop
        terminates), but exhausting it with a cursor outstanding is a refusal,
        not an answer.

        RED on `2d633ef`, where this returned 20 partial candidates."""
        pages = [
            ([_experiment_item(f"case-{i}", f"t{i}")], "always-more")
            for i in range(100)
        ]
        client, calls = _paging_client("exp-run-a", pages)
        with pytest.raises(CalibrationTruncatedError) as excinfo:
            collect_experiment_candidates(client, "books_v1", "run-", 8)
        # it still TERMINATES -- the bound is doing its job
        assert len(calls) == 20
        # and it names the run it refused, or the operator cannot act on it
        assert "run-a" in str(excinfo.value)

    def test_a_run_ending_exactly_at_the_bound_does_not_raise(self):
        """The converse, and the one that stops the fix being satisfied by
        raising whenever the bound is REACHED: a run whose last page lands on
        page 20 with no cursor is complete, not truncated. An off-by-one here
        aborts a build on a perfectly good run."""
        pages = [
            ([_experiment_item(f"case-{i}", f"t{i}")], "more") for i in range(19)
        ]
        pages.append(([_experiment_item("case-19", "t19")], None))
        client, calls = _paging_client("exp-run-a", pages)
        (_, candidates) = collect_experiment_candidates(
            client, "books_v1", "run-", 8
        )[0]
        assert len(calls) == 20
        assert len(candidates) == 20

    def test_a_truncated_run_is_not_swallowed_by_the_callers_broad_handler(self):
        """The raise is worthless if `collect_candidates` degrades on it.
        `CalibrationTruncatedError` is a `CalibrationDataError`, so the
        caller's narrow arm re-raises it instead of returning a green,
        legacy-only pack."""
        pages = [
            ([_experiment_item(f"case-{i}", f"t{i}")], "always-more")
            for i in range(100)
        ]
        client, _calls = _paging_client("exp-run-a", pages)
        client.api.datasets.get_runs = lambda name, limit: _Obj(data=[])
        with pytest.raises(CalibrationTruncatedError):
            collect_candidates(client, ["books_v1"], "run-")


class TestEvalIdGroupsAreStillRequested:
    """⚠️ @el r3 FL-2 — `_EVAL_ID_GROUP_OBSERVED` was defined and never read.

    Its stated job is to stop a future narrowing of `_EXPERIMENT_ITEM_FIELDS`
    from dropping the one group the live probe found `eval_id` in. A comment
    cannot do that job: it is a claim about intent, and only a caller is a
    claim about behaviour — this file's own r2 lesson.

    The assertion is not trivial because the claim spans two vocabularies.
    `_EVAL_ID_GROUPS` is the RESPONSE spelling (snake_case attributes on
    `ExperimentItem`, pinned by
    `TestSymbolPresence.test_experiment_item_has_no_dataset_item_id`);
    `_EXPERIMENT_ITEM_FIELDS` is the REQUEST spelling (the camelCase enum the
    API reads). One of them is what the server sees, and it is not the one the
    constant is written in.
    """

    def test_the_observed_group_is_read_and_is_still_requested(self):
        assert _EVAL_ID_GROUP_OBSERVED in _EVAL_ID_GROUPS
        requested = _EXPERIMENT_ITEM_FIELDS.split(",")
        assert _EVAL_ID_GROUP_REQUEST_FIELDS[_EVAL_ID_GROUP_OBSERVED] in requested

    def test_every_group_the_code_reads_is_also_asked_for(self):
        """The generalisation, so the guard is not satisfiable by keeping ONE
        spelling alive: reading a group the request never asked for is a
        silent always-None, which is exactly how `eval_id` would go missing."""
        requested = set(_EXPERIMENT_ITEM_FIELDS.split(","))
        for group in _EVAL_ID_GROUPS:
            assert group in _EVAL_ID_GROUP_REQUEST_FIELDS, group
            assert _EVAL_ID_GROUP_REQUEST_FIELDS[group] in requested, group

    def test_the_two_spellings_are_genuinely_different(self):
        """Vacuity guard. If the map were identity everywhere, the rows above
        would pass while asserting nothing about the camelCase enum the API
        actually reads."""
        assert (
            _EVAL_ID_GROUP_REQUEST_FIELDS["experiment_item_metadata"]
            != "experiment_item_metadata"
        )


def _raise_400(**_kw):
    """Every request fails identically, as a plain `RuntimeError`.

    🔴 r5 — worth being exact about what this double now models, because it is
    NOT what its old docstring claimed. A real wrong-`fields` spelling arrives
    from the pinned SDK as `ApiError(status_code=400)`, and an expired key as
    `UnauthorizedError(401)`; this raises a bare `RuntimeError`, which the
    classifier deliberately treats as UNCLASSIFIABLE and degrades. So this
    fixture no longer exercises the classification rule at all — it exercises
    the **gated count backstop**, at `attempted == 3`, which is exactly the
    world the count is kept for. `_ApiError400` below is the classifier's
    fixture, and it is a real `ApiError`.

    ➡️ *A double that models a fault by its message models nothing about its
    class.* The old docstring named three faults it could not represent."""
    raise RuntimeError("400 Bad Request: invalid value for 'fields'")


def _raise_status(status):
    """Every request fails with the SAME real `ApiError`. The r7 FL-1 rows need
    a fault that persists across consecutive requests, because that is what
    rate limiting actually does -- `_raise_400`'s bare `RuntimeError` is
    unclassifiable and would exercise the count without exercising the
    carve-out."""
    def _raise(**_kw):
        raise ApiError(status_code=status, body={"message": "fixture"})

    return _raise


def _api_error(status):
    """A REAL `langfuse.api.core.ApiError`, not a look-alike. The classifier
    reads `isinstance` and `.status_code`, so a duck-typed stand-in would make
    every row below green regardless of what the rule does."""
    return ApiError(status_code=status, body={"message": "fixture"})


def _flaky_items_client(fail_on, exc, runs=("run-c", "run-b", "run-a")):
    """A langfuse double whose `experiments.list_items` raises `exc` for the
    experiment whose run name is `fail_on`, and returns one item otherwise.
    Run names sort time-descending as given."""
    def list_items(**kw):
        run = kw["experiment_id"].removeprefix("exp-")
        if run == fail_on:
            raise exc
        return _Obj(data=[_experiment_item(f"case-{run}", f"t-{run}")], meta=_Obj(cursor=None))

    api = _Obj()
    api.datasets = _Obj(get=lambda name: _Obj(id="ds-1"))
    api.experiments = _Obj(
        list=lambda **kw: _Obj(data=[_experiment(n, f"exp-{n}") for n in runs]),
        list_items=list_items,
    )
    return _Obj(api=api)


class TestOneBadRunDoesNotDiscardTheGoodOnes:
    """🔴 Codex `r3832494812` (P2), against head `47773e5`.

    A transport failure on one experiment propagated out of
    `collect_experiment_candidates`, and `collect_candidates`'s broad arm then
    replaced the WHOLE experiments leg with `[]` — so one flaky read discarded
    every run already fetched. After the legacy endpoint retires that dataset
    yields nothing at all. The legacy per-run loop has always skipped the run
    and kept the rest."""

    def test_a_transient_failure_skips_its_run_and_keeps_the_others(self):
        """RED before the fix: this returned zero runs, because the exception
        left the function instead of the loop iteration."""
        client = _flaky_items_client("run-b", RuntimeError("connection reset"))
        runs = collect_experiment_candidates(client, "books_v1", "run-", 8)
        assert len(runs) == 2
        traces = {c["trace_id"] for _st, cands in runs for c in cands}
        assert traces == {"t-run-c", "t-run-a"}

    def test_the_fail_closed_signal_still_propagates(self):
        """CONVERSE 1, and the reason this is not a bare `except Exception`.
        `CalibrationDataError` is the malformed-write signal FL-A exists for;
        swallowing it here restores the silent degrade it was raised to stop."""
        client = _flaky_items_client("run-b", CalibrationDataError("no eval_id"))
        with pytest.raises(CalibrationDataError):
            collect_experiment_candidates(client, "books_v1", "run-", 8)

    def test_a_truncated_run_still_propagates(self):
        """CONVERSE 2: `CalibrationTruncatedError` subclasses
        `CalibrationDataError` precisely so a refusal cannot be re-read as
        "transient". Without this row the fix is satisfiable by catching
        everything except the one class named in the row above."""
        client = _flaky_items_client("run-b", CalibrationTruncatedError("short"))
        with pytest.raises(CalibrationTruncatedError):
            collect_experiment_candidates(client, "books_v1", "run-", 8)

    def test_the_skip_names_the_run_it_dropped(self, capsys):
        """A skip that prints nothing is a silent shortfall wearing a
        different hat: a thinner pack would be indistinguishable from a
        smaller dataset. The legacy loop names its run; so does this one."""
        client = _flaky_items_client("run-b", RuntimeError("connection reset"))
        collect_experiment_candidates(client, "books_v1", "run-", 8)
        out = capsys.readouterr().out
        assert "run-b" in out
        assert "connection reset" in out


class TestEveryRunSkippedIsNotAnEmptyLeg:
    """🔴 @el r4 FL-1, against head `3dc7e53`.

    The per-run skip arm discriminates on the exception CLASS, and class
    cannot see whether a fault is per-RUN. A wrong or narrowed camelCase
    spelling in `_EXPERIMENT_ITEM_FIELDS` is a 400 on EVERY request — not
    transport, not run-scoped — and so are an expired key, a revoked scope and
    the endpoint moving. Each arrives as a plain exception once per iteration,
    so every run was skipped, `run_lists` came back `[]`, and the pack was
    built from the legacy leg and reported as a success.

    ➡️ *The scope of a degrade is a claim about the fault's SUBJECT, and the
    subject is not readable from the type — it is readable from the outcome.*
    A skip is only per-run if some other run succeeded."""

    def test_every_run_failing_raises_rather_than_yielding_an_empty_leg(self):
        """RED on `3dc7e53`: returned `[]` and the caller shipped a legacy-only
        pack as complete."""
        client = _flaky_items_client(None, RuntimeError("400 invalid fields"))
        # `fail_on=None` never matches a run name, so make it match all of them.
        client.api.experiments.list_items = _raise_400
        with pytest.raises(CalibrationDataError) as excinfo:
            collect_experiment_candidates(client, "books_v1", "run-", 8)
        assert "3/3" in str(excinfo.value)

    def test_one_of_three_failing_still_returns_the_other_two(self):
        """CONVERSE 1 — the genuinely per-run fault is untouched. Without it
        the fix is satisfiable by raising on any skip at all, which would undo
        Codex `r3832494812` one round after it landed."""
        client = _flaky_items_client("run-b", RuntimeError("connection reset"))
        runs = collect_experiment_candidates(client, "books_v1", "run-", 8)
        assert len(runs) == 2

    def test_a_dataset_with_no_matching_runs_is_still_a_legitimate_empty_leg(self):
        """CONVERSE 2, and it is the row that keeps the build alive until PR2.
        `attempted == 0` is not "the leg could not be read", it is "there is
        nothing in it" — the state EVERY dataset is in today, before the writes
        move. Raising on it would fail every build immediately."""
        client = _flaky_items_client("run-b", RuntimeError("x"), runs=())
        assert collect_experiment_candidates(client, "books_v1", "run-", 8) == []

    def test_the_raise_is_not_swallowed_by_the_callers_broad_arm(self):
        """CONVERSE 3. `CalibrationDataError` is chosen precisely because
        `collect_candidates` re-raises exactly that class and degrades on
        everything else — the mistake `CalibrationTruncatedError` was
        subclassed to avoid. A plain `RuntimeError` here would be caught one
        level up and the whole finding would be invisible again."""
        client = _flaky_items_client(None, RuntimeError("x"))
        client.api.experiments.list_items = _raise_400
        client.api.datasets.get_runs = _raise
        with pytest.raises(CalibrationDataError):
            collect_candidates(client, ["books_v1"], "run-", 8)


class TestExperimentFaultClass:
    """🔴 @el review r5 — the ruling that replaced his own r4 FL-1.

    r4 asked for an OUTCOME rule: *a skip is only per-run if some other run
    succeeded*. True as a sentence, and unable to be expressed by the predicate
    it asked for — at `attempted == 1` there is no other run, so
    `skipped == attempted` is satisfied by any single failure and carries zero
    bits about systematicity. `storyland_eval` with a monthly prefix routinely
    matches exactly one experiment, so n=1 is the MIDDLE of this distribution,
    and the count rule both (a) aborted the build on an ordinary transient
    failure there, and (b) could not fire at n=1 on the 400 it was justified
    with.

    ➡️ *An outcome-based discriminator inherits its population's SIZE as a hard
    bound on what it can distinguish.*

    These rows drive `attempted == 1` on purpose: it is the case the previous
    rule could not speak about, and every row here is RED on `69379df`."""

    @staticmethod
    def _one_run(exc):
        return _flaky_items_client("only", exc, runs=("only",))

    def test_a_transient_failure_at_n_of_1_degrades_instead_of_aborting(self):
        """RED on `69379df`: raised `CalibrationDataError`, because
        `skipped == attempted` held vacuously at n=1. This is Codex
        `r3835776926` and it is the common case for a monthly prefix."""
        client = self._one_run(RuntimeError("connection reset by peer"))
        assert collect_experiment_candidates(client, "books_v1", "on", 8) == []

    def test_a_400_at_n_of_1_raises(self):
        """RED on `69379df`: returned `[]` and the pack shipped legacy-only and
        green. A narrowed camelCase spelling in `_EXPERIMENT_ITEM_FIELDS` is a
        400 on EVERY request — the exact fault r4 FL-1 was written for, on the
        exact n at which the counting rule cannot fire."""
        client = self._one_run(_api_error(400))
        with pytest.raises(CalibrationDataError) as excinfo:
            collect_experiment_candidates(client, "books_v1", "on", 8)
        assert "request-shaped" in str(excinfo.value)

    @pytest.mark.parametrize("status", [401, 403, 404, 405])
    def test_auth_and_endpoint_faults_raise_at_every_n(self, status):
        """An expired key, a revoked scope, the endpoint moving. None of these
        gets better on the next run and none is run-scoped."""
        client = self._one_run(_api_error(status))
        with pytest.raises(CalibrationDataError):
            collect_experiment_candidates(client, "books_v1", "on", 8)

    @pytest.mark.parametrize("status", [429, 408])
    def test_the_carve_outs_degrade_although_they_are_4xx(self, status):
        """🔴 @el's carve-outs, and the row he asked not to be skipped. 429 and
        408 are 4xx by NUMBER and transport-shaped by NATURE. A monthly
        calibration that hits a rate limit must not abort the build, and a bare
        4xx band rule would silently eat both."""
        client = self._one_run(_api_error(status))
        assert collect_experiment_candidates(client, "books_v1", "on", 8) == []

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_server_side_faults_degrade(self, status):
        """A 5xx is the server stating it could not answer THIS attempt."""
        client = self._one_run(_api_error(status))
        assert collect_experiment_candidates(client, "books_v1", "on", 8) == []

    def test_an_api_error_with_no_status_degrades(self):
        """`ApiError.status_code` is `Optional[int]` in the pinned SDK — see
        `test_api_error_status_code_is_optional_in_the_pinned_sdk`. An
        unclassifiable fault takes the branch that does not stop the build; the
        gated count is the backstop for it."""
        client = self._one_run(ApiError(body={"message": "no status"}))
        assert collect_experiment_candidates(client, "books_v1", "on", 8) == []

    def test_two_of_two_transport_failures_still_raise(self):
        """🔴 The row that pins the count rule SURVIVED at `attempted >= 2`
        rather than being deleted with its n=1 arm. This is the world
        classification misses: an endpoint genuinely 5xx-ing on every request
        degrades every run correctly, and after 2026-11-16 the legacy leg is
        empty by construction, so the pack would go out thin and green."""
        client = _flaky_items_client(None, RuntimeError("x"), runs=("a", "b"))
        client.api.experiments.list_items = _raise_400
        with pytest.raises(CalibrationDataError) as excinfo:
            collect_experiment_candidates(client, "books_v1", "", 8)
        assert "2/2" in str(excinfo.value)

    # --- 🔴 r7 FL-1 (@el, Codex `r3837185006`) -------------------------------
    #
    # The classifier said DEGRADE on a 429 and the count then aborted the build
    # anyway, in the exact situation the carve-out was written for: rate
    # limiting persists across consecutive requests, which is why 429 is in the
    # set at all. Two rules over one population, one classifying and one
    # counting, do not compose.

    @pytest.mark.parametrize("status", [429, 408])
    def test_every_run_rate_limited_degrades_and_the_build_continues(self, status):
        """🔴 RED before this commit: `skipped == attempted` at 2/2 raised
        `CalibrationDataError` on a leg the classifier had just ruled
        retry-able. @el's r5 words were "a monthly calibration that hits a rate
        limit must not abort the build"; at `attempted >= 2` it did."""
        client = _flaky_items_client(None, _api_error(status), runs=("a", "b"))
        client.api.experiments.list_items = _raise_status(status)
        assert collect_experiment_candidates(client, "books_v1", "", 8) == []

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_every_run_5xx_still_raises(self, status):
        """CONVERSE, and the whole reason the count was narrowed rather than
        deleted: an endpoint genuinely 5xx-ing on every request is the one
        world classification misses, and after 2026-11-16 the legacy leg is
        empty by construction, so the pack would otherwise go out thin and
        green. Narrowing the counted population must not empty it."""
        client = _flaky_items_client(None, _api_error(status), runs=("a", "b"))
        client.api.experiments.list_items = _raise_status(status)
        with pytest.raises(CalibrationDataError) as excinfo:
            collect_experiment_candidates(client, "books_v1", "", 8)
        assert "2/2" in str(excinfo.value)

    def test_a_mixed_window_does_not_abort(self):
        """🔴 The row that states the semantics rather than leaving them to be
        inferred. One run 429, one run 503, both skipped, the leg unread. It
        does NOT abort: the count requires that EVERY attempted run failed for
        a countable reason, so a window containing a fault the classifier
        positively named retry-able is not evidence that the leg is
        systematically unreadable. Without this row the choice between
        `countable == attempted` and `countable >= 1 and skipped == attempted`
        is invisible, and they differ exactly here."""
        seq = iter([_api_error(429), _api_error(503)])

        def _raise_next(**kw):
            raise next(seq)

        client = _flaky_items_client(None, RuntimeError("unused"), runs=("a", "b"))
        client.api.experiments.list_items = _raise_next
        assert collect_experiment_candidates(client, "books_v1", "", 8) == []

    def test_the_carve_out_is_narrow_a_bare_exception_still_counts(self):
        """🔴 The row that refuses the OTHER fix. Writing the carve-out as
        `not _experiment_fault_is_systematic(e)` reads plausibly and empties the
        counter: a bare `RuntimeError`, a 5xx and a status-less `ApiError` are
        all non-systematic, so nothing would ever be counted and the backstop
        would be deleted while looking narrowed. Only a POSITIVELY identified
        transient is carved out."""
        assert _experiment_fault_is_carved_out(_api_error(429)) is True
        assert _experiment_fault_is_carved_out(_api_error(408)) is True
        assert _experiment_fault_is_carved_out(_api_error(503)) is False
        assert _experiment_fault_is_carved_out(RuntimeError("x")) is False
        assert _experiment_fault_is_carved_out(ApiError(body={"m": "no status"})) is False
        # ...and the two predicates agree about the carve-out, which is the
        # relationship that makes the count the classifier's own leftovers.
        for st in (429, 408):
            assert _experiment_fault_is_systematic(_api_error(st)) is False
            assert _experiment_fault_is_carved_out(_api_error(st)) is True

    def test_one_of_three_transport_failures_still_returns_two(self):
        """CONVERSE — carried from r4 and re-asserted here, because the change
        that removes the n=1 abort must not remove the per-run skip with it."""
        client = _flaky_items_client("run-b", RuntimeError("connection reset"))
        assert len(collect_experiment_candidates(client, "books_v1", "run-", 8)) == 2

    def test_api_error_status_code_is_optional_in_the_pinned_sdk(self):
        """🔴 Grounded against the SDK, not against a double — the technique
        `TestSymbolPresence` exists for, applied to the fact the whole
        classification rests on. If a future langfuse stops setting
        `status_code`, or renames it, every behavioural row above would keep
        passing against `_api_error()` while the real client degraded silently
        on a 400. Only the SDK can refute that, and this is where it does."""
        import inspect

        params = inspect.signature(ApiError.__init__).parameters
        assert "status_code" in params
        assert ApiError(status_code=400, body=None).status_code == 400
        assert ApiError(body=None).status_code is None

    @pytest.mark.parametrize(
        "name,status",
        [("Error", 400), ("UnauthorizedError", 401), ("AccessDeniedError", 403),
         ("NotFoundError", 404), ("MethodNotAllowedError", 405),
         ("ServiceUnavailableError", 503)],
    )
    def test_the_sdks_typed_errors_carry_the_status_the_rule_reads(self, name, status):
        """🔴 The half I would have got wrong by reasoning. langfuse raises
        TYPED subclasses — `UnauthorizedError`, `NotFoundError` — not bare
        `ApiError`s, so classifying on `status_code` is only sound while every
        one of them actually sets it. They do, each with a fixed value, and
        this row asserts that from the pinned package. A typed error that
        stopped setting its status would degrade instead of raising, which is
        the failure direction that ships."""
        import langfuse.api as api

        cls = getattr(api, name)
        assert issubclass(cls, ApiError)
        exc = cls(headers=None) if name == "ServiceUnavailableError" else cls(body=None)
        assert exc.status_code == status
        assert _experiment_fault_is_systematic(exc) is (400 <= status < 500)


# --- MYS-914 Codex P1 fixes --------------------------------------------------


def _raise_api_error(status):
    """A callable that raises a real SDK `ApiError` with `status`.

    Deliberately an `ApiError` and not a `RuntimeError` carrying the number in
    its message: 🔴 the r5 rule reads the CLASS and the `status_code`
    attribute, so a double that models a fault by its MESSAGE models nothing
    about the thing under test and is blind to the rule the moment it lands.
    """

    def _raiser(*_a, **_kw):
        raise _api_error(status)

    return _raiser


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

    def test_an_incidental_keyerror_in_the_sdk_still_degrades(self):
        """🔴 @el review r2, FL-A. The narrow arm used to catch `LookupError`,
        which is the BASE class of `KeyError` and `IndexError`. The leg it
        wraps is a third-party SDK call chain, so ANY incidental container
        miss under `experiments.list` — deserialization, a dict lookup — was
        indistinguishable from our own data-contract signal and aborted the
        whole calibration build.

        ➡️ A guard stated over a base class inherits every meaning that class
        already had. The signal now has its own name.

        RED on `2d633ef`: this raised `KeyError` out of `collect_candidates`
        instead of returning the legacy candidates."""

        def _key_error(**_kw):
            raise KeyError("data")

        client = _client(legacy={"run-a": [("case-1", "t1")]})
        client.api.experiments.list = _key_error
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1"]

    def test_an_incidental_indexerror_in_the_sdk_still_degrades(self):
        """Same class, the other member — `IndexError` is what an SDK raises
        indexing an empty response list, so it is the likelier of the two."""

        def _index_error(**_kw):
            raise IndexError("list index out of range")

        client = _client(legacy={"run-a": [("case-1", "t1")]})
        client.api.experiments.list = _index_error
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1"]

    def test_our_own_signal_is_still_loud(self):
        """The converse of the two rows above: narrowing the arm must not also
        let a genuine data-contract violation degrade. A dataset name that does
        not resolve still aborts the build."""
        client = _client(legacy={"run-a": [("case-1", "t1")]})
        client.api.datasets.get = lambda name: _Obj(id=None)
        with pytest.raises(CalibrationDataError):
            collect_candidates(client, ["books_v1"], "run-")


class TestSystematicFaultAtTheLegsEntryPoint:
    """🔴 @el review r6 FL-1 (Codex `r3836481943`).

    The r5 classifier went in at the per-run loop inside
    `collect_experiment_candidates`. But that function OPENS with two un-`try`'d
    external calls — `resolve_dataset_id()` (`datasets.get`) and
    `experiments.list` — so neither reaches it. An `ApiError` from either is not
    a `CalibrationDataError`, so it fell to `collect_candidates`'s broad arm and
    collapsed the whole experiments leg to `[]` behind a warning.

    🔴 And the faults the classifier's own docstring names — an expired key
    (401), a revoked scope (403), the endpoint moving (404/405) — do not arrive
    at `list_items`. They arrive HERE, at the first call. So the primary rule
    was absent from exactly the path its motivating examples take. After
    2026-11-16 the legacy leg is empty by construction, which makes that a green
    build and an empty pack: this card's own failure, one call site above the
    fix for it.

    ➡️ *A rule installed at the site where a fault was OBSERVED does not cover
    the site where it ENTERS.* The fix is the leg's single choke point rather
    than a `try` per call, so the fifth call site added to this leg is covered
    before it is written.

    RED on `4fc05a3`: every raising row below returned the legacy candidates.
    """

    def test_a_401_from_experiments_list_aborts_rather_than_emptying_the_leg(self):
        client = _client(legacy={"run-a": [("case-1", "t1")]})
        client.api.experiments.list = _raise_api_error(401)
        with pytest.raises(CalibrationDataError, match="request-shaped fault"):
            collect_candidates(client, ["books_v1"], "run-")

    @pytest.mark.parametrize("status", [400, 403, 404, 405])
    def test_every_request_shaped_status_aborts_at_the_entry_point(self, status):
        """The rule is the CLASS, not the one status. 400 is the narrowed-field
        spelling, 403 a revoked scope, 404/405 the endpoint moving."""
        client = _client(legacy={"run-a": [("case-1", "t1")]})
        client.api.experiments.list = _raise_api_error(status)
        with pytest.raises(CalibrationDataError):
            collect_candidates(client, ["books_v1"], "run-")

    def test_the_dataset_lookup_is_covered_too_not_just_the_list_call(self):
        """🔴 `resolve_dataset_id` runs BEFORE `experiments.list`, so on an
        expired key it is the call that fails first. A fix wrapping only
        `experiments.list` passes every row above and still loses this one —
        which is the whole reason @el asked for the choke point."""
        client = _client(legacy={"run-a": [("case-1", "t1")]})
        client.api.datasets.get = _raise_api_error(401)
        with pytest.raises(CalibrationDataError, match="request-shaped fault"):
            collect_candidates(client, ["books_v1"], "run-")

    @pytest.mark.parametrize("status", [500, 502, 503, 429, 408])
    def test_the_converse_a_transient_fault_still_degrades_to_the_legacy_leg(self, status):
        """🔴 Without this the fix is satisfiable by raising on everything,
        which trades a fail-open for a build that cannot survive a flaky
        endpoint. 429 and 408 are the r5 carve-outs: 4xx by number, transient
        by meaning, and they must still degrade at this site too."""
        client = _client(legacy={"run-a": [("case-1", "t1")]})
        client.api.experiments.list = _raise_api_error(status)
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1"]

    def test_an_api_error_with_no_status_still_degrades_here(self):
        """`ApiError.status_code` is `Optional[int]` in the pinned SDK, and the
        entry point must read it the same way the loop does — one predicate,
        two call sites, no second interpretation."""
        client = _client(legacy={"run-a": [("case-1", "t1")]})

        def _no_status(**_kw):
            raise ApiError(body={"message": "no status"})

        client.api.experiments.list = _no_status
        got = collect_candidates(client, ["books_v1"], "run-")
        assert [c["trace_id"] for c in got] == ["t1"]

    def test_the_in_loop_check_survives_it_was_not_replaced(self):
        """The two checks do different jobs: this one decides whether the LEG
        exists, the in-loop one scopes a degrade to ONE run and feeds the
        `attempted >= 2` count rule. Moving the check up must not delete it —
        a 400 from `list_items` still aborts, with the per-run message."""
        client = _client(experiments={"run-a": [_experiment_item("case-1", "t1")]})
        client.api.experiments.list_items = _raise_api_error(400)
        with pytest.raises(CalibrationDataError, match="experiment run"):
            collect_candidates(client, ["books_v1"], "run-")


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
