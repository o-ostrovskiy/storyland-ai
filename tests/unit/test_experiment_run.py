"""MYS-951 — the eval write path is an EXPERIMENT, not a dataset-run item.

The rows here fall into three groups, and the middle one is the point:

* the attribute keys are ours, so they are pinned against the installed SDK;
* what we WRITE is what `judge_calibration` (PR1's read union) READS -- the two
  halves of MYS-909 have never been exercised against each other, because the
  experiments leg has always been empty;
* the class guard: no file in this tree may CALL the retired endpoint.
"""

import ast
import re
from pathlib import Path

import pytest

from evaluation.tools import experiment_run as er
from evaluation.tools.judge_calibration import (
    _EVAL_ID_GROUPS,
    _EVAL_ID_GROUP_OBSERVED,
    experiment_item_case_id,
)

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "evaluation" / "tools"
EVAL_WRITERS = [
    "run_scheduled_eval.py",
    "run_local_atmosphere_eval.py",
    "run_place_to_book_eval.py",
    "run_expansion_eval.py",
]


class TestAttributeKeysMatchTheSdk:
    """The keys are spelled out in our source; the SDK owns their meaning.

    Owning the strings keeps a private-module reorganisation from crashing an
    eval mid-run with spend already committed. The cost is that a rename in the
    SDK would silently stop linking -- these rows are that cost being paid.
    """

    def test_every_key_equals_the_sdk_constant(self):
        from langfuse._client.attributes import LangfuseOtelSpanAttributes as A

        assert er.EXPERIMENT_ID == A.EXPERIMENT_ID
        assert er.EXPERIMENT_NAME == A.EXPERIMENT_NAME
        assert er.EXPERIMENT_METADATA == A.EXPERIMENT_METADATA
        assert er.EXPERIMENT_DATASET_ID == A.EXPERIMENT_DATASET_ID
        assert er.EXPERIMENT_ITEM_ID == A.EXPERIMENT_ITEM_ID
        assert er.EXPERIMENT_ITEM_METADATA == A.EXPERIMENT_ITEM_METADATA
        assert (
            er.EXPERIMENT_ITEM_ROOT_OBSERVATION_ID
            == A.EXPERIMENT_ITEM_ROOT_OBSERVATION_ID
        )
        assert er.ENVIRONMENT == A.ENVIRONMENT

    def test_the_experiment_environment_is_the_sdks_own(self):
        from langfuse._client.constants import LANGFUSE_SDK_EXPERIMENT_ENVIRONMENT

        assert er.EXPERIMENT_ENVIRONMENT == LANGFUSE_SDK_EXPERIMENT_ENVIRONMENT


class TestExperimentIdentity:
    def test_id_is_stable_for_one_run_name(self):
        # Two calls in two processes must agree, or a resumed run splits into
        # two half-populated experiments -- neither of them visibly wrong.
        assert er.experiment_id_for_run("eval_run_1") == er.experiment_id_for_run(
            "eval_run_1"
        )

    def test_id_differs_between_runs(self):
        assert er.experiment_id_for_run("eval_run_1") != er.experiment_id_for_run(
            "eval_run_2"
        )

    def test_id_has_the_shape_langfuse_documents(self):
        # "16 lowercase hexadecimal characters" -- SDK, _create_observation_id.
        assert re.fullmatch(r"[0-9a-f]{16}", er.experiment_id_for_run("eval_run_1"))


def _regroup(attributes: dict) -> dict:
    """Flat OTel attributes -> the grouped shape the read API hands back.

    ⚠️ This is the MODELLED step in the round-trip below and it is named rather
    than hidden: the backend's flat-key -> group-name mapping is not something
    this repo can execute. What is not modelled is which group the read side
    looks in, and that half is established -- the Local Runner's 2026-08-20
    probe found `eval_id` in `experiment_item_metadata`.
    """
    groups = {
        er.EXPERIMENT_ITEM_METADATA: "experiment_item_metadata",
        er.EXPERIMENT_METADATA: "experiment_metadata",
    }
    out = {name: {} for name in groups.values()}
    for key, value in attributes.items():
        for prefix, name in groups.items():
            if key.startswith(prefix + "."):
                out[name][key[len(prefix) + 1 :]] = value
    return out


class _FakeExperimentItem:
    def __init__(self, grouped):
        self.id = "experiment-item-scoped-id"
        self.metadata = grouped.get("metadata")
        self.experiment_item_metadata = grouped.get("experiment_item_metadata")
        self.experiment_metadata = grouped.get("experiment_metadata")


class TestTheWriteIsWhatPr1Reads:
    """🔴 The contract row. PR1's experiments leg has never seen real data.

    `experiment_item_case_id` RAISES when it cannot find `eval_id`, and that
    raise propagates and aborts a calibration build. So "the writer puts it
    somewhere" is not enough: it has to be in a group the reader asks for.
    """

    def _attrs(self):
        return er.experiment_item_attributes(
            run_name="eval_run_20260824_010203_v3",
            run_metadata={"dataset_name": "itinerary_v1", "prompt_version": "v3"},
            dataset_id="dataset-abc",
            dataset_item_id="dataset-item-42",
            root_observation_id="0123456789abcdef",
        )

    def test_the_reader_recovers_the_dataset_item_id(self):
        item = _FakeExperimentItem(_regroup(self._attrs()))
        assert experiment_item_case_id(item) == "dataset-item-42"

    def test_eval_id_is_in_the_group_the_live_probe_observed(self):
        # Not merely "in one of the three". The other two are read as a hedge
        # against the value moving; this is the one it was seen in.
        assert _EVAL_ID_GROUP_OBSERVED in _EVAL_ID_GROUPS
        grouped = _regroup(self._attrs())
        assert grouped[_EVAL_ID_GROUP_OBSERVED]["eval_id"] == "dataset-item-42"

    def test_run_metadata_does_not_carry_the_eval_id(self):
        # 🔴 The negative control, and the mistake it refuses is the plausible
        # one: putting eval_id in the RUN metadata reads fine, groups into
        # `experiment_metadata`, and would make the row above pass for the
        # wrong reason while the observed group stayed empty.
        grouped = _regroup(self._attrs())
        assert "eval_id" not in grouped["experiment_metadata"]

    def test_identity_and_environment_are_set(self):
        attrs = self._attrs()
        assert attrs[er.EXPERIMENT_NAME] == "eval_run_20260824_010203_v3"
        assert attrs[er.EXPERIMENT_ID] == er.experiment_id_for_run(
            "eval_run_20260824_010203_v3"
        )
        assert attrs[er.EXPERIMENT_DATASET_ID] == "dataset-abc"
        assert attrs[er.EXPERIMENT_ITEM_ID] == "dataset-item-42"
        assert attrs[er.EXPERIMENT_ITEM_ROOT_OBSERVATION_ID] == "0123456789abcdef"
        assert attrs[er.ENVIRONMENT] == er.EXPERIMENT_ENVIRONMENT

    def test_every_attribute_value_is_an_otel_scalar(self):
        # A dict as an attribute value is dropped by the OTel SDK with a
        # warning, not an error -- so this would be a silent half-write.
        for key, value in self._attrs().items():
            assert isinstance(value, (str, int, float, bool)), key


class _FakeOtel:
    def __init__(self):
        self.attributes = None

    def set_attributes(self, attributes):
        self.attributes = attributes


class _FakeSpan:
    def __init__(self):
        self.id = "fedcba9876543210"
        self._otel_span = _FakeOtel()


class TestLinkExperimentItem:
    def test_it_sets_the_attributes_on_the_span(self):
        span = _FakeSpan()
        returned = er.link_experiment_item(
            span,
            run_name="la_eval_1",
            run_metadata={"evaluation_type": "local_atmosphere"},
            dataset_id="d1",
            dataset_item_id="i1",
        )
        assert span._otel_span.attributes == returned
        assert returned[er.EXPERIMENT_ITEM_ROOT_OBSERVATION_ID] == span.id


def _references_retired_write(source: str) -> bool:
    """True iff the source REACHES `<...>.dataset_run_items.create` at all.

    🔴 AST, not grep, and the difference is load-bearing in this very PR:
    `experiment_run.py` and `judge_calibration.py` both NAME the retired
    endpoint in prose, to explain why it is gone and what still has to migrate.
    A textual guard would red on the documentation of its own subject, and the
    fix a hurried reader reaches for is deleting the explanation.
    ➡️ *A guard against code must parse; a guard against a WORD cannot tell the
    call from the sentence about it.*

    🔴 And it matches the ATTRIBUTE, not the call. The first version of this
    guard looked for a `Call` whose callee was `….dataset_run_items.create`,
    and it read the langfuse SDK's own `run_experiment` as clean -- because
    that code passes the method as a VALUE: `asyncio.to_thread(
    self.api.dataset_run_items.create, …)`. The endpoint is invoked and the
    callee never appears in call position. ➡️ *A function can be invoked
    without ever being the callee -- `to_thread`, `partial`, `submit` and every
    scheduler take it as an argument.* A call-shaped guard is a guard against
    one syntax, not against the dependency.
    """
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "create"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "dataset_run_items"
        ):
            return True
    return False


class TestNoEvalToolCallsTheRetiredWrite:
    def test_the_detector_is_not_vacuous(self):
        # Without this row the class guard below passes on a detector that
        # never matches anything -- which is how a guard retires itself.
        assert _references_retired_write(
            "langfuse.api.dataset_run_items.create(run_name='x')"
        )
        assert not _references_retired_write("x = 1")

    def test_the_detector_sees_it_passed_as_a_value(self):
        # 🔴 The form that defeated the first version of this guard, and the
        # form the SDK itself uses. Invoked, never in call position.
        assert _references_retired_write(
            "await asyncio.to_thread(self.api.dataset_run_items.create, run_name='x')"
        )
        assert _references_retired_write(
            "f = functools.partial(client.api.dataset_run_items.create)"
        )

    def test_a_mention_in_prose_is_not_a_reference(self):
        assert not _references_retired_write(
            '"""We used to call dataset_run_items.create here."""\nx = 1\n'
        )

    @pytest.mark.parametrize("name", EVAL_WRITERS)
    def test_writer_does_not_call_it(self, name):
        assert not _references_retired_write((TOOLS / name).read_text())

    def test_no_file_in_the_tree_calls_it(self):
        # The CLASS gate: a fifth tool added later is covered by existing,
        # not by being remembered. Scoped to evaluation/ because that is where
        # dataset runs are written; a caller anywhere else would be a new
        # architecture, not a regression.
        offenders = [
            str(p.relative_to(REPO))
            for p in (REPO / "evaluation").rglob("*.py")
            if _references_retired_write(p.read_text())
        ]
        assert offenders == []


class TestWhyNotRunExperiment:
    """The SDK's own runner is disqualified by AC-1, and that is a fact about
    the installed version rather than a preference. When it stops being true,
    this row reds and the choice can be revisited on purpose."""

    def test_sdk_run_experiment_still_calls_the_retired_endpoint(self):
        import langfuse._client.client as sdk_client

        assert _references_retired_write(Path(sdk_client.__file__).read_text()), (
            "langfuse's own run_experiment no longer calls dataset_run_items.create "
            "-- re-evaluate MYS-951's choice of the raw-attribute path"
        )
