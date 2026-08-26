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
        assert er.EXPERIMENT_DESCRIPTION == A.EXPERIMENT_DESCRIPTION
        assert er.ENVIRONMENT == A.ENVIRONMENT

    def test_the_experiment_environment_is_the_sdks_own(self):
        from langfuse._client.constants import LANGFUSE_SDK_EXPERIMENT_ENVIRONMENT

        assert er.EXPERIMENT_ENVIRONMENT == LANGFUSE_SDK_EXPERIMENT_ENVIRONMENT


def _eid(run_name, dataset_id="dataset-abc", dataset_name="itinerary_v1"):
    return er.experiment_id_for_run(
        run_name, dataset_id=dataset_id, dataset_name=dataset_name
    )


class TestExperimentIdentity:
    """The id is minted here, so every property the backend used to give us for
    free is now a row. Two of them pull in opposite directions -- stability
    across a retry, separation across datasets -- and the second was missing
    from the first version of this module (Codex, r1)."""

    def test_id_is_stable_for_one_run_name(self):
        # Two calls in two processes must agree, or a resumed run splits into
        # two half-populated experiments -- neither of them visibly wrong.
        assert _eid("eval_run_1") == _eid("eval_run_1")

    def test_id_differs_between_runs(self):
        assert _eid("eval_run_1") != _eid("eval_run_2")

    def test_id_has_the_shape_langfuse_documents(self):
        # "16 lowercase hexadecimal characters" -- SDK, _create_observation_id.
        assert re.fullmatch(r"[0-9a-f]{16}", _eid("eval_run_1"))

    def test_two_datasets_in_the_same_second_are_two_experiments(self):
        # 🔴 The row for the defect. `run_evaluation_on_dataset` is called once
        # per dataset in a loop and builds `eval_run_{...second...}_{version}`
        # with no dataset in it, so this collision is reachable rather than
        # theoretical -- and the merged experiment would carry two dataset ids
        # while looking like one healthy run.
        same_name = "eval_run_20260824_010203_v3"
        assert _eid(same_name, dataset_id="dataset-A") != _eid(
            same_name, dataset_id="dataset-B"
        )

    def test_the_dataset_name_separates_runs_when_the_id_is_absent(self):
        # `dataset_id` comes from `getattr(dataset, "id", None)`, so it can be
        # None. Without this the fix would hold only while the SDK object
        # happened to expose an id -- the degradation being silent, as usual.
        same_name = "eval_run_20260824_010203_v3"
        assert _eid(same_name, dataset_id=None, dataset_name="itinerary_v1") != _eid(
            same_name, dataset_id=None, dataset_name="place_to_book_v1"
        )

    def test_the_scope_join_is_unambiguous(self):
        # 🔴 The boundary control. Concatenating without a separator makes
        # ("ab", "c") and ("a", "bc") the same bytes; this pair reds the moment
        # `_SCOPE_SEP` is dropped and passes for no other reason.
        assert er.experiment_id_for_run(
            "run", dataset_id="ab", dataset_name="c"
        ) != er.experiment_id_for_run("run", dataset_id="a", dataset_name="bc")

    def test_the_formula_does_not_branch_on_a_missing_scope(self):
        # Both absent is a legal call, not a crash -- and it must still be
        # stable, because a tool that cannot name its dataset still has to keep
        # its own items together.
        bare = er.experiment_id_for_run("r", dataset_id=None, dataset_name=None)
        assert re.fullmatch(r"[0-9a-f]{16}", bare)
        assert bare == er.experiment_id_for_run("r", dataset_id=None, dataset_name=None)

    def test_the_dataset_scope_cannot_be_omitted_by_a_future_caller(self):
        # 🔴 The anti-regression control, and the reason both parameters are
        # required rather than defaulted: with defaults, a new call site that
        # forgot them would silently get the name-only identity back -- the
        # defect returning through the door marked convenience, green all the
        # way. This row is what makes that a TypeError instead.
        with pytest.raises(TypeError):
            er.experiment_id_for_run("eval_run_1")


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
            dataset_name="itinerary_v1",
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
            "eval_run_20260824_010203_v3",
            dataset_id="dataset-abc",
            dataset_name="itinerary_v1",
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
            dataset_name="local_atmosphere_v1",
            dataset_item_id="i1",
        )
        assert span._otel_span.attributes == returned
        assert returned[er.EXPERIMENT_ITEM_ROOT_OBSERVATION_ID] == span.id

    def test_the_description_is_set_when_the_caller_has_one(self):
        span = _FakeSpan()
        returned = er.link_experiment_item(
            span,
            run_name="la_eval_1",
            run_metadata=None,
            dataset_id="d1",
            dataset_name="local_atmosphere_v1",
            dataset_item_id="i1",
            run_description="local-atmosphere eval of local_atmosphere_v1",
        )
        assert returned[er.EXPERIMENT_DESCRIPTION] == (
            "local-atmosphere eval of local_atmosphere_v1"
        )

    def test_the_key_is_absent_rather_than_empty_when_it_is_not(self):
        # An OTel attribute set to "" is a value, and a dashboard renders it as
        # a blank label rather than as no label. Omission is the honest shape.
        span = _FakeSpan()
        returned = er.link_experiment_item(
            span,
            run_name="la_eval_1",
            run_metadata=None,
            dataset_id="d1",
            dataset_name="local_atmosphere_v1",
            dataset_item_id="i1",
        )
        assert er.EXPERIMENT_DESCRIPTION not in returned


@pytest.fixture(scope="module")
def sdk_span():
    """One REAL `LangfuseSpan` from the installed SDK, exported in-memory.

    No network: the client is handed an `InMemorySpanExporter`, so nothing is
    sent anywhere and the exported span is readable in-process.
    """
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-lf-mys951-accessor-row",
        secret_key="sk-lf-mys951-accessor-row",
        host="http://127.0.0.1:1",
        span_exporter=exporter,
        flush_at=1,
        flush_interval=0.1,
    )
    yield client, exporter


class TestTheAccessorIsPinnedToTheInstalledSdk:
    """🔴 FL-1 (@el, r1) — the doctrine that protects the KEYS now covers the
    ACCESSOR, which is the line that actually writes.

    `link_experiment_item` ends at `span._otel_span.set_attributes(...)`: a
    private SDK attribute. The keys are pinned by importing the private module
    *from the test*, so a rename reds in CI. `_otel_span` had none of that --
    `_FakeSpan` below defines its own, so every row above passes on a shape
    nobody had checked against the installed SDK. The suite was modelling the
    coupling instead of pinning it.

    `langfuse` is pinned at 4.14.0, so this cannot break on its own; it breaks
    on a deliberate bump, a bump runs CI, and CI had nothing that would see it.
    The first machine to notice would have been the runner, mid-eval, with spend
    committed -- the exact sentence the module docstring is written against.

    ➡️ *A test that defines the shape it depends on cannot report that the shape
    changed. Only the installed thing can testify about the installed thing.*
    """

    def test_a_real_sdk_span_exposes_the_accessor(self, sdk_span):
        client, _ = sdk_span
        span = client.start_observation(name="mys951-probe", as_type="span")
        try:
            assert hasattr(span, "_otel_span"), (
                "the installed langfuse span no longer exposes `_otel_span`; "
                "link_experiment_item's only write would raise at eval time"
            )
            assert callable(getattr(span._otel_span, "set_attributes", None))
        finally:
            span.end()

    def test_the_write_reaches_the_exported_span(self, sdk_span):
        """The end-to-end half: not just that the accessor exists, but that
        writing through it lands on the span the exporter actually ships."""
        client, exporter = sdk_span
        exporter.clear()
        span = client.start_observation(name="mys951-roundtrip", as_type="span")
        written = er.link_experiment_item(
            span,
            run_name="scheduled_eval_20260826",
            run_metadata={"evaluation_type": "scheduled"},
            dataset_id="dataset-abc",
            dataset_name="itinerary_v1",
            dataset_item_id="item-7",
            run_description="Scheduled evaluation of itinerary_v1",
        )
        span.end()
        client.flush()

        finished = exporter.get_finished_spans()
        assert len(finished) == 1, f"expected one exported span, got {len(finished)}"
        exported = dict(finished[0].attributes or {})

        # Vacuity control FIRST: `written` is the loop's own population, so a
        # helper that returned {} would make every assertion below true while
        # writing nothing at all. Name the keys that must be there.
        required = {
            er.ENVIRONMENT,
            er.EXPERIMENT_ID,
            er.EXPERIMENT_NAME,
            er.EXPERIMENT_DATASET_ID,
            er.EXPERIMENT_ITEM_ID,
            er.EXPERIMENT_ITEM_ROOT_OBSERVATION_ID,
            er.EXPERIMENT_DESCRIPTION,
        }
        assert required <= set(written), sorted(required - set(written))

        missing = sorted(k for k in written if k not in exported)
        assert not missing, f"written but never exported: {missing}"
        for key, value in written.items():
            assert exported[key] == value, key

    def test_the_class_alone_would_be_a_vacuous_check(self):
        """The negative control, and the reason the rows above build an instance.

        `_otel_span` is assigned in `__init__` (`self._otel_span = otel_span`),
        not declared on the class -- so `hasattr(LangfuseSpan, "_otel_span")` is
        False *today*, while the coupling is perfectly intact. A class-level row
        would have failed for the wrong reason, and the obvious repair for a red
        row of that shape is to delete it.
        """
        from langfuse._client.span import LangfuseSpan

        assert not hasattr(LangfuseSpan, "_otel_span")

    def test_the_public_surface_still_drops_a_raw_attribute(self, sdk_span):
        """@el asked: if a supported public route exists, prefer it. It does not
        -- and the way it does not is the dangerous kind.

        `LangfuseObservationWrapper.update(**kwargs)` accepts arbitrary keywords
        and its own docstring says they are ignored. So the public call is not
        merely unavailable: it SUCCEEDS and writes nothing. A future SDK that
        starts honouring it reds this row, and the choice gets revisited on
        purpose rather than by drift -- the `TestWhyNotRunExperiment` shape.
        """
        client, exporter = sdk_span
        exporter.clear()
        span = client.start_observation(name="mys951-public-route", as_type="span")
        span.update(**{er.EXPERIMENT_ID: "0123456789abcdef"})
        span.end()
        client.flush()

        exported = dict((exporter.get_finished_spans()[0].attributes) or {})
        assert exported.get(er.EXPERIMENT_ID) != "0123456789abcdef", (
            "langfuse.update() now honours raw attribute keywords -- prefer the "
            "public route over span._otel_span and update ADR #28"
        )


def _link_call_keywords(source: str):
    """Every `link_experiment_item(...)` call in `source`, as keyword-name sets."""
    calls = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "link_experiment_item"
        ):
            calls.append({kw.arg for kw in node.keywords if kw.arg})
    return calls


class TestEveryCallSitePassesTheDatasetScope:
    """🔴 The required keyword is only enforced where the call RUNS.

    `experiment_id_for_run`'s mandatory arguments raise a TypeError -- but the
    four call sites sit inside `finally:` blocks in async loops that no unit
    test executes, so the first machine to notice a forgotten `dataset_name=`
    would be the runner, mid-eval, with spend already committed. The signature
    is the rule; this is the row that reads the call sites without running them.

    ➡️ *A required argument is checked at the call, and a call nobody executes
    is not checked at all.*
    """

    def test_the_reader_is_not_vacuous(self):
        seen = sum(
            len(_link_call_keywords((TOOLS / name).read_text()))
            for name in EVAL_WRITERS
        )
        assert seen >= len(EVAL_WRITERS), f"found {seen} link_experiment_item calls"

    @pytest.mark.parametrize("name", EVAL_WRITERS)
    def test_the_writer_names_its_dataset(self, name):
        for keywords in _link_call_keywords((TOOLS / name).read_text()):
            assert "dataset_id" in keywords, name
            assert "dataset_name" in keywords, name

    @pytest.mark.parametrize("name", EVAL_WRITERS)
    def test_the_writer_names_its_run_description(self, name):
        """The label each tool used to pass to the retired write.

        `run_description` is optional in the signature (a caller with nothing to
        say is legitimate), so nothing at the call would catch a writer that
        quietly stopped sending one -- which is exactly how it went missing in
        r1: the parameter disappeared with the endpoint, and a dropped label is
        invisible in a diff and only visible in a dashboard.
        """
        for keywords in _link_call_keywords((TOOLS / name).read_text()):
            assert "run_description" in keywords, name

    def test_the_reader_sees_a_missing_keyword(self):
        # The negative control: the same reader over a call that omits it.
        offender = "link_experiment_item(span, run_name=n, dataset_id=d)"
        assert _link_call_keywords(offender) == [{"run_name", "dataset_id"}]


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
