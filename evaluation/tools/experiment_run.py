"""Write eval dataset runs as Langfuse **Experiments**, via OTel span attributes.

MYS-951 / PR2 of MYS-909. Langfuse retires ``POST /dataset-run-items`` on
**2026-11-16**, six days after launch, and it is the only way the four eval
tools have ever recorded a run. This module is the replacement, and it is one
implementation rather than four so the write contract has a single home --
``judge_calibration.py`` reads that contract from the other side.

-- Why the raw-OTel path and not ``dataset.run_experiment`` ------------------

The card offered two replacements. Only one of them can satisfy AC-1 ("no
remaining references to ``dataset_run_items.create``"), and it is not the
obvious one:

🔴 **The SDK's own experiment runner still calls the deprecated endpoint.** At
the pin this repo installs (``langfuse==4.14.0``) ``Langfuse.run_experiment``
reaches ``self.api.dataset_run_items.create`` for every dataset-backed item
(``_client/client.py``) -- and it does so as ``asyncio.to_thread(that_method,
…)``, so the method is invoked while never appearing in call position.
Migrating onto ``run_experiment`` would have moved the legacy write one layer
down, out of our repo and out of reach of the guard: AC-1's grep would go green
over a codebase that still writes through the endpoint being retired.
➡️ *A guard over our own source is a claim about our source, not about the
call.* ``tests/unit/test_experiment_run.py`` asserts this against the INSTALLED
SDK, so the day it stops being true the row reds and the choice gets revisited
on purpose rather than by drift.

The second path -- ``propagate_attributes()`` -- has no experiment parameter at
all in its public signature, which leaves the attributes themselves.

-- The one assumption this PR cannot verify in-sandbox ----------------------

⚠️ In the SDK the experiment's identity comes FROM the legacy write:
``experiment_id = dataset_run_id or fallback_experiment_id``, where
``dataset_run_id`` is what ``dataset_run_items.create`` returned. Dropping that
call means we mint the id ourselves, and whether the backend joins a
self-minted experiment to its dataset is a question only a live run answers.
That is exactly AC-2/AC-3, and they are G4-spend gated. Nothing here claims
they hold; the code is written so a single run settles it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional

# 🔴 Spelled out here rather than imported from ``langfuse._client.attributes``
# on purpose. That module is private, so importing it makes an SDK
# reorganisation a crash at eval time -- on the machine, mid-run, with spend
# already committed. Owning the strings makes the coupling explicit, and
# ``TestAttributeKeysMatchTheSdk`` asserts every one of them against the
# installed SDK, so a rename reds in CI instead of at 3am on the runner.
EXPERIMENT_ID = "langfuse.experiment.id"
EXPERIMENT_NAME = "langfuse.experiment.name"
EXPERIMENT_METADATA = "langfuse.experiment.metadata"
EXPERIMENT_DATASET_ID = "langfuse.experiment.dataset.id"
EXPERIMENT_ITEM_ID = "langfuse.experiment.item.id"
EXPERIMENT_ITEM_METADATA = "langfuse.experiment.item.metadata"
EXPERIMENT_ITEM_ROOT_OBSERVATION_ID = "langfuse.experiment.item.root_observation_id"
ENVIRONMENT = "langfuse.environment"

# The SDK sets this environment on every experiment item, and its propagation
# layer FORCES it whenever the root-observation attribute is present -- so it is
# part of the shape, not a flavour we chose.
# ⚠️ Consequence worth stating: eval traces move from the ambient environment
# into `sdk-experiment`. Any saved Langfuse view filtered on the old value stops
# showing eval runs. Flagged in the PR rather than discovered in a dashboard.
EXPERIMENT_ENVIRONMENT = "sdk-experiment"

# Langfuse observation ids are "16 lowercase hexadecimal characters" (SDK
# docstring for `_create_observation_id`). The experiment id shares that space,
# so a minted one has to match or the backend has every right to refuse it.
_ID_HEX_CHARS = 16


def experiment_id_for_run(run_name: str) -> str:
    """A stable 16-hex experiment id for one run, derived from its name.

    Derived rather than random for one reason: every item of a run must land on
    the SAME experiment, and the four tools build their items in loops that can
    be retried. A random id would be minted per process, so a resumed run would
    silently split into two experiments -- two half-populations, neither wrong
    on its face, which is the failure this fleet keeps meeting under other
    names. The run names already carry a timestamp, so deriving from the name
    is unique per run without being unique per attempt.
    """
    return hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:_ID_HEX_CHARS]


def _flatten(prefix: str, values: Optional[Mapping[str, Any]]) -> dict:
    """One OTel attribute per key, values coerced to strings.

    OTel attributes are scalars, so a nested dict cannot be an attribute value.
    The SDK flattens the same way (`{span_key}.{k}`), and the READ side depends
    on that spelling: `judge_calibration.experiment_item_case_id` looks up
    `eval_id` inside the `experiment_item_metadata` group.
    """
    if not values:
        return {}
    flat = {}
    for key, value in values.items():
        if value is None:
            continue
        flat[f"{prefix}.{key}"] = (
            value if isinstance(value, str) else json.dumps(value, default=str)
        )
    return flat


def experiment_item_attributes(
    *,
    run_name: str,
    run_metadata: Optional[Mapping[str, Any]],
    dataset_id: Optional[str],
    dataset_item_id: str,
    root_observation_id: str,
    item_metadata: Optional[Mapping[str, Any]] = None,
) -> dict:
    """The full attribute set that makes one root span an experiment item.

    🔴 `eval_id` is not optional and is not decoration. `ExperimentItem` exposes
    no dataset-item id of its own -- its `experiment_item_id` is run-scoped, so
    the same evalset case gets a different id in every run -- and
    `judge_calibration.experiment_item_case_id` therefore reads the dataset-item
    id out of the item metadata under this exact key, raising when it is absent.
    ⚠️ It is put in the ITEM metadata specifically: the live probe (2026-08-20)
    found `eval_id` in `experiment_item_metadata` only. Putting it in the span's
    ordinary `metadata=` -- where the tools put it today -- lands it in TRACE
    metadata, a different group, and PR1's experiments leg would abort on every
    item this PR writes.
    """
    attributes = {
        ENVIRONMENT: EXPERIMENT_ENVIRONMENT,
        EXPERIMENT_ID: experiment_id_for_run(run_name),
        EXPERIMENT_NAME: run_name,
        EXPERIMENT_ITEM_ID: dataset_item_id,
        EXPERIMENT_ITEM_ROOT_OBSERVATION_ID: root_observation_id,
    }
    if dataset_id:
        attributes[EXPERIMENT_DATASET_ID] = dataset_id
    attributes.update(_flatten(EXPERIMENT_METADATA, run_metadata))
    attributes.update(
        _flatten(
            EXPERIMENT_ITEM_METADATA,
            {**(item_metadata or {}), "eval_id": dataset_item_id},
        )
    )
    return attributes


def link_experiment_item(
    span: Any,
    *,
    run_name: str,
    run_metadata: Optional[Mapping[str, Any]],
    dataset_id: Optional[str],
    dataset_item_id: str,
    item_metadata: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Mark `span` as one item of the `run_name` experiment. Returns what it set.

    Replaces the `langfuse.api.dataset_run_items.create(...)` call that used to
    sit in each tool's `finally:` block, and keeps its position there: a case
    that raised is still a case that RAN, and dropping it from the run would
    make a failing evalset look smaller rather than worse.
    """
    attributes = experiment_item_attributes(
        run_name=run_name,
        run_metadata=run_metadata,
        dataset_id=dataset_id,
        dataset_item_id=dataset_item_id,
        root_observation_id=span.id,
        item_metadata=item_metadata,
    )
    span._otel_span.set_attributes(attributes)
    return attributes
