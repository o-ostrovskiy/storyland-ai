"""
Scheduled evaluation runner for StoryLand AI.

This script runs automated evaluations against the Langfuse dataset and logs results
for tracking quality over time. Designed to be run via cron or GitHub Actions.
"""

import os
import sys
import json
import time
import random
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.logging import get_logger, configure_logging
from common.config import load_config
from core.extraction import downgrade_ungrounded_match_types, validate_composer_envelope
from core.prompts import build_composition_prompt
from core.retry import build_retry_options
from core.session_state import SessionStateAccessor, SessionStateKeys
from services.session_service import create_session_service
from core.executor import DISCOVERY_RESEARCHER_AUTHORS, RESEARCHER_PAYLOAD_KEYS
from agents.orchestrator import (
    create_book_to_place_discovery_workflow,
    create_book_to_place_composition_workflow,
)
from agents.prompts import load_prompts, CURRENT_PROMPT_VERSION, AgentPrompts
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import LoggingPlugin
from plugins.langfuse_plugin import LangfusePlugin
from google.genai import types
import uuid

# LLM-as-judge scorer (Issue #96)
from evaluation.tools.llm_scorer import score_itinerary, score_criteria_coverage
from evaluation.tools.experiment_run import link_experiment_item

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

logger = get_logger("storyland.scheduled_eval")


# Dedicated runner per non-itinerary flow, for the routing log message.
DEDICATED_RUNNERS = {
    "place_to_book": "evaluation/tools/run_place_to_book_eval.py",
    "local_atmosphere": "evaluation/tools/run_local_atmosphere_eval.py",
    "expansion": "evaluation/tools/run_expansion_eval.py",
}


def count_itinerary_cities(itinerary_data: Optional[Dict[str, Any]]) -> int:
    """Cities in a composer payload, whichever shape it arrives in.

    The composer emits a ComposerEnvelope — ``{"itinerary": {"cities": [...]},
    "suggestions": [...]}`` — so the top-level ``["cities"]`` read this replaced
    returned 0 for every successful run, including the 11-city itineraries sat
    right beside it in the same result file. The count is reported to Langfuse
    and to the summary, so "0 cities" read as a broken composition when nothing
    was wrong.

    Prefers ``validate_composer_envelope`` (the same validator the executor
    uses) and falls back to a plain nested read so a payload that fails schema
    validation still counts rather than silently reporting zero.
    """
    if not itinerary_data:
        return 0
    envelope = validate_composer_envelope(itinerary_data)
    if envelope is not None:
        return len(envelope[0].get("cities") or [])
    inner = itinerary_data.get("itinerary")
    if isinstance(inner, dict):
        return len(inner.get("cities") or [])
    return len(itinerary_data.get("cities") or [])


def unverified_payload_keys(langfuse_plugin, present_keys) -> List[str]:
    """Discovery payload keys that no positive search receipt vouches for.

    The same derivation ``WorkflowExecutor._unverified_discovery_keys`` makes in
    production, so the eval's session state carries the same fact production's
    does. Without it the eval scores an itinerary production would have
    demoted, and the artifact stops describing the deployed system precisely in
    the unsearched-researcher scenario this metric exists to measure.

    🔴 r4 — and "the same derivation" is a claim that has to be re-earned every
    time production's changes. This function asked ``unsearched_agents``, which
    omits an agent the ledger never observed; when the production path moved to
    positive receipts over PRESENT payloads, this one silently became the more
    permissive of the two and the docstring above went from true to false
    without a line of it changing. Fixing a class means sweeping every site
    that makes the inference, not the one review pointed at.

    ``present_keys`` is the caller's ``SessionStateAccessor.present_discovery_keys``
    — required rather than defaulted, because a default here is exactly how the
    two derivations drift apart again.
    """
    searched = langfuse_plugin.searched_agents(RESEARCHER_PAYLOAD_KEYS)
    return sorted(
        key
        for name, key in RESEARCHER_PAYLOAD_KEYS.items()
        if key in present_keys and name not in searched
    )


def apply_production_grounding_downgrade(itinerary_data, grounding_state):
    """Apply production's match-type downgrade to a composed eval itinerary.

    The eval extracts JSON straight out of the composer's text and therefore
    bypasses ``extract_itinerary_from_response``, which is where production
    demotes literal/historical stops that no searched researcher supports. The
    judge then scored claims the product would never have shipped.

    Mutates and returns ``itinerary_data`` (envelope or bare itinerary, both
    shapes occur -- see ``count_itinerary_cities``).

    r3: both ways this can quietly do nothing now say so. A dict that is
    neither an envelope nor a bare itinerary used to fall through to a
    zero-city loop and return untouched, so the results file could not
    distinguish "gated, nothing to change" from "shape unrecognised, nothing
    gated" -- which is exactly the eval-stopped-representing-production
    failure this function was added to fix, surviving as a property of the
    fix. And a session whose fail-closed pass never ran fails the guard open;
    that is the right default for an eval, but it must not be silent.
    """
    if not isinstance(itinerary_data, dict):
        return itinerary_data
    inner = itinerary_data.get("itinerary")
    payload = inner if isinstance(inner, dict) else itinerary_data
    if not isinstance(payload.get("cities"), list):
        logger.warning(
            "eval_grounding_downgrade_shape_unrecognised",
            keys=",".join(sorted(k for k in payload if isinstance(k, str))[:10]),
        )
        return itinerary_data
    if not grounding_state.discovery_verification_ran:
        logger.warning(
            "eval_grounding_downgrade_no_verdict",
            reason="unverified_discovery key absent; guard fails open",
        )
    downgrade_ungrounded_match_types(
        payload,
        grounding_state.grounding_research_text,
        grounding_state.all_discovery_unverified,
    )
    return itinerary_data


def summarize_search_grounding(langfuse_plugin) -> Dict[str, Any]:
    """Which discovery researchers actually called google_search (MYS-817).

    A DETERMINISTIC fact, reported beside the judge's scores rather than
    folded into them. The judge grades itinerary quality and has no way to
    tell a searched answer from a remembered one — which is precisely how
    MYS-816 went unnoticed: a researcher that skips the search still returns a
    plausible itinerary that scores fine.

    Report-only for now, deliberately. Skips are currently near-universal
    (~1-2 of 4 researchers on most runs), so a hard gate would be red on every
    run from day one and would simply get switched off. Turn `grounded` vs
    `total` into a pass rule once MYS-816's fix has driven it green.

    THREE-WAY, not two-way. `researchers_grounded` counts only researchers
    POSITIVELY observed calling google_search; a researcher the ledger never
    saw at all is reported as `unobserved`, never silently folded into either
    side. Deriving it as `total - len(unsearched)` was the bug this docstring
    used to describe as a feature: `unsearched_agents` omits a never-observed
    agent by design, so an empty ledger — a broken observation seam — reported
    a clean 4/4, and a partial ledger inflated the number the same way. That is
    the MYS-492 class the enforcement path already had to fix: "everything was
    verified" must be a measurement, never an inference from an absence.

    Invariant: grounded + unsearched + unobserved == total.
    """
    searched = sorted(langfuse_plugin.searched_agents(DISCOVERY_RESEARCHER_AUTHORS))
    unsearched = sorted(langfuse_plugin.unsearched_agents(DISCOVERY_RESEARCHER_AUTHORS))
    accounted = set(searched) | set(unsearched)
    unobserved = sorted(a for a in DISCOVERY_RESEARCHER_AUTHORS if a not in accounted)
    return {
        "researchers_total": len(DISCOVERY_RESEARCHER_AUTHORS),
        "researchers_grounded": len(searched),
        "unsearched": unsearched,
        "unobserved": unobserved,
    }


def aggregate_search_grounding(
    case_results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Roll the per-case grounding facts up to a dataset-level view.

    Returns None when no case carried the field (an older results file, or a
    run that failed before discovery), so the summary can omit the block
    rather than print a misleading all-zero one.

    `cases_fully_grounded` is `grounded == total`, NOT "nothing in `unsearched`".
    The second form counts a case whose ledger observed nobody at all as fully
    grounded — the same absence-as-evidence inversion `summarize_search_grounding`
    fixes one level down.

    ⚠️ Forward-only, and the previous version of this docstring overclaimed
    (r3). It does not crash on a results file written before `unobserved`
    existed, but those files carry a `researchers_grounded` produced by the
    old `total - len(unsearched)` subtraction -- so a broken-ledger case was
    serialised as a clean 4/4, and `grounded == total` still counts it fully
    grounded. Old numbers cannot be repaired from here; only runs recorded
    after `summarize_search_grounding` became positive-receipt-only are
    trustworthy.
    """
    present = [c["search_grounding"] for c in case_results if c.get("search_grounding")]
    if not present:
        return None

    def by_frequency(counts: Dict[str, int]) -> Dict[str, int]:
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    offenders: Dict[str, int] = {}
    unobserved_counts: Dict[str, int] = {}
    for entry in present:
        for agent in entry.get("unsearched") or []:
            offenders[agent] = offenders.get(agent, 0) + 1
        for agent in entry.get("unobserved") or []:
            unobserved_counts[agent] = unobserved_counts.get(agent, 0) + 1
    return {
        "cases": len(present),
        "cases_fully_grounded": sum(
            1
            for e in present
            if e.get("researchers_total")
            and e.get("researchers_grounded") == e.get("researchers_total")
        ),
        "researchers_grounded": sum(e.get("researchers_grounded", 0) for e in present),
        "researchers_total": sum(e.get("researchers_total", 0) for e in present),
        # Which researcher skips most often — the actionable number, since the
        # offender varies run to run rather than being one broken agent.
        "unsearched_by_agent": by_frequency(offenders),
        # Kept separate from the offenders: a researcher we never saw is an
        # instrumentation failure to chase, not a model that ignored its prompt.
        "unobserved_by_agent": by_frequency(unobserved_counts),
    }


def count_scored_cases(result: Dict[str, Any]) -> int:
    """Cases in a dataset result that actually carry judge scores."""
    return sum(1 for c in (result.get("case_results") or []) if c.get("scores"))


def dataset_failure_reason(result: Dict[str, Any]) -> Optional[str]:
    """Why this dataset result should fail CI, or None if it passes.

    "The judge scored nothing" is a failure in its own right (MYS-825). On
    2026-08-09 the judge model began answering 404; because each case caught
    its own scoring error, all 18 cases recorded ``scores: None`` and the run
    still exited 0 under "✅ All evaluations completed successfully". A quality
    gate that measured nothing must not report success — "no result" is the one
    outcome that looks identical to a passing one, so it has to be called out
    explicitly rather than inferred from the numbers nobody reads.

    An empty dataset (no cases at all) is NOT a failure; the caller reports
    that separately as a configuration state.
    """
    if "error" in result:
        return f"ERROR: {result['error']}"

    failed = result.get("failed_cases", 0)
    if failed:
        return (
            f"ERROR: {failed} case(s) failed evaluation "
            f"(out of {result.get('total_cases', 0)})"
        )

    skipped = result.get("skipped_cases", 0)
    if skipped:
        return f"ERROR: {skipped} case(s) skipped due to parsing failures"

    evaluated = result.get("evaluated_cases", 0)
    if evaluated > 0 and count_scored_cases(result) == 0:
        return (
            f"ERROR: the judge scored 0 of {evaluated} evaluated case(s) — "
            "this run measured nothing. Check the judge model in "
            "evaluation/tools/llm_scorer.py (_DEFAULT_JUDGE_MODEL); model "
            "retirements surface here as a per-case 404."
        )

    return None


def select_itinerary_datasets(
    datasets_info: Dict[str, Any],
) -> tuple[List[str], List[Dict[str, str]]]:
    """Split registry entries into itinerary datasets vs. routed-away ones.

    The scheduled runner only knows how to drive the itinerary (book→place)
    workflow; datasets registered with any other ``flow`` (place_to_book,
    local_atmosphere, expansion) have dedicated runners and must not be fed
    through this script — their cases would either all skip (failing CI) or,
    worse, run through the wrong flow and produce plausible-but-bogus judge
    scores. A missing ``flow`` means "itinerary" (registries written before
    the field existed contain only itinerary datasets).

    Returns:
        (selected_dataset_names, routed_entries) where each routed entry is
        ``{"dataset_name": ..., "flow": ...}``.
    """
    selected: List[str] = []
    routed: List[Dict[str, str]] = []
    for entry in datasets_info.get("datasets", []):
        flow = entry.get("flow", "itinerary")
        if flow == "itinerary":
            selected.append(entry["dataset_name"])
        else:
            routed.append({"dataset_name": entry["dataset_name"], "flow": flow})
    return selected, routed


def _summarize_by_shape(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate scored cases by preference shape.

    Returns {"with_preferences": {...}, "without_preferences": {...}}, each
    carrying n, mean average, and mean total tokens — the recorded artifact
    the per-shape gate and the mechanism check read.
    """
    shapes: Dict[str, Dict[str, Any]] = {}
    for shape_name, wants in (("with_preferences", True), ("without_preferences", False)):
        in_shape = [
            c for c in case_results
            if "has_preferences" in c and bool(c.get("has_preferences")) == wants
        ]
        scored = [c for c in in_shape if c.get("scores")]
        # Evaluated-but-unscored cases (e.g. the judge omitted a demanded
        # dimension and scoring failed) must stay VISIBLE in the cell — a
        # shrinking n with no marker would be a silent case-count
        # inconsistency replacing the silent dimension-count one (#229 review).
        n_unscored = len(in_shape) - len(scored)
        if not scored:
            shapes[shape_name] = {"n": 0, "n_unscored": n_unscored}
            continue
        averages = [c["scores"]["average"] for c in scored]
        tokens = [
            c["token_usage"]["total_tokens"]
            for c in scored
            if c.get("token_usage") and c["token_usage"].get("total_tokens")
        ]
        shapes[shape_name] = {
            "n": len(scored),
            "n_unscored": n_unscored,
            "mean_average": round(sum(averages) / len(averages), 3),
            "mean_total_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
        }
    return shapes


def select_spot_check_cases(
    dataset_results: List[Dict[str, Any]],
    k: int = 2,
    seed: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Select up to k scored cases across all datasets for human spot-check review.

    Selection is seeded (run-date string by default) so a re-run of the same
    day's evaluation flags the same cases. Only cases that completed with
    judge scores are eligible — a case the judge couldn't score needs a
    failure investigation, not a calibration label.

    Args:
        dataset_results: Per-dataset result dicts (each with case_results)
        k: Maximum number of cases to flag
        seed: Seed for deterministic selection

    Returns:
        List of flagged-case descriptors (dataset, item_id, run_name,
        trace_id, book_title), sorted for stable output
    """
    candidates = []
    for dataset_result in dataset_results:
        for case in dataset_result.get("case_results", []):
            if case.get("status") == "evaluated" and case.get("scores"):
                candidates.append({
                    "dataset": dataset_result.get("dataset_name"),
                    "item_id": case.get("item_id"),
                    "run_name": case.get("run_name"),
                    "trace_id": case.get("trace_id"),
                    "book_title": case.get("book_title"),
                })

    if not candidates:
        return []

    rng = random.Random(seed)
    selected = rng.sample(candidates, min(k, len(candidates)))
    return sorted(selected, key=lambda c: (c["dataset"] or "", c["item_id"] or ""))


def enqueue_for_human_review(
    langfuse: Any,
    queue_id: str,
    selected: List[Dict[str, Any]],
) -> int:
    """
    Add flagged traces to a Langfuse annotation queue for human review.

    Non-fatal by design: an enqueue failure must never fail the eval run —
    the pending_review record in the results JSON is the source of truth
    either way.

    Returns:
        Number of items successfully enqueued
    """
    enqueued = 0
    for case in selected:
        trace_id = case.get("trace_id")
        if not trace_id:
            logger.warning(
                "spot_check_enqueue_skipped",
                item_id=case.get("item_id"),
                reason="no trace_id recorded",
            )
            continue
        try:
            langfuse.api.annotation_queues.create_queue_item(
                queue_id,
                object_id=trace_id,
                object_type="TRACE",
            )
            enqueued += 1
        except Exception as e:
            logger.warning(
                "spot_check_enqueue_failed",
                item_id=case.get("item_id"),
                trace_id=trace_id,
                error=str(e),
            )
    return enqueued


def select_first_region(region_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Automated region selection for evaluation mode - Issue #97.

    Selects the first region from the analysis for deterministic evaluation.
    This replaces the human-in-the-loop region selection needed for automation.

    Args:
        region_analysis: Region analysis dict from discovery workflow

    Returns:
        List containing the first region, or empty list if no regions
    """
    regions = region_analysis.get("regions", [])

    if not regions:
        logger.warning("no_regions_found_for_selection")
        return []

    # Select first region for deterministic evaluation
    selected = [regions[0]]
    region_name = selected[0].get("region_name", "Unknown")

    logger.info(
        "automated_region_selected",
        region_id=selected[0].get("region_id"),
        region_name=region_name,
        total_regions=len(regions),
    )

    return selected


def select_all_regions(region_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Automated region selection for evaluation mode — selects all regions.

    Returns all discovered regions so the composer builds a complete
    multi-region itinerary.

    Args:
        region_analysis: Region analysis dict from discovery workflow

    Returns:
        List of all regions, or empty list if none found
    """
    regions = region_analysis.get("regions", [])

    if not regions:
        logger.warning("no_regions_found_for_selection")
        return []

    logger.info(
        "automated_all_regions_selected",
        total_regions=len(regions),
        region_names=[r.get("region_name", "Unknown") for r in regions],
    )

    return regions


async def run_evaluation_on_dataset(
    dataset_name: str,
    config: Any,
    max_cases: int = 10,
    region_selection: str = "all",
    item_ids: Optional[List[str]] = None,
    start_time: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
    prompt_version: str = CURRENT_PROMPT_VERSION,
) -> Dict[str, Any]:
    """
    Run evaluation on a Langfuse dataset.

    Args:
        dataset_name: Name of the Langfuse dataset
        config: Application configuration
        max_cases: Maximum number of test cases to evaluate
        item_ids: If provided, only evaluate items with these IDs

    Returns:
        Evaluation results summary
    """
    logger.info("starting_evaluation", dataset_name=dataset_name, max_cases=max_cases)

    if not LANGFUSE_AVAILABLE:
        logger.error("langfuse_not_available")
        return {"error": "Langfuse not installed"}

    langfuse = Langfuse(
        secret_key=config.langfuse_secret_key,
        public_key=config.langfuse_public_key,
        host=config.langfuse_host or "https://cloud.langfuse.com",
    )

    try:
        dataset = langfuse.get_dataset(dataset_name)
    except Exception as e:
        logger.error(
            "dataset_fetch_failed",
            dataset_name=dataset_name,
            error=str(e),
        )
        return {
            "dataset_name": dataset_name,
            "error": f"Failed to fetch dataset: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "total_cases": 0,
            "evaluated_cases": 0,
        }

    try:
        items = list(dataset.items)
    except Exception as e:
        logger.error(
            "dataset_items_fetch_failed",
            dataset_name=dataset_name,
            error=str(e),
        )
        return {
            "dataset_name": dataset_name,
            "error": f"Failed to fetch dataset items: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "total_cases": 0,
            "evaluated_cases": 0,
        }

    if item_ids:
        items = [item for item in items if item.id in item_ids]
    items_to_evaluate = items[:max_cases]
    total_cases = len(items_to_evaluate)  # Actual attempted count, not full dataset size
    evaluated_cases = 0  # Real workflow evaluations only
    placeholder_cases = 0  # Placeholder executions (not real evals)
    failed_cases = 0
    skipped_cases = 0  # Cases that couldn't be parsed or processed
    case_results = []

    logger.info(
        "dataset_loaded",
        dataset_name=dataset_name,
        total_cases=total_cases,
        total_items=len(items),
        max_cases=max_cases,
    )

    prompts = load_prompts(prompt_version)

    run_name = f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{prompt_version}"
    run_metadata = {
        "dataset_name": dataset_name,
        "evaluation_type": "scheduled",
        "max_cases": max_cases,
        "prompt_version": prompt_version,
    }

    for item in items_to_evaluate:
        if start_time is not None and timeout_seconds is not None:
            elapsed = time.monotonic() - start_time
            remaining = timeout_seconds - elapsed
            if remaining < 120:  # 2-minute safety margin per case
                logger.warning(
                    "timeout_budget_nearly_exhausted",
                    dataset_name=dataset_name,
                    item_id=item.id,
                    elapsed_seconds=round(elapsed),
                    remaining_seconds=round(remaining),
                    message="Skipping remaining cases to exit gracefully",
                )
                break

        try:
            input_data = item.input
            expected_output = item.expected_output
            item_metadata = item.metadata or {}

            logger.info(
                "evaluating_case",
                dataset_name=dataset_name,
                item_id=item.id,
            )

            # Create a Langfuse run for this dataset item (v4: manual span + dataset run item link)
            # start_as_current_observation sets OTel context so @observe in score_itinerary nests here
            with langfuse.start_as_current_observation(
                as_type="span",
                name=run_name,
                # eval_id is the DATASET-ITEM id, matching the three sibling
                # eval writers. The Experiments API exposes no dataset-item id
                # on an experiment item -- its experiment_item_id is run-scoped
                # -- so this is the only thing that lets a calibration read name
                # the same evalset case across runs (MYS-909).
                metadata={
                    "prompt_version": prompt_version,
                    **run_metadata,
                    "eval_id": item.id,
                },
            ) as root_span:
                try:
                    result = await _run_evaluation_case(
                        input_data=input_data,
                        expected_output=expected_output,
                        item_metadata=item_metadata,
                        config=config,
                        root_span=root_span,
                        region_selection=region_selection,
                        prompts=prompts,
                    )

                    case_status = result.get("status", "unknown")

                    if case_status == "placeholder":
                        logger.info(
                            "placeholder_execution",
                            item_id=item.id,
                            note="Workflow execution not implemented (see issue #95)",
                        )
                finally:
                    # MYS-951: the run is recorded as an EXPERIMENT (OTel
                    # attributes) instead of through the deprecated
                    # POST /dataset-run-items, retired 2026-11-16. Still in
                    # `finally:` for the reason the old call was: a case that
                    # raised is a case that RAN, and dropping it would make a
                    # failing evalset look smaller rather than worse.
                    link_experiment_item(
                        root_span,
                        run_name=run_name,
                        run_metadata=run_metadata,
                        dataset_id=getattr(dataset, "id", None),
                        dataset_name=dataset_name,
                        dataset_item_id=item.id,
                    )

            case_result = {
                "item_id": item.id,
                "run_name": run_name,
                # Langfuse trace link — lets the human spot-check and the
                # calibration tooling find this exact generation later.
                "trace_id": root_span.trace_id,
                **result,
            }
            case_results.append(case_result)

            if case_status == "evaluated":
                evaluated_cases += 1
                logger.info(
                    "case_evaluated",
                    dataset_name=dataset_name,
                    item_id=item.id,
                    run_name=run_name,
                )
            elif case_status == "placeholder":
                placeholder_cases += 1
                logger.info(
                    "placeholder_counted",
                    dataset_name=dataset_name,
                    item_id=item.id,
                )
            elif case_status == "skipped":
                skipped_cases += 1
                logger.warning(
                    "case_skipped",
                    dataset_name=dataset_name,
                    item_id=item.id,
                    reason=result.get("reason", "unknown"),
                )

        except Exception as e:
            logger.error(
                "case_evaluation_failed",
                dataset_name=dataset_name,
                item_id=item.id if hasattr(item, 'id') else 'unknown',
                error=str(e),
            )
            failed_cases += 1
            case_results.append({
                "item_id": item.id if hasattr(item, 'id') else 'unknown',
                "status": "failed",
                "error": str(e),
            })

    langfuse.flush()

    results = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "total_cases": total_cases,
        "evaluated_cases": evaluated_cases,
        "placeholder_cases": placeholder_cases,
        "failed_cases": failed_cases,
        "skipped_cases": skipped_cases,
        "case_results": case_results,
        # Per-shape aggregation (PR-4 step zero): preference-carrying cases
        # exercise the API-contract path; preference-free cases are the shape
        # 100% of prod traffic takes (MYS-392). Gates read the shapes
        # separately — never a silent blend.
        "by_shape": _summarize_by_shape(case_results),
        # Deterministic grounding roll-up (MYS-817). Kept OUT of the judge
        # scores on purpose: it is a measured fact about whether the
        # researchers searched, not an opinion about the output. None when no
        # case reached discovery, so an all-zero block never reads as "nothing
        # was grounded". No pass rule yet — see summarize_search_grounding.
        "search_grounding": aggregate_search_grounding(case_results),
    }

    logger.info(
        "evaluation_complete",
        dataset=dataset_name,
        evaluated=evaluated_cases,
        placeholders=placeholder_cases,
        failed=failed_cases,
        skipped=skipped_cases,
        total=total_cases,
        message=f"View results at {config.langfuse_host}",
    )

    return results


async def _run_evaluation_case(
    input_data: Dict[str, Any],
    expected_output: Any,
    item_metadata: Dict[str, Any],
    config: Any,
    root_span: Any,
    region_selection: str = "all",
    prompts: AgentPrompts | None = None,
) -> Dict[str, Any]:
    """
    Run evaluation for a single test case.

    Executes the full StoryLand workflow:
    1. Metadata extraction - identify book and author
    2. Discovery - find locations and group into travel regions
    3. Automated region selection - Issue #97 (select first/all regions)
    4. Composition - create itinerary for selected region(s)

    Args:
        input_data: Input from dataset item
        expected_output: Expected output (if available)
        item_metadata: Metadata from dataset item (contains session_input with preferences)
        config: Application configuration
        root_span: Langfuse root span (from item.run() context manager)

    Returns:
        Evaluation result with status "evaluated" on success
    """
    try:
        book_title = input_data.get('book_title', '').strip()
        author = input_data.get('author', '').strip()

        if not book_title:
            logger.warning("missing_book_title", input_data=input_data)
            return {"status": "skipped", "reason": "No book_title in input data"}

        logger.info(
            "starting_workflow_evaluation",
            book_title=book_title,
            author=author or "unknown",
        )

        # Same retry config as main.py
        retry_config = build_retry_options(
            attempts=config.retry_attempts,
            exp_base=config.retry_exp_base,
            initial_delay=1,
            max_delay=config.retry_max_delay,
        )
        # ADK 2.x: the api_key goes through client_kwargs (a bare api_key=
        # kwarg is silently dropped and auth falls back to env).
        model = Gemini(
            model=config.model_name,
            client_kwargs={"api_key": config.google_api_key},
            retry_options=retry_config
        )

        session_service = create_session_service(
            connection_string=None,
            use_database=False  # Use in-memory for eval
        )

        langfuse_plugin = LangfusePlugin(
            secret_key=config.langfuse_secret_key,
            public_key=config.langfuse_public_key,
            host=config.langfuse_host,
        )

        session_input = item_metadata.get("session_input", {})
        quality_criteria = item_metadata.get("quality_criteria")
        user_id = session_input.get("user_id", "eval_user")

        session_id = str(uuid.uuid4())
        initial_state = {
            "book_title": book_title,
            "author": author or "",
        }

        # Preferences arrive either under state["user:preferences"] (proper
        # structure) or a top-level "preferences" key (legacy format).
        session_state = session_input.get("state", {})
        if "user:preferences" in session_state:
            initial_state["user:preferences"] = session_state["user:preferences"]
        elif "preferences" in session_input:
            initial_state["user:preferences"] = session_input["preferences"]

        await session_service.create_session(
            app_name="storyland",
            user_id=user_id,
            session_id=session_id,
            state=initial_state,
        )

        logger.info("eval_session_created", session_id=session_id[:8])

        logger.info("eval_phase_1_start", phase="metadata_confirmation")
        root_span.update(metadata={"current_phase": "metadata_confirmation"})

        exact_title = book_title
        exact_author = author or ""

        session = await session_service.get_session(
            app_name="storyland", user_id=user_id, session_id=session_id
        )
        session.state["book_metadata"] = {
            "book_title": exact_title,
            "author": exact_author,
        }

        logger.info(
            "eval_metadata_confirmed",
            exact_title=exact_title,
            exact_author=exact_author,
        )
        root_span.update(
            input={
                "book_title": exact_title,
                "author": exact_author,
                "preferences": item_metadata.get("session_input", {}).get("state", {}).get("user:preferences", {}),
            },
            metadata={
                "phase_1_complete": True,
                "book_title": exact_title,
                "author": exact_author,
            }
        )

        logger.info("eval_phase_2_start", phase="location_discovery")
        root_span.update(metadata={"current_phase": "discovery"})

        try:
            discovery_workflow = create_book_to_place_discovery_workflow(
                model, book_title=exact_title, author=exact_author, prompts=prompts
            )
            langfuse_plugin.root_name = discovery_workflow.name
            discovery_runner = Runner(
                node=discovery_workflow,
                app_name="storyland",
                session_service=session_service,
                plugins=[LoggingPlugin(), langfuse_plugin],
            )

            discovery_prompt = f"""Discover travel locations for "{exact_title}" by {exact_author}.

Find cities, landmarks, and author-related sites, then group them into practical travel regions."""
            discovery_message = types.Content(
                role="user", parts=[types.Part(text=discovery_prompt)]
            )

            async with discovery_runner:
                async for event in discovery_runner.run_async(
                    user_id=user_id, session_id=session_id, new_message=discovery_message
                ):
                    pass

            session = await session_service.get_session(
                app_name="storyland", user_id=user_id, session_id=session_id
            )
            region_analysis = session.state.get("region_analysis", {})

            # Deterministic grounding check (MYS-817). The judge scores output
            # QUALITY and cannot tell a searched answer from a remembered one,
            # which is how MYS-816 stayed invisible: a researcher that skips
            # google_search still produces a plausible, well-scoring itinerary.
            # Read off the plugin's ledger, which is populated regardless of
            # whether Langfuse itself is enabled.
            search_grounding = summarize_search_grounding(langfuse_plugin)

            logger.info(
                "eval_regions_discovered",
                num_regions=len(region_analysis.get("regions", []))
            )

            root_span.update(
                metadata={
                    "phase_2_complete": True,
                    "num_regions": len(region_analysis.get("regions", [])),
                }
            )
        except Exception as e:
            logger.error(
                "eval_discovery_failed",
                error=str(e),
                book_title=exact_title,
            )
            root_span.update(
                metadata={
                    "phase_2_error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            raise

        # Automated region selection (Issue #97)
        logger.info("eval_region_selection", mode="automated", strategy=region_selection)
        if region_selection == "all":
            selected_regions = select_all_regions(region_analysis)
        else:
            selected_regions = select_first_region(region_analysis)

        if not selected_regions:
            error_msg = "No regions available to create an itinerary"
            logger.error("eval_no_regions_available", book_title=exact_title)
            return {
                "status": "failed",
                "reason": error_msg,
                "book_title": exact_title,
                "author": exact_author,
            }

        session = await session_service.get_session(
            app_name="storyland", user_id=user_id, session_id=session_id
        )
        session.state["selected_regions"] = selected_regions

        logger.info("eval_selected_regions_stored", region_count=len(selected_regions))

        # Phase 3 (Composition) - numbering kept in spans for Langfuse compat
        logger.info("eval_phase_3_start", phase="itinerary_composition")

        root_span.update(
            metadata={
                "current_phase": "composition",
                "region_count": len(selected_regions),
                "selected_regions": [r.get("region_name") for r in selected_regions],
            }
        )

        try:
            composition_workflow = create_book_to_place_composition_workflow(model, prompts=prompts)
            langfuse_plugin.root_name = composition_workflow.name
            composition_runner = Runner(
                node=composition_workflow,
                app_name="storyland",
                session_service=session_service,
                plugins=[LoggingPlugin(), langfuse_plugin],
            )

            # Same prompt path as production (core.prompts.build_composition_prompt)
            # INCLUDING the explicit grounding from session state — under the
            # graph runtime (ADR #24) the composer sees none of the discovery
            # conversation, so an eval that composed from region names alone
            # would score a different, under-grounded workflow than prod.
            grounding_session = await session_service.get_session(
                app_name="storyland", user_id=user_id, session_id=session_id
            )
            # Carry the unverified-researcher fact into the eval's state the
            # way discover() carries it in production, so grounding_research_text
            # and all_discovery_unverified read the same here as they do there.
            grounding_state_dict = dict(grounding_session.state if grounding_session else {})
            grounding_state_dict[SessionStateKeys.UNVERIFIED_DISCOVERY] = (
                unverified_payload_keys(
                    langfuse_plugin,
                    SessionStateAccessor(grounding_state_dict).present_discovery_keys,
                )
            )
            grounding_state = SessionStateAccessor(grounding_state_dict)
            composition_prompt = build_composition_prompt(
                exact_title,
                exact_author,
                selected_regions,
                book_context=grounding_state.book_context,
                city_discovery=grounding_state.city_discovery,
                landmark_discovery=grounding_state.landmark_discovery,
                author_sites=grounding_state.author_sites,
                preferences=grounding_state.user_preferences,
            )
            composition_message = types.Content(
                role="user", parts=[types.Part(text=composition_prompt)]
            )

            final_response = None
            async with composition_runner:
                async for event in composition_runner.run_async(
                    user_id=user_id, session_id=session_id, new_message=composition_message
                ):
                    if event.is_final_response():
                        final_response = event

            itinerary_data = None
            if final_response and final_response.content and final_response.content.parts:
                for part in final_response.content.parts:
                    if hasattr(part, "text") and part.text:
                        json_start = part.text.find("{")
                        json_end = part.text.rfind("}") + 1

                        if json_start >= 0 and json_end > json_start:
                            try:
                                itinerary_data = json.loads(part.text[json_start:json_end])
                                break
                            except json.JSONDecodeError as e:
                                logger.warning("eval_json_parse_error", error=str(e))

            # Score what production would have SHIPPED, not what the composer
            # said (MYS-817). This is the one place the eval can diverge from
            # the deployed pipeline without anything going red.
            itinerary_data = apply_production_grounding_downgrade(
                itinerary_data, grounding_state
            )

            logger.info(
                "eval_composition_complete",
                itinerary_created=itinerary_data is not None,
                num_cities=count_itinerary_cities(itinerary_data),
            )

            root_span.update(
                output=itinerary_data,
                metadata={
                    "phase_3_complete": True,
                    "itinerary_created": itinerary_data is not None,
                    "num_cities": count_itinerary_cities(itinerary_data),
                }
            )
        except Exception as e:
            logger.error(
                "eval_composition_failed",
                error=str(e),
                book_title=exact_title,
            )
            root_span.update(
                metadata={
                    "phase_3_error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            raise

        token_stats = langfuse_plugin.get_session_stats()
        logger.info(
            "eval_workflow_complete",
            book_title=exact_title,
            author=exact_author,
            itinerary_created=itinerary_data is not None,
            total_tokens=token_stats.get('total_tokens', 0),
            cost_usd=token_stats.get('cost_usd', 0),
        )

        await langfuse_plugin.flush()

        # Phase 4: LLM-as-judge scoring (Issue #96)
        scores_data = None
        if itinerary_data:
            logger.info("eval_phase_4_start", phase="llm_scoring")

            try:
                preferences = session.state.get("user:preferences", {})

                # quality_criteria deliberately NOT passed to the quality
                # judge (MYS-586 ablation: the injection made books_v1 grade
                # compliance-with-specifics, not quality — divergence −0.95
                # with it, +0.07 without). Compliance is its own score below.
                scores = await score_itinerary(
                    api_key=config.google_api_key,
                    book_title=exact_title,
                    author=exact_author,
                    input_text=book_title,
                    itinerary=itinerary_data,
                    preferences=preferences,
                    expected_output=expected_output,
                )

                root_span.score_trace(
                    name="book_relevance",
                    value=scores.book_relevance,
                    comment="LLM-as-judge: Connection to book's settings, themes, or author (1-5)",
                )
                if scores.preference_adherence is not None:
                    root_span.score_trace(
                        name="preference_adherence",
                        value=scores.preference_adherence,
                        comment="LLM-as-judge: Respect for user preferences (1-5)",
                    )
                root_span.score_trace(
                    name="completeness",
                    value=scores.completeness,
                    comment="LLM-as-judge: Comprehensive details included (1-5)",
                )
                root_span.score_trace(
                    name="actionability",
                    value=scores.actionability,
                    comment="LLM-as-judge: Practical and actionable information (1-5)",
                )
                root_span.score_trace(
                    name="geographical_accuracy",
                    value=scores.geographical_accuracy,
                    comment="LLM-as-judge: Accuracy of locations (1-5)",
                )
                root_span.score_trace(
                    name="engagement",
                    value=scores.engagement,
                    comment="LLM-as-judge: Engaging descriptions that evoke book's spirit (1-5)",
                )

                scores_data = {
                    "book_relevance": scores.book_relevance,
                    "completeness": scores.completeness,
                    "actionability": scores.actionability,
                    "geographical_accuracy": scores.geographical_accuracy,
                    "engagement": scores.engagement,
                    "average": round(scores.average_score(), 2),
                    "scoring_method": "llm_judge",
                    "scored_at": datetime.now().isoformat(),
                }
                # Present only when scored (no-preference cases average 5 dims).
                if scores.preference_adherence is not None:
                    scores_data["preference_adherence"] = scores.preference_adherence

                # Compliance with book-specific criteria: a SEPARATE score,
                # never blended into the quality average — one number carrying
                # quality and compliance meant a reader couldn't tell which
                # half moved. Failure records a visible null, not an absence.
                if quality_criteria:
                    try:
                        coverage = await score_criteria_coverage(
                            api_key=config.google_api_key,
                            book_title=exact_title,
                            author=exact_author,
                            quality_criteria=quality_criteria,
                            itinerary=itinerary_data,
                                )
                        scores_data["criteria_coverage"] = coverage
                        root_span.score_trace(
                            name="criteria_coverage",
                            value=coverage,
                            comment="Compliance with book-specific criteria (1-5) — separate from quality",
                        )
                    except Exception as coverage_error:
                        scores_data["criteria_coverage"] = None
                        logger.warning(
                            "criteria_coverage_failed",
                            error=str(coverage_error),
                            book_title=exact_title,
                        )

                logger.info(
                    "eval_scoring_complete",
                    book_relevance=scores.book_relevance,
                    preference_adherence=scores.preference_adherence,
                    completeness=scores.completeness,
                    actionability=scores.actionability,
                    geographical_accuracy=scores.geographical_accuracy,
                    engagement=scores.engagement,
                    average_score=scores_data["average"],
                )

            except Exception as e:
                logger.warning(
                    "eval_scoring_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    book_title=exact_title,
                )
                # Don't fail evaluation if scoring fails - scoring is optional

        result = {
            "status": "evaluated",
            "book_title": exact_title,
            "author": exact_author,
            "input": book_title,
            # Shape marker for per-shape gating: preference-carrying cases
            # exercise the API-contract path; preference-free cases are the
            # shape 100% of prod traffic takes (MYS-392). Derived from
            # initial_state (always bound), not the scoring-block local.
            "has_preferences": bool(initial_state.get("user:preferences")),
            "itinerary_created": itinerary_data is not None,
            "num_cities": count_itinerary_cities(itinerary_data),
            "num_regions": len(selected_regions),
            "token_usage": token_stats,
            # Deterministic, not judged — see summarize_search_grounding.
            "search_grounding": search_grounding,
            # Full payload + the preferences the judge saw, so results JSONs
            # are self-contained for human review and judge calibration
            # (previously only Langfuse traces carried the itinerary).
            "preferences": initial_state.get("user:preferences"),
            "itinerary": itinerary_data,
        }

        if scores_data:
            result["scores"] = scores_data

        return result

    except Exception as e:
        logger.error("evaluation_case_error", error=str(e), error_type=type(e).__name__)
        raise


async def run_all_evaluations(
    output_dir: str = "evaluation/results",
    max_cases_per_dataset: int = 10,
    dataset_names: Optional[List[str]] = None,
    region_selection: str = "all",
    item_ids: Optional[List[str]] = None,
    timeout_minutes: Optional[float] = None,
    prompt_version: str = CURRENT_PROMPT_VERSION,
) -> List[Dict[str, Any]]:
    """
    Run evaluations on specified or all available datasets.

    Args:
        output_dir: Directory to save evaluation results
        max_cases_per_dataset: Maximum cases to evaluate per dataset
        dataset_names: List of dataset names to evaluate. If None, evaluates
                      datasets found in evaluation/langfuse_datasets.json or
                      defaults to ["storyland_eval", "books_v1"]

    Returns:
        List of evaluation result summaries
    """
    config = load_config()

    if not all([config.langfuse_secret_key, config.langfuse_public_key]):
        logger.error(
            "langfuse_not_configured",
            message="Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to run evaluations",
        )
        # Return error result instead of empty list to distinguish from legitimate empty results
        return [{
            "dataset_name": "config_error",
            "error": "Langfuse credentials not configured. Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY.",
            "timestamp": datetime.now().isoformat(),
            "total_cases": 0,
            "evaluated_cases": 0,
        }]

    if dataset_names:
        datasets = dataset_names
        logger.info("using_provided_datasets", datasets=datasets)
    else:
        # Try to load from langfuse_datasets.json (created by langfuse_eval.py)
        datasets_file = Path("evaluation/langfuse_datasets.json")
        if datasets_file.exists():
            try:
                with open(datasets_file, 'r') as f:
                    datasets_info = json.load(f)
                    datasets, routed = select_itinerary_datasets(datasets_info)
                    for entry in routed:
                        # Routed, not failed, not silent: these datasets are
                        # healthy — they just belong to a dedicated runner.
                        logger.info(
                            "dataset_routed_to_dedicated_runner",
                            dataset_name=entry["dataset_name"],
                            flow=entry["flow"],
                            runner=DEDICATED_RUNNERS.get(entry["flow"], "unknown"),
                        )
                    if not datasets:
                        logger.warning(
                            "no_itinerary_datasets_discovered",
                            file=str(datasets_file),
                            message="Registry has no itinerary-flow datasets; "
                                    "falling back to defaults",
                        )
                        datasets = ["storyland_eval", "books_v1"]
                    logger.info(
                        "loaded_datasets_from_file",
                        file=str(datasets_file),
                        datasets=datasets,
                    )
            except Exception as e:
                logger.warning(
                    "failed_to_load_datasets_file",
                    file=str(datasets_file),
                    error=str(e),
                )
                datasets = ["storyland_eval", "books_v1"]
        else:
            datasets = ["storyland_eval", "books_v1"]
            logger.info("using_default_datasets", datasets=datasets)

    start_time = time.monotonic()
    timeout_seconds = timeout_minutes * 60 if timeout_minutes is not None else None

    if timeout_seconds is not None:
        logger.info(
            "timeout_budget_set",
            timeout_minutes=timeout_minutes,
        )

    results = []
    for dataset_name in datasets:
        if timeout_seconds is not None:
            elapsed = time.monotonic() - start_time
            remaining = timeout_seconds - elapsed
            if remaining < 120:  # 2-minute safety margin
                logger.warning(
                    "timeout_budget_exhausted",
                    elapsed_seconds=round(elapsed),
                    remaining_seconds=round(remaining),
                    skipping_datasets=datasets[datasets.index(dataset_name):],
                    message="Skipping remaining datasets to exit gracefully",
                )
                break

        try:
            result = await run_evaluation_on_dataset(
                dataset_name=dataset_name,
                config=config,
                max_cases=max_cases_per_dataset,
                region_selection=region_selection,
                item_ids=item_ids,
                start_time=start_time,
                timeout_seconds=timeout_seconds,
                prompt_version=prompt_version,
            )
            results.append(result)
        except Exception as e:
            logger.error(
                "dataset_evaluation_failed",
                dataset_name=dataset_name,
                error=str(e),
            )
            # Continue with other datasets instead of aborting
            results.append({
                "dataset_name": dataset_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "total_cases": 0,
                "evaluated_cases": 0,
            })

    # Human spot-check: flag 1-2 scored cases per run for hand review.
    # The record lives in the results JSON regardless of what happens in
    # Langfuse, so a review that never happens stays visible as
    # pending_review in the trend report — skipped, not silent.
    selected = select_spot_check_cases(
        results, k=2, seed=datetime.now().strftime("%Y%m%d")
    )
    human_spot_check: Dict[str, Any] = {
        "selected": selected,
        "status": "pending_review" if selected else "no_scored_cases",
        "selected_at": datetime.now().isoformat(),
        "review_queue_id": None,
        "enqueued": 0,
    }
    review_queue_id = os.getenv("LANGFUSE_REVIEW_QUEUE_ID")
    if selected and review_queue_id and LANGFUSE_AVAILABLE:
        try:
            langfuse = Langfuse(
                secret_key=config.langfuse_secret_key,
                public_key=config.langfuse_public_key,
                host=config.langfuse_host or "https://cloud.langfuse.com",
            )
            human_spot_check["enqueued"] = enqueue_for_human_review(
                langfuse, review_queue_id, selected
            )
            human_spot_check["review_queue_id"] = review_queue_id
        except Exception as e:
            logger.warning(
                "spot_check_queue_unavailable",
                queue_id=review_queue_id,
                error=str(e),
            )
    logger.info(
        "spot_check_selected",
        num_selected=len(selected),
        items=[c["item_id"] for c in selected],
        enqueued=human_spot_check["enqueued"],
    )

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'human_spot_check': human_spot_check,
        }, f, indent=2)

    logger.info("all_evaluations_complete", output_file=output_file)
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run scheduled evaluations')
    parser.add_argument(
        '--output-dir',
        default='evaluation/results',
        help='Directory to save evaluation results',
    )
    parser.add_argument(
        '--max-cases',
        type=int,
        default=10,
        help='Maximum cases to evaluate per dataset',
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets to evaluate (e.g., storyland_eval books_v1). '
             'If not specified, evaluates all configured datasets.',
    )
    parser.add_argument(
        '--region-selection',
        choices=['first', 'all'],
        default='all',
        help="Region selection strategy: 'first' (single region) or 'all' (all regions). Default: all",
    )
    parser.add_argument(
        '--item-ids',
        nargs='+',
        help='Specific item IDs to evaluate (e.g., query_013 query_014). Evaluates all items if not specified.',
    )
    parser.add_argument(
        '--timeout-minutes',
        type=float,
        default=None,
        help='Maximum total runtime in minutes. Script exits gracefully before this limit, '
             'saving results for all completed cases.',
    )
    parser.add_argument(
        '--prompt-version',
        default=CURRENT_PROMPT_VERSION,
        help='Prompt version label for A/B comparison in Langfuse (e.g. v2, v3). '
             'Included in the run name and metadata so runs can be filtered by version.',
    )

    args = parser.parse_args()

    results = asyncio.run(
        run_all_evaluations(
            output_dir=args.output_dir,
            max_cases_per_dataset=args.max_cases,
            dataset_names=args.datasets,
            region_selection=args.region_selection,
            item_ids=args.item_ids,
            timeout_minutes=args.timeout_minutes,
            prompt_version=args.prompt_version,
        )
    )

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    total_evaluated = 0
    total_placeholders = 0
    total_failed_datasets = 0
    total_skipped = 0
    total_datasets = len(results)

    for result in results:
        print(f"\nDataset: {result.get('dataset_name', 'unknown')}")
        print(f"Timestamp: {result.get('timestamp', 'N/A')}")

        evaluated = result.get('evaluated_cases', 0)
        placeholders = result.get('placeholder_cases', 0)
        total_cases = result.get('total_cases', 0)
        failed_cases = result.get('failed_cases', 0)
        skipped_cases = result.get('skipped_cases', 0)

        if evaluated > 0:
            print(f"Real evaluations: {evaluated} cases")
        if placeholders > 0:
            print(f"Placeholder runs: {placeholders} cases (workflow not implemented)")
        if skipped_cases > 0:
            print(f"SKIPPED: {skipped_cases} case(s) could not be parsed")
        if evaluated == 0 and placeholders == 0 and total_cases > 0:
            print(f"Evaluated: 0 cases")

        total_evaluated += evaluated
        total_placeholders += placeholders

        # Failure = explicit error, failed cases, skipped cases, OR a judge that
        # scored nothing (MYS-825). Empty dataset (total_cases == 0) is NOT a
        # failure. See dataset_failure_reason for why "measured nothing" counts.
        failure = dataset_failure_reason(result)

        if failure:
            total_failed_datasets += 1
            print(failure)
        elif total_cases == 0:
            total_skipped += 1
            print("INFO: Dataset is empty (no test cases)")
        else:
            unscored = evaluated - count_scored_cases(result)
            if unscored > 0:
                print(f"WARNING: {unscored} of {evaluated} case(s) evaluated but unscored")

        # Deterministic (not judged): did the researchers actually search?
        grounding = result.get("search_grounding")
        if grounding:
            print(
                f"Search grounding [deterministic]: "
                f"{grounding['researchers_grounded']}/{grounding['researchers_total']} "
                f"researcher runs grounded; "
                f"{grounding['cases_fully_grounded']}/{grounding['cases']} "
                f"case(s) fully grounded"
            )
            if grounding["unsearched_by_agent"]:
                offenders = ", ".join(
                    f"{agent}×{n}" for agent, n in grounding["unsearched_by_agent"].items()
                )
                print(f"  skipped google_search: {offenders}")
            if grounding.get("unobserved_by_agent"):
                missing = ", ".join(
                    f"{agent}×{n}"
                    for agent, n in grounding["unobserved_by_agent"].items()
                )
                print(
                    "  NEVER OBSERVED (instrumentation, not the model): "
                    f"{missing}"
                )

    print("\n" + "=" * 60)
    print(f"Total: {total_evaluated} real evaluation(s), {total_placeholders} placeholder(s)")
    if total_skipped > 0:
        print(f"Empty datasets: {total_skipped}")
    print(f"Failed datasets: {total_failed_datasets}")
    print("=" * 60)

    # Nonzero exit on any dataset failure so GitHub Actions reports it;
    # empty datasets don't count as failures.
    actual_datasets = total_datasets - total_skipped

    if actual_datasets == 0 and total_failed_datasets > 0:
        print("\n❌ ERROR: Configuration or setup error - cannot run evaluations")
        sys.exit(1)
    elif actual_datasets > 0 and total_failed_datasets == actual_datasets:
        print("\n❌ ERROR: All dataset evaluations failed")
        sys.exit(1)
    elif total_failed_datasets > 0:
        print(f"\n❌ ERROR: {total_failed_datasets}/{actual_datasets} dataset(s) failed")
        sys.exit(1)
    elif actual_datasets == 0:
        print("\n⚠️  WARNING: No datasets to evaluate (all empty)")
    else:
        if total_evaluated > 0:
            print("\n✅ All evaluations completed successfully")
        elif total_placeholders > 0:
            print("\n⚠️  Placeholder runs completed (no real workflow evaluation)")
            print("   Implement issues #95, #96, #97 for actual quality evaluation")
        else:
            print("\n✅ Execution completed (no cases to evaluate)")


if __name__ == '__main__':
    main()
