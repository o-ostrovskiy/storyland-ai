"""Run the expansion (suggestion-chip) deterministic eval against Langfuse.

The expansion flow adds new places to an existing itinerary when the user
clicks a suggestion chip. Its contract is deterministic, so — like the
place→book grounding gate (``run_place_to_book_eval.py``) — it is scored with
hard checks, not an LLM judge.

Each ``expansion_v1`` case is a TWO-STEP harness driven through the live
``WorkflowExecutor`` (the production code path — never the raw expansion
workflow, because the invariants under test are executor logic:
``source="expansion"`` stamping, the dedupe post-filter, chip-id stamping,
and trusted-chip resolution per MYS-167):

  1. ``executor.discover`` → ``RegionsReady`` (job_id + regions);
  2. ``executor.compose`` on the first region → ``ItineraryReady`` (base
     itinerary + server-stamped suggestion chips);
  3. pick a chip (case's ``chip_keyword`` substring match on label /
     action_prompt, falling back to the first chip) and ``executor.expand``
     → ``ExpansionReady``;
  4. deterministic gate: at least ``min_new_places`` new stops, every new stop
     stamped ``source="expansion"``, no duplicates against the base itinerary
     (case-insensitive name match — the executor's own dedupe key), every new
     stop validates as a ``CityStop``, and follow-up chips (≤4 by schema)
     carry non-empty unique server-stamped ids.

A compose that legally returns zero chips is recorded as ``no_chips`` and
counted skipped (scored ``chips_available=0``), never failed — LLM chip-yield
flakiness must not read as a deterministic-gate failure. The SOFT_CHIP_CAP=6
empty-chips branch is not exercised live (it would cost 6 paid expands per
case); the observable one-expansion contract is asserted instead and the cap
branch stays pinned by unit tests.

Exit code reflects infrastructure health, not case verdicts — consume
``passed``/``failed`` from the results JSON, mirroring the place→book runner.

Usage:
    python evaluation/tools/run_expansion_eval.py [--max-cases N] [--no-register]
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.tools.experiment_run import link_experiment_item  # noqa: E402

from common.logging import get_logger
from models.itinerary import CityStop

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    LANGFUSE_AVAILABLE = False

logger = get_logger("storyland.expansion_eval")

DATASET_NAME = "expansion_v1"
EVALSET_FILE = "evaluation/expansion_v1.evalset.json"

# ExpansionResult schema caps follow-up chips at 2-4; the executor may also
# legally return zero (soft cap). Anything above 4 is a contract break.
MAX_FOLLOWUP_CHIPS = 4


def pick_chip(suggestions: List[dict], keyword: Optional[str]) -> Optional[dict]:
    """Pick the chip to click: keyword substring match, else the first chip.

    The keyword is matched case-insensitively against both ``label`` and
    ``action_prompt``. Returns None when there are no chips at all.
    """
    if not suggestions:
        return None
    if keyword:
        kw = keyword.lower()
        for chip in suggestions:
            haystack = f"{chip.get('label', '')} {chip.get('action_prompt', '')}".lower()
            if kw in haystack:
                return chip
    return suggestions[0]


def collect_stop_names(itinerary: dict) -> set:
    """Lowercased bare stop names across all cities — the executor's dedupe key."""
    names = set()
    for city in (itinerary or {}).get("cities", []):
        for stop in city.get("stops", []):
            name = (stop.get("name") or "").strip().lower()
            if name:
                names.add(name)
    return names


def check_expansion_case(
    base_itinerary: dict,
    expansion: Dict[str, Any],
    min_new_places: int = 1,
) -> Dict[str, Any]:
    """Deterministic gate for one expansion result.

    ``expansion`` carries the ``ExpansionReady`` payload:
    ``{"parent_city": str, "places": [...], "suggestions": [...]}``.
    """
    places = expansion.get("places", [])
    suggestions = expansion.get("suggestions", [])
    parent_city = expansion.get("parent_city", "")

    base_names = collect_stop_names(base_itinerary)
    new_names = [(p.get("name") or "").strip().lower() for p in places]
    duplicates = sorted(base_names.intersection(n for n in new_names if n))

    source_stamped = all(p.get("source") == "expansion" for p in places)

    schema_errors: List[str] = []
    for p in places:
        try:
            CityStop.model_validate(p)
        except Exception as e:
            schema_errors.append(f"{p.get('name', '<unnamed>')}: {e}")

    chip_ids = [c.get("id") for c in suggestions]
    # Vacuously true for an empty chip list — the soft cap legally empties it.
    chip_ids_stamped = all(bool(cid) for cid in chip_ids) and (
        len(set(chip_ids)) == len(chip_ids)
    )
    chips_within_cap = len(suggestions) <= MAX_FOLLOWUP_CHIPS

    base_cities = {
        (c.get("name") or "").strip().lower()
        for c in (base_itinerary or {}).get("cities", [])
    }
    parent_city_known = parent_city.strip().lower() in base_cities

    passed = (
        len(places) >= min_new_places
        and source_stamped
        and not duplicates
        and not schema_errors
        and chip_ids_stamped
        and chips_within_cap
    )

    return {
        "n_new_places": len(places),
        "min_new_places": min_new_places,
        "source_stamped": bool(source_stamped),
        "duplicates": duplicates,
        "no_duplicates": not duplicates,
        "places_schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "n_suggestions": len(suggestions),
        "chips_within_cap": bool(chips_within_cap),
        "chip_ids_stamped": bool(chip_ids_stamped),
        # Informational only: executor.expand's merge fallback can legitimately
        # rewrite parent_city onto the first city, so this never gates.
        "parent_city_known": bool(parent_city_known),
        "parent_city": parent_city,
        "pass": bool(passed),
    }


def build_mechanism_section() -> Dict[str, Any]:
    """The ``mechanism`` block every eval report must carry (eval protocol)."""
    return {
        "deterministic": [
            "n_new_places >= min_new_places",
            "every new stop stamped source='expansion' (executor contract)",
            "no duplicate stop names vs the base itinerary (case-insensitive — "
            "the executor's own dedupe key)",
            "every new stop validates as CityStop (models/itinerary.py)",
            f"follow-up chips <= {MAX_FOLLOWUP_CHIPS} with non-empty unique "
            "server-stamped ids",
        ],
        "llm_judge": None,
        "preference_shapes": "not applicable — all cases are preference-free; "
                             "per-shape aggregation therefore does not apply "
                             "to this dataset",
        "pass_rule": "all deterministic checks pass; a zero-chip compose is "
                     "skipped (no_chips), not failed",
        "harness": "WorkflowExecutor discover → compose(first region) → "
                   "expand(chip) — production code path, cache disabled, "
                   "in-memory sessions",
    }


async def run(max_cases: Optional[int], register: bool, output_dir: str) -> Dict[str, Any]:
    from common.config import load_config
    from common.logging import configure_logging

    config = load_config()
    configure_logging(level=config.log_level, enable_adk_debug=config.enable_adk_debug)

    if not LANGFUSE_AVAILABLE:
        raise RuntimeError("langfuse package not available")
    if not (config.langfuse_public_key and config.langfuse_secret_key):
        raise RuntimeError(
            "Langfuse credentials missing — set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"
        )

    if register:
        from evaluation.tools.langfuse_eval import LangfuseEvalPipeline
        LangfuseEvalPipeline().create_dataset_from_evalset(
            EVALSET_FILE, dataset_name=DATASET_NAME
        )
        logger.info("expansion_dataset_refreshed", dataset=DATASET_NAME)

    from core.executor import WorkflowExecutor
    from core.types import ExecutorConfig
    from core.events import (
        RegionsReady,
        ItineraryReady,
        ExpansionReady,
        WorkflowError,
    )

    exec_config = ExecutorConfig.from_config(config)
    exec_config.cache_enabled = False  # fresh discovery per case, no replay
    exec_config.use_database = False   # in-memory sessions
    executor = WorkflowExecutor(exec_config)

    langfuse = Langfuse(
        secret_key=config.langfuse_secret_key,
        public_key=config.langfuse_public_key,
        host=config.langfuse_host or "https://cloud.langfuse.com",
    )
    dataset = langfuse.get_dataset(DATASET_NAME)
    items = list(dataset.items)
    if max_cases is not None:
        items = items[:max_cases]

    run_name = f"exp_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # model_under_test labels every record: baselines are model-bound, and a
    # model lift must read as a re-baseline, not a quality regression.
    run_metadata = {
        "dataset_name": DATASET_NAME,
        "evaluation_type": "expansion",
        "model_under_test": config.model_name,
    }
    case_results: List[Dict[str, Any]] = []

    logger.info("starting_expansion_eval", dataset=DATASET_NAME,
                cases=len(items), run_name=run_name)

    for item in items:
        inp = item.input or {}
        meta = item.metadata or {}
        book_title = (inp.get("book_title") or "").strip()
        author = (inp.get("author") or "").strip()
        chip_keyword = meta.get("chip_keyword")
        min_new_places = meta.get("min_new_places", 1)

        if not book_title:
            logger.warning("expansion_item_missing_input", item_id=item.id)
            case_results.append({
                "item_id": item.id, "status": "skipped",
                "reason": "no book_title in input",
            })
            continue

        with langfuse.start_as_current_observation(
            as_type="span", name=run_name, metadata={**run_metadata, "eval_id": item.id},
        ) as root_span:
            try:
                result = await _run_case(
                    executor=executor,
                    book_title=book_title,
                    author=author,
                    chip_keyword=chip_keyword,
                    min_new_places=min_new_places,
                    region_selection=meta.get("region_selection", "first"),
                    events=(RegionsReady, ItineraryReady, ExpansionReady, WorkflowError),
                )

                root_span.update(
                    input={"book_title": book_title, "author": author,
                           "chip_keyword": chip_keyword},
                    output=result,
                )
                status = result["status"]
                if status == "no_chips":
                    root_span.score_trace(
                        name="chips_available", value=0.0,
                        comment=f"{book_title}: compose returned zero suggestion "
                                "chips — case skipped, not failed",
                    )
                elif status == "failed":
                    root_span.score_trace(
                        name="case_pass", value=0.0,
                        comment=f"{book_title}: {result.get('stage')} stage failed — "
                                f"{result.get('reason')}",
                    )
                else:  # evaluated
                    checks = result["checks"]
                    root_span.score_trace(
                        name="chips_available", value=1.0,
                        comment="Compose produced suggestion chips",
                    )
                    root_span.score_trace(
                        name="case_pass", value=float(checks["pass"]),
                        comment=f"{book_title}: {checks['n_new_places']} new place(s) "
                                f"via chip '{result.get('chip_label')}'",
                    )
                    root_span.score_trace(
                        name="no_duplicates", value=float(checks["no_duplicates"]),
                        comment="1.0 = no new stop duplicates a base-itinerary stop",
                    )
                    root_span.score_trace(
                        name="source_stamped", value=float(checks["source_stamped"]),
                        comment="1.0 = every new stop carries source='expansion'",
                    )
                    root_span.score_trace(
                        name="places_schema_valid",
                        value=float(checks["places_schema_valid"]),
                        comment="1.0 = every new stop validates as CityStop",
                    )
                    root_span.score_trace(
                        name="chip_ids_stamped", value=float(checks["chip_ids_stamped"]),
                        comment="1.0 = follow-up chips carry non-empty unique server ids",
                    )
                    root_span.score_trace(
                        name="new_places_count", value=float(checks["n_new_places"]),
                        comment="Count of new stops returned by the expansion",
                    )

                case_results.append({
                    "item_id": item.id, "book_title": book_title,
                    "run_name": run_name, **result,
                })
                logger.info(
                    "expansion_case_done", item_id=item.id, status=status,
                    passed=result.get("checks", {}).get("pass"),
                )
            finally:
                # MYS-951 — experiment attributes, not the retired
                # POST /dataset-run-items. See experiment_run.py.
                link_experiment_item(
                    root_span,
                    run_name=run_name,
                    run_metadata=run_metadata,
                    dataset_id=getattr(dataset, "id", None),
                    dataset_name=DATASET_NAME,
                    dataset_item_id=item.id,
                )

    langfuse.flush()
    await executor.close()

    evaluated = [c for c in case_results if c.get("status") == "evaluated"]
    passed = sum(1 for c in evaluated if c["checks"]["pass"])
    summary = {
        "dataset_name": DATASET_NAME,
        "run_name": run_name,
        "model_under_test": config.model_name,
        "timestamp": datetime.now().isoformat(),
        "total": len(case_results),
        "evaluated": len(evaluated),
        "passed": passed,
        "failed": len(evaluated) - passed
                  + sum(1 for c in case_results if c.get("status") == "failed"),
        "skipped": sum(1 for c in case_results
                       if c.get("status") in ("skipped", "no_chips")),
        "mechanism": build_mechanism_section(),
        "case_results": case_results,
    }

    out_path = Path(output_dir) / f"expansion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print(f"expansion deterministic eval — {DATASET_NAME}")
    print("=" * 72)
    for c in case_results:
        status = c.get("status")
        if status == "evaluated":
            checks = c["checks"]
            print(f"  [{'PASS' if checks['pass'] else 'FAIL'}] {c['item_id']:<22} "
                  f"new={checks['n_new_places']} dup={len(checks['duplicates'])} "
                  f"src={'ok' if checks['source_stamped'] else 'BAD'} "
                  f"schema={'ok' if checks['places_schema_valid'] else 'BAD'} "
                  f"chips={checks['n_suggestions']}"
                  f"{'' if checks['chip_ids_stamped'] else ' ids=BAD'}")
        elif status == "no_chips":
            print(f"  [SKIP] {c['item_id']:<22} compose returned zero chips")
        elif status == "failed":
            print(f"  [FAIL] {c['item_id']:<22} {c.get('stage')}: {c.get('reason')}")
        else:
            print(f"  [SKIP] {c['item_id']} ({c.get('reason')})")
    print("-" * 72)
    print("mechanism:")
    print(json.dumps(summary["mechanism"], indent=2))
    print("=" * 72)
    print(f"Result: {passed}/{len(evaluated)} PASS | Langfuse run: {run_name}")
    print(f"Results saved to {out_path}")
    return summary


async def _run_case(
    executor,
    book_title: str,
    author: str,
    chip_keyword: Optional[str],
    min_new_places: int,
    region_selection: str,
    events,
) -> Dict[str, Any]:
    """Drive discover → compose → expand for one case; return a result dict."""
    RegionsReady, ItineraryReady, ExpansionReady, WorkflowError = events

    regions_ev = error_ev = None
    async for ev in executor.discover(
        book_title=book_title, author=author, user_id="eval_user",
    ):
        if isinstance(ev, RegionsReady):
            regions_ev = ev
        elif isinstance(ev, WorkflowError):
            error_ev = ev
    if regions_ev is None or not regions_ev.regions:
        return {"status": "failed", "stage": "discover",
                "reason": error_ev.message if error_ev else "no regions"}

    job_id = regions_ev.job_id
    if region_selection == "all":
        region_ids = [r.get("region_id") for r in regions_ev.regions]
    else:
        region_ids = [regions_ev.regions[0].get("region_id")]

    itinerary_ev = error_ev = None
    async for ev in executor.compose(
        job_id=job_id, region_ids=region_ids, user_id="eval_user",
    ):
        if isinstance(ev, ItineraryReady):
            itinerary_ev = ev
        elif isinstance(ev, WorkflowError):
            error_ev = ev
    if itinerary_ev is None:
        return {"status": "failed", "stage": "compose",
                "reason": error_ev.message if error_ev else "no ItineraryReady"}

    base_itinerary = itinerary_ev.itinerary
    chips = itinerary_ev.suggestions
    if not chips:
        return {"status": "no_chips", "job_id": job_id}

    chip = pick_chip(chips, chip_keyword)

    expansion_ev = error_ev = None
    async for ev in executor.expand(
        job_id=job_id,
        action_id=chip.get("id", ""),
        action_label=chip.get("label", ""),
        action_prompt=chip.get("action_prompt", ""),
        user_id="eval_user",
    ):
        if isinstance(ev, ExpansionReady):
            expansion_ev = ev
        elif isinstance(ev, WorkflowError):
            error_ev = ev
    if expansion_ev is None:
        return {"status": "failed", "stage": "expand",
                "reason": error_ev.message if error_ev else "no ExpansionReady"}

    expansion = {
        "parent_city": expansion_ev.parent_city,
        "places": expansion_ev.places,
        "suggestions": expansion_ev.suggestions,
    }
    checks = check_expansion_case(base_itinerary, expansion, min_new_places)
    return {
        "status": "evaluated",
        "job_id": job_id,
        "chip_label": chip.get("label"),
        "chip_matched_keyword": bool(
            chip_keyword
            and chip_keyword.lower() in
            f"{chip.get('label', '')} {chip.get('action_prompt', '')}".lower()
        ),
        "checks": checks,
        "expansion": expansion,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the expansion deterministic eval against Langfuse"
    )
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of cases (default: all)")
    parser.add_argument("--output-dir", default="evaluation/results",
                        help="Directory for the local results JSON")
    parser.add_argument(
        "--no-register", action="store_true",
        help="Skip refreshing dataset items from the evalset (assume already correct)",
    )
    args = parser.parse_args()
    asyncio.run(run(max_cases=args.max_cases, register=not args.no_register,
                    output_dir=args.output_dir))


if __name__ == "__main__":
    main()
