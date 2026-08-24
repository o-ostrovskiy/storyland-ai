"""Run the local-atmosphere ("book near me") eval against Langfuse.

The local-atmosphere flow composes an itinerary near the user's location whose
mood evokes a chosen book — used when the reader cannot travel to the book's
actual setting. It has its own runner because the scheduled itinerary eval
(``run_scheduled_eval.py``) drives the book→place discovery/composition
workflows, which is the WRONG flow for these cases: a local-atmosphere case
run through it would produce plausible-but-bogus judge scores.

For each ``local_atmosphere_v1`` dataset item this script:
  1. drives the live ``WorkflowExecutor.local_atmosphere`` flow (the same code
     path as ``POST /itinerary/local-atmosphere``, including extraction and
     validation);
  2. runs the deterministic gate: the result must validate as a
     ``ComposerEnvelope``, plus an opportunistic radius check over any geo
     fields (``CityStop`` carries no coordinates today, so this normally
     reports ``no_geo_fields`` and radius adherence rides the per-case
     ``geographical_accuracy`` judge criterion);
  3. scores quality with the existing LLM-as-judge path
     (``llm_scorer.score_itinerary``, 6 dimensions);
  4. logs a Langfuse dataset run with per-case scores. Per the eval protocol,
     ``preference_adherence`` is reported ONLY for cases that carry
     ``user:preferences`` (the judge always emits it, but for preference-free
     cases it is judged against nothing and would pollute aggregates), and the
     report includes per-preference-shape aggregation plus a ``mechanism``
     section.

Exit code reflects infrastructure health (missing credentials, Langfuse
unavailable), not case verdicts — consume ``passed``/``failed`` from the
results JSON, mirroring ``run_place_to_book_eval.py``.

Usage:
    python evaluation/tools/run_local_atmosphere_eval.py [--max-cases N] [--no-register]
"""

import asyncio
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.logging import get_logger
from evaluation.tools.llm_scorer import _DEFAULT_JUDGE_MODEL
from evaluation.tools.experiment_run import link_experiment_item
from models.itinerary import ComposerEnvelope

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    LANGFUSE_AVAILABLE = False

logger = get_logger("storyland.local_atmosphere_eval")

DATASET_NAME = "local_atmosphere_v1"
EVALSET_FILE = "evaluation/local_atmosphere_v1.evalset.json"

# Judge dimensions reported for every case; preference_adherence is added only
# for cases that carry user:preferences (eval protocol).
JUDGE_DIMS_ALWAYS = (
    "book_relevance",
    "completeness",
    "actionability",
    "geographical_accuracy",
    "engagement",
)

# Haversine measures straight-line distance; radius_km is a travel radius
# ("~1 hour of driving"), so allow generous slack before calling a stop out.
GEO_SLACK = 1.5


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two WGS84 points, in km."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def scan_geo_points(obj: Any) -> List[Tuple[float, float]]:
    """Recursively collect (lat, lng) pairs from any dicts carrying them.

    Opportunistic: ``CityStop`` has no coordinate fields today, so this
    usually returns ``[]`` — but if geo fields are ever added to the payload,
    the radius check starts biting without a runner change. Accepts both
    ``lat``/``lng`` and ``latitude``/``longitude`` spellings; ignores dicts
    with only one half of the pair or non-numeric values.
    """
    points: List[Tuple[float, float]] = []
    if isinstance(obj, dict):
        for lat_key, lng_key in (("lat", "lng"), ("latitude", "longitude")):
            lat, lng = obj.get(lat_key), obj.get(lng_key)
            if isinstance(lat, (int, float)) and isinstance(lng, (int, float)) \
                    and not isinstance(lat, bool) and not isinstance(lng, bool):
                points.append((float(lat), float(lng)))
        for value in obj.values():
            points.extend(scan_geo_points(value))
    elif isinstance(obj, list):
        for item in obj:
            points.extend(scan_geo_points(item))
    return points


def check_radius(
    itinerary: dict,
    lat: float,
    lng: float,
    radius_km: float,
    slack: float = GEO_SLACK,
) -> Dict[str, Any]:
    """Opportunistic radius gate over any geo fields in the itinerary payload.

    Returns ``radius_check``: ``"pass"`` (all found points within
    ``radius_km * slack``), ``"fail"`` (at least one outside), or
    ``"no_geo_fields"`` (nothing to check — NOT a failure; radius adherence
    is then asserted only via the judge's geographical_accuracy criterion).
    """
    points = scan_geo_points(itinerary)
    if not points:
        return {"geo_points_found": 0, "outside_radius": 0,
                "radius_check": "no_geo_fields"}
    outside = sum(
        1 for (plat, plng) in points
        if haversine_km(lat, lng, plat, plng) > radius_km * slack
    )
    return {
        "geo_points_found": len(points),
        "outside_radius": outside,
        "radius_check": "pass" if outside == 0 else "fail",
    }


def validate_envelope(
    itinerary: dict, suggestions: list
) -> Tuple[bool, Optional[str]]:
    """Validate the flow output as a ComposerEnvelope (the formatter contract)."""
    try:
        ComposerEnvelope.model_validate(
            {"itinerary": itinerary, "suggestions": suggestions}
        )
        return True, None
    except Exception as e:
        return False, str(e)


def check_local_atmosphere_case(
    itinerary: dict,
    suggestions: list,
    lat: float,
    lng: float,
    radius_km: float,
) -> Dict[str, Any]:
    """Deterministic gate for one local-atmosphere result."""
    envelope_valid, envelope_error = validate_envelope(itinerary, suggestions)
    radius = check_radius(itinerary, lat, lng, radius_km)
    cities = itinerary.get("cities", []) if isinstance(itinerary, dict) else []
    num_stops = sum(len(c.get("stops", [])) for c in cities if isinstance(c, dict))
    return {
        "envelope_valid": envelope_valid,
        "envelope_error": envelope_error,
        **radius,
        "num_cities": len(cities),
        "num_stops": num_stops,
        "num_suggestions": len(suggestions),
        "deterministic_pass": envelope_valid and radius["radius_check"] != "fail",
    }


def aggregate_by_preference_shape(case_results: List[dict]) -> Dict[str, Any]:
    """Per-shape score aggregation (eval protocol).

    Splits judged cases on ``has_preferences``. The preference-free aggregate
    EXCLUDES ``preference_adherence``: the judge always emits all 6 dimensions
    (``llm_scorer.ItineraryScores``), but for a case with no preferences the
    dimension is judged against nothing and would only add noise.
    """
    def _avg(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 2) if values else None

    def _aggregate(cases: List[dict], dims: Tuple[str, ...]) -> Dict[str, Any]:
        avg_scores = {}
        for dim in dims:
            values = [
                c["scores"][dim] for c in cases
                if c.get("scores") and dim in c["scores"]
            ]
            if values:
                avg_scores[dim] = _avg(values)
        all_values = [v for dim in dims for v in (
            [c["scores"][dim] for c in cases if c.get("scores") and dim in c["scores"]]
        )]
        return {"n": len(cases), "avg_scores": avg_scores, "average": _avg(all_values)}

    scored = [c for c in case_results if c.get("scores")]
    with_prefs = [c for c in scored if c.get("has_preferences")]
    without_prefs = [c for c in scored if not c.get("has_preferences")]
    return {
        "with_preferences": _aggregate(
            with_prefs, JUDGE_DIMS_ALWAYS + ("preference_adherence",)
        ),
        "without_preferences": {
            **_aggregate(without_prefs, JUDGE_DIMS_ALWAYS),
            "note": "preference_adherence excluded — no preferences to adhere to",
        },
    }


def build_mechanism_section() -> Dict[str, Any]:
    """The ``mechanism`` block every eval report must carry (eval protocol)."""
    return {
        "deterministic": [
            "ComposerEnvelope schema validation (models/itinerary.py) over the "
            "ItineraryReady payload",
            "opportunistic haversine radius check over any lat/lng fields in the "
            "payload (normally no_geo_fields — CityStop carries no coordinates; "
            "radius adherence is asserted via each case's geographical_accuracy "
            "judge criterion)",
        ],
        "llm_judge": {
            "scorer": "evaluation/tools/llm_scorer.score_itinerary",
            "model": _DEFAULT_JUDGE_MODEL,
            "dimensions": list(JUDGE_DIMS_ALWAYS) + ["preference_adherence"],
            "note": "preference_adherence reported only for cases with "
                    "user:preferences; per-shape aggregates split on that flag",
        },
        "pass_rule": "deterministic_pass AND judge completed (judge values are "
                     "trend metrics, not thresholds)",
        "harness": "WorkflowExecutor.local_atmosphere (production code path, "
                   "cache disabled, in-memory sessions)",
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

    # Refresh the dataset items from the evalset so they carry the location
    # inputs and quality criteria.
    if register:
        from evaluation.tools.langfuse_eval import LangfuseEvalPipeline
        LangfuseEvalPipeline().create_dataset_from_evalset(
            EVALSET_FILE, dataset_name=DATASET_NAME
        )
        logger.info("local_atmosphere_dataset_refreshed", dataset=DATASET_NAME)

    # Production executor, eval-tuned: no discovery cache (every case pays a
    # fresh run — a replayed result would eval the cache, not the flow) and
    # in-memory sessions.
    from core.executor import WorkflowExecutor
    from core.types import ExecutorConfig
    from core.events import ItineraryReady, WorkflowError
    from evaluation.tools.llm_scorer import score_itinerary

    exec_config = ExecutorConfig.from_config(config)
    exec_config.cache_enabled = False
    exec_config.use_database = False
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

    run_name = f"la_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # model_under_test labels every record: baselines are model-bound, and a
    # model lift must read as a re-baseline, not a quality regression.
    run_metadata = {
        "dataset_name": DATASET_NAME,
        "evaluation_type": "local_atmosphere",
        "model_under_test": config.model_name,
    }
    case_results: List[Dict[str, Any]] = []

    logger.info("starting_local_atmosphere_eval", dataset=DATASET_NAME,
                cases=len(items), run_name=run_name)

    for item in items:
        inp = item.input or {}
        meta = item.metadata or {}
        book_title = (inp.get("book_title") or "").strip()
        author = (inp.get("author") or "").strip()
        location_label = inp.get("location_label")
        lat, lng = inp.get("lat"), inp.get("lng")
        radius_km = inp.get("radius_km", 80)
        preferences = (
            meta.get("session_input", {}).get("state", {}).get("user:preferences")
        )
        quality_criteria = meta.get("quality_criteria")

        if not (book_title and location_label and lat is not None and lng is not None):
            logger.warning("local_atmosphere_item_missing_input", item_id=item.id)
            case_results.append({
                "item_id": item.id, "status": "skipped",
                "reason": "missing book_title/location_label/lat/lng in input",
            })
            continue

        with langfuse.start_as_current_observation(
            as_type="span", name=run_name, metadata={**run_metadata, "eval_id": item.id},
        ) as root_span:
            try:
                itinerary_ev: Optional[ItineraryReady] = None
                error_ev: Optional[WorkflowError] = None
                async for ev in executor.local_atmosphere(
                    book_title=book_title,
                    author=author,
                    location_label=location_label,
                    lat=float(lat),
                    lng=float(lng),
                    radius_km=int(radius_km),
                    preferences=preferences,
                    user_id="eval_user",
                ):
                    if isinstance(ev, ItineraryReady):
                        itinerary_ev = ev
                    elif isinstance(ev, WorkflowError):
                        error_ev = ev

                if itinerary_ev is None:
                    reason = error_ev.message if error_ev else "no ItineraryReady event"
                    root_span.update(
                        input={"book_title": book_title, "location_label": location_label},
                        output={"status": "failed", "reason": reason},
                    )
                    root_span.score_trace(
                        name="case_pass", value=0.0,
                        comment=f"{book_title} @ {location_label}: flow failed — {reason}",
                    )
                    case_results.append({
                        "item_id": item.id, "status": "failed", "pass": False,
                        "has_preferences": bool(preferences), "reason": reason,
                        "run_name": run_name,
                    })
                    logger.warning("local_atmosphere_case_flow_failed",
                                   item_id=item.id, reason=reason)
                    continue

                checks = check_local_atmosphere_case(
                    itinerary_ev.itinerary, itinerary_ev.suggestions,
                    float(lat), float(lng), float(radius_km),
                )

                scores_data: Optional[Dict[str, Any]] = None
                judge_ok = False
                try:
                    scores = await score_itinerary(
                        api_key=config.google_api_key,
                        book_title=book_title,
                        author=author,
                        input_text=(
                            f"{book_title} near {location_label} "
                            f"(within ~{radius_km} km)"
                        ),
                        itinerary=itinerary_ev.itinerary,
                        preferences=preferences or {},
                        quality_criteria=quality_criteria,
                        expected_output=None,
                    )
                    judge_ok = True
                    scores_data = {dim: getattr(scores, dim) for dim in JUDGE_DIMS_ALWAYS}
                    if preferences:
                        scores_data["preference_adherence"] = scores.preference_adherence
                    scores_data["judge_average"] = round(
                        sum(scores_data[d] for d in scores_data) / len(scores_data), 2
                    )
                except Exception as e:  # judge failure fails the case: quality IS the eval
                    logger.warning("local_atmosphere_scoring_failed",
                                   item_id=item.id, error=str(e))

                passed = checks["deterministic_pass"] and judge_ok

                root_span.update(
                    input={
                        "book_title": book_title, "author": author,
                        "location_label": location_label,
                        "lat": lat, "lng": lng, "radius_km": radius_km,
                        "preferences": preferences or {},
                    },
                    output={"checks": checks, "scores": scores_data},
                )
                root_span.score_trace(
                    name="case_pass", value=float(passed),
                    comment=f"{book_title} @ {location_label}: "
                            f"envelope={checks['envelope_valid']}, "
                            f"radius={checks['radius_check']}, judge_ok={judge_ok}",
                )
                root_span.score_trace(
                    name="envelope_valid", value=float(checks["envelope_valid"]),
                    comment="1.0 = ItineraryReady payload validates as ComposerEnvelope",
                )
                if checks["radius_check"] != "no_geo_fields":
                    root_span.score_trace(
                        name="radius_within",
                        value=float(checks["radius_check"] == "pass"),
                        comment=f"{checks['outside_radius']}/{checks['geo_points_found']} "
                                f"geo points outside {radius_km} km × {GEO_SLACK}",
                    )
                if scores_data:
                    for dim, value in scores_data.items():
                        if dim == "judge_average":
                            continue
                        root_span.score_trace(
                            name=dim, value=float(value),
                            comment="LLM-as-judge (1-5)",
                        )
                    root_span.score_trace(
                        name="judge_average", value=scores_data["judge_average"],
                        comment="Mean of reported judge dimensions "
                                "(preference_adherence only when preferences exist)",
                    )

                case_results.append({
                    "item_id": item.id, "status": "evaluated", "pass": bool(passed),
                    "has_preferences": bool(preferences),
                    "book_title": book_title, "location_label": location_label,
                    "radius_km": radius_km, "checks": checks,
                    "scores": scores_data, "judge_ok": judge_ok,
                    "run_name": run_name,
                })
                logger.info(
                    "local_atmosphere_case_evaluated", item_id=item.id,
                    passed=passed, envelope_valid=checks["envelope_valid"],
                    radius_check=checks["radius_check"], judge_ok=judge_ok,
                )
            finally:
                # MYS-951 — experiment attributes, not the retired
                # POST /dataset-run-items. See experiment_run.py.
                link_experiment_item(
                    root_span,
                    run_name=run_name,
                    run_metadata=run_metadata,
                    dataset_id=getattr(dataset, "id", None),
                    dataset_item_id=item.id,
                )

    langfuse.flush()
    await executor.close()

    evaluated = [c for c in case_results if "pass" in c]
    passed = sum(1 for c in evaluated if c["pass"])
    summary = {
        "dataset_name": DATASET_NAME,
        "run_name": run_name,
        "model_under_test": config.model_name,
        "timestamp": datetime.now().isoformat(),
        "total": len(case_results),
        "evaluated": len(evaluated),
        "passed": passed,
        "failed": len(evaluated) - passed,
        "skipped": sum(1 for c in case_results if c.get("status") == "skipped"),
        "aggregates_by_preference_shape": aggregate_by_preference_shape(case_results),
        "mechanism": build_mechanism_section(),
        "case_results": case_results,
    }

    out_path = Path(output_dir) / f"local_atmosphere_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print(f"local-atmosphere eval — {DATASET_NAME}")
    print("=" * 72)
    for c in case_results:
        if "pass" in c:
            checks = c.get("checks", {})
            judge_avg = (c.get("scores") or {}).get("judge_average", "-")
            print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['item_id']:<22} "
                  f"prefs={'y' if c['has_preferences'] else 'n'} "
                  f"envelope={checks.get('envelope_valid')} "
                  f"radius={checks.get('radius_check', '-'):<13} "
                  f"judge_avg={judge_avg}")
        else:
            print(f"  [SKIP] {c['item_id']} ({c.get('reason')})")
    print("-" * 72)
    print("Per-shape aggregates (eval protocol):")
    print(json.dumps(summary["aggregates_by_preference_shape"], indent=2))
    print("mechanism:")
    print(json.dumps(summary["mechanism"], indent=2))
    print("=" * 72)
    print(f"Result: {passed}/{len(evaluated)} PASS | Langfuse run: {run_name}")
    print(f"Results saved to {out_path}")
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the local-atmosphere eval against Langfuse"
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
