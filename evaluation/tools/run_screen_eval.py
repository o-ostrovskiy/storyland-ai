"""Score the screen path (film & series journeys) against ``screen_v1`` (ai-private#68).

The scorers (``screen_scorers.py``) are deterministic; this runner feeds them.
Until ai-private#67 lands there is no screen discovery to call, so the runner
scores a file of discovery outputs, one per evalset case::

    {"screen_game_of_thrones": {"itinerary": {...}, "screen_context": {...}}, ...}

ai-private#67 produces that file from a live run (G4 spend: Olga's call) and
quotes this runner's numbers in its PR. With ``--log-langfuse`` each case is
written as one item of a Langfuse experiment on the ``screen_v1`` dataset,
through the experiment attributes (``experiment_run.py``), never the retired
``POST /dataset-run-items`` (cut-off 2026-11-16).

Release gate: claim precision ≥ 0.9 over the set, book-pairing honesty = 1.0
(any invented book blocks), every case scored.

Usage:
    python evaluation/tools/run_screen_eval.py --outputs path/to/outputs.json [--log-langfuse]
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.tools.release_gate import add_gate_args, exit_with_verdict  # noqa: E402
from evaluation.tools.screen_scorers import (  # noqa: E402
    BOOK_PAIRING_HONESTY,
    CLAIM_PRECISION,
    CLAIM_PRECISION_FLOOR,
    NAMED_CLAIMS,
    aggregate,
    score_case,
)

DATASET_NAME = "screen_v1"
EVALSET_FILE = "evaluation/screen_v1.evalset.json"

MIN_PASS_RATE = 0.0  # the card gates on the two metrics, not on a per-case pass rate
HARD_METRICS = (BOOK_PAIRING_HONESTY,)
METRIC_FLOORS = {CLAIM_PRECISION: CLAIM_PRECISION_FLOOR}


def load_cases(path: str = EVALSET_FILE) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text())["eval_cases"]


def score_outputs(cases: List[Dict[str, Any]], outputs: Dict[str, Any], run_name: str) -> Dict[str, Any]:
    """Summary in the shape ``release_gate`` reads. A case with no output is unscored (blocks the release)."""
    case_results: List[Dict[str, Any]] = []
    for case in cases:
        eval_id = case["eval_id"]
        if eval_id not in outputs:
            case_results.append({"item_id": eval_id, "status": "skipped", "reason": "no discovery output"})
            continue
        scored = score_case(outputs[eval_id], case["expected_output"])
        case_results.append({"item_id": eval_id, "title": case["screen_title"], **scored})
    evaluated = [c for c in case_results if "pass" in c]
    passed = sum(1 for c in evaluated if c["pass"])
    return {
        "dataset_name": DATASET_NAME,
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "total": len(case_results),
        "evaluated": len(evaluated),
        "passed": passed,
        "failed": len(evaluated) - passed,
        "metrics": aggregate(evaluated),
        "case_results": case_results,
    }


def log_to_langfuse(summary: Dict[str, Any], outputs: Dict[str, Any]) -> None:  # pragma: no cover - live only
    from langfuse import Langfuse

    from common.config import load_config
    from evaluation.tools.experiment_run import link_experiment_item

    config = load_config()
    if not (config.langfuse_public_key and config.langfuse_secret_key):
        raise RuntimeError("Langfuse credentials missing — set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY")
    langfuse = Langfuse(
        secret_key=config.langfuse_secret_key,
        public_key=config.langfuse_public_key,
        host=config.langfuse_host or "https://cloud.langfuse.com",
    )
    dataset = langfuse.get_dataset(DATASET_NAME)
    run_metadata = {"dataset_name": DATASET_NAME, "evaluation_type": "screen_claims"}
    for case in summary["case_results"]:
        if "pass" not in case:
            continue
        with langfuse.start_as_current_observation(
            as_type="span", name=summary["run_name"], metadata={**run_metadata, "eval_id": case["item_id"]},
        ) as span:
            try:
                span.update(input={"title": case["title"]}, output=outputs.get(case["item_id"]))
                span.score_trace(name="case_pass", value=float(case["pass"]))
                for metric in (CLAIM_PRECISION, BOOK_PAIRING_HONESTY, NAMED_CLAIMS):
                    if case.get(metric) is not None:
                        span.score_trace(name=metric, value=float(case[metric]))
            finally:
                link_experiment_item(
                    span,
                    run_name=summary["run_name"],
                    run_metadata=run_metadata,
                    dataset_id=getattr(dataset, "id", None),
                    dataset_name=DATASET_NAME,
                    dataset_item_id=case["item_id"],
                    run_description=f"screen-path claim eval of {DATASET_NAME}",
                )
    langfuse.flush()


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Score screen-path discovery outputs against screen_v1")
    parser.add_argument("--outputs", required=True, help="JSON file: {eval_id: discovery output}")
    parser.add_argument("--evalset", default=EVALSET_FILE)
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--log-langfuse", action="store_true", help="Write the run as a Langfuse experiment")
    add_gate_args(parser, MIN_PASS_RATE)
    args = parser.parse_args(argv)

    outputs = json.loads(Path(args.outputs).read_text())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = score_outputs(load_cases(args.evalset), outputs, run_name=f"screen_eval_{stamp}")
    if args.log_langfuse:
        log_to_langfuse(summary, outputs)

    out_path = Path(args.output_dir) / f"screen_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    for c in summary["case_results"]:
        if "pass" in c:
            print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['item_id']:<30} "
                  f"claim_precision={c[CLAIM_PRECISION]} honesty={c[BOOK_PAIRING_HONESTY]} named={c[NAMED_CLAIMS]}")
        else:
            print(f"  [SKIP] {c['item_id']} ({c['reason']})")
    print(f"Metrics: {json.dumps(summary['metrics'])} | results: {out_path}")
    return exit_with_verdict(
        summary, args.min_pass_rate, HARD_METRICS, results_path=out_path, metric_floors=METRIC_FLOORS,
    )


if __name__ == "__main__":
    sys.exit(main())
