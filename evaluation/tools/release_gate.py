"""Release verdict for the paid eval runners (2026-11-19 audit).

The runners used to print a report and exit 0 even at 0/14 PASS, so a run could not stop a release. Every runner
now ends with ``exit_with_verdict(summary, ...)``: it prints one ``RELEASE: ALLOWED`` / ``RELEASE: BLOCKED`` line,
records the verdict in the results JSON, and exits non-zero when blocked.

Blocked when:
  * no case was evaluated (nothing measured is not a pass);
  * a case was skipped or errored (``total`` > ``evaluated`` — an unscored case hides a regression);
  * the pass rate is below ``min_pass_rate``;
  * a hard metric (one where any miss is a release blocker, e.g. fabricated grounding) is below 1.0;
  * a floored metric (``metric_floors``, e.g. the screen path's claim precision ≥ 0.9) is below its floor.

Paid evals stay manual; this only makes their result unambiguous.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def verdict(
    summary: Dict[str, Any],
    min_pass_rate: float,
    hard_metrics: Iterable[str] = (),
    allow_skipped: bool = False,
    metric_floors: Optional[Mapping[str, float]] = None,
) -> Tuple[bool, List[str]]:
    """(allowed, reasons). ``reasons`` lists every broken rule, empty when allowed."""
    total = int(summary.get("total") or 0)
    evaluated = int(summary.get("evaluated") or 0)
    passed = int(summary.get("passed") or 0)
    metrics = summary.get("metrics") or {}
    reasons: List[str] = []

    if evaluated == 0:
        reasons.append(f"no case evaluated ({total} in the run)")
    else:
        rate = passed / evaluated
        if rate < min_pass_rate:
            reasons.append(f"pass rate {passed}/{evaluated} = {rate:.0%} < {min_pass_rate:.0%}")
    if not allow_skipped and total > evaluated:
        reasons.append(f"{total - evaluated} case(s) skipped or errored, not scored")
    for m in hard_metrics:
        if m not in metrics:
            if evaluated:
                reasons.append(f"hard metric {m} missing from the run")
        elif float(metrics[m]) < 1.0:
            reasons.append(f"hard metric {m} = {metrics[m]} < 1.0")
    for m, floor in (metric_floors or {}).items():
        if m not in metrics:
            if evaluated:
                reasons.append(f"floored metric {m} missing from the run")
        elif float(metrics[m]) < floor:
            reasons.append(f"metric {m} = {metrics[m]} < {floor}")
    return (not reasons), reasons


def exit_with_verdict(
    summary: Dict[str, Any],
    min_pass_rate: float,
    hard_metrics: Iterable[str] = (),
    allow_skipped: bool = False,
    results_path: Optional[Path] = None,
    metric_floors: Optional[Mapping[str, float]] = None,
) -> int:
    """Print the verdict, write it into the results JSON (if given), and return the process exit code."""
    hard_metrics = list(hard_metrics)
    allowed, reasons = verdict(summary, min_pass_rate, hard_metrics, allow_skipped, metric_floors)
    summary["release_gate"] = {
        "allowed": allowed,
        "reasons": reasons,
        "min_pass_rate": min_pass_rate,
        "hard_metrics": hard_metrics,
        "metric_floors": dict(metric_floors or {}),
    }
    if results_path is not None and Path(results_path).exists():
        Path(results_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if allowed:
        print(f"RELEASE: ALLOWED (pass rate ≥ {min_pass_rate:.0%}, hard metrics clean)")
        return 0
    print("RELEASE: BLOCKED — " + "; ".join(reasons), file=sys.stderr)
    print("RELEASE: BLOCKED")
    return 1


def add_gate_args(parser, default_min_pass_rate: float) -> None:
    parser.add_argument(
        "--min-pass-rate", type=float, default=default_min_pass_rate,
        help=f"Release floor as a fraction of evaluated cases (default {default_min_pass_rate})",
    )
