"""
Judge-calibration tooling for the LLM-as-judge scorer.

Two subcommands:

  build-queue  Pull recent judge-scored itineraries from Langfuse dataset
               runs, create per-dimension human score configs (the exact
               SCORING_CRITERIA rubric text), enqueue the traces into a
               Langfuse annotation queue for hand labeling, and write a
               local manifest (the join key for the agreement analysis).

  agreement    After human labels exist: pull ANNOTATION-source scores for
               the manifest's traces and report per-dimension agreement
               (mean absolute difference + direction bias) between the
               judge and the human labels.

The judge has never been anchored to human labels (July 2026 architecture
review item #11); measured run-to-run noise is large (books_v1 ±0.40,
storyland_eval ±0.17), so the human labels gathered through this queue are
what rubric re-anchoring and gate recomputation will be based on.

Requires LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_HOST in the
environment (or .env). Deliberately does NOT use common.config.load_config:
that requires the full app environment (GOOGLE_API_KEY etc.), which this
read-mostly tool doesn't need.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

from common.logging import get_logger
from evaluation.tools.llm_scorer import SCORING_CRITERIA

logger = get_logger("storyland.judge_calibration")

DIMENSIONS = list(SCORING_CRITERIA.keys())
HUMAN_SCORE_PREFIX = "human_"

CALIBRATION_DIR = Path("evaluation/calibration")
DEFAULT_MANIFEST = CALIBRATION_DIR / "queue_manifest_2026-07.json"
DEFAULT_QUEUE_NAME = "judge-calibration-2026-07"
DEFAULT_DATASETS = ["storyland_eval", "books_v1"]
DEFAULT_RUN_PREFIX = "eval_run_202607"


# ---------------------------------------------------------------------------
# Pure selection / analysis logic (unit-tested without a Langfuse client)
# ---------------------------------------------------------------------------

def interleave_by_run(
    per_dataset_runs: Dict[str, List[List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """
    Flatten per-dataset run lists into one candidate order.

    Input maps dataset name -> list of runs (newest first), each run being a
    list of candidate dicts. Output alternates datasets run-by-run (newest
    run of dataset A, newest of B, second-newest of A, ...) so a target-count
    cutoff samples all datasets and prefers the freshest generations.
    """
    ordered: List[Dict[str, Any]] = []
    max_runs = max((len(runs) for runs in per_dataset_runs.values()), default=0)
    for rank in range(max_runs):
        for dataset in per_dataset_runs:
            runs = per_dataset_runs[dataset]
            if rank < len(runs):
                ordered.extend(runs[rank])
    return ordered


def select_candidates(
    candidates: List[Dict[str, Any]],
    target: int = 30,
    max_per_case: int = 2,
) -> List[Dict[str, Any]]:
    """
    Take up to `target` candidates, capping generations per dataset case.

    The cap keeps the pack diverse: the same evalset case generated in two
    different runs is two distinct itineraries (useful for consistency
    checks), but three-plus would crowd out other books.
    """
    selected: List[Dict[str, Any]] = []
    per_case: Counter = Counter()
    for candidate in candidates:
        key = (candidate["dataset"], candidate["item_id"])
        if per_case[key] >= max_per_case:
            continue
        per_case[key] += 1
        selected.append(candidate)
        if len(selected) >= target:
            break
    return selected


def compute_agreement(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Per-dimension judge-vs-human agreement over labeled manifest items.

    Each item carries judge_scores and human_scores dicts (dimension ->
    value, either possibly missing a dimension). Only dimensions present in
    BOTH contribute pairs.

    Returns per dimension: n pairs, mean absolute difference, mean signed
    difference (judge − human; positive = judge scores higher than the
    human), and the count of large disagreements (|Δ| ≥ 2), plus the
    case-level list of those large disagreements.
    """
    per_dimension: Dict[str, Any] = {}
    large_disagreements: List[Dict[str, Any]] = []

    for dimension in DIMENSIONS:
        deltas = []
        for item in items:
            judge = (item.get("judge_scores") or {}).get(dimension)
            human = (item.get("human_scores") or {}).get(dimension)
            if judge is None or human is None:
                continue
            delta = judge - human
            deltas.append(delta)
            if abs(delta) >= 2:
                large_disagreements.append({
                    "dimension": dimension,
                    "dataset": item.get("dataset"),
                    "item_id": item.get("item_id"),
                    "run_name": item.get("run_name"),
                    "book_title": item.get("book_title"),
                    "judge": judge,
                    "human": human,
                })
        if deltas:
            per_dimension[dimension] = {
                "n": len(deltas),
                "mean_abs_diff": round(sum(abs(d) for d in deltas) / len(deltas), 3),
                "bias": round(sum(deltas) / len(deltas), 3),
                "large_disagreements": sum(1 for d in deltas if abs(d) >= 2),
            }
        else:
            per_dimension[dimension] = {"n": 0}

    labeled = [i for i in items if i.get("human_scores")]
    return {
        "n_manifest_items": len(items),
        "n_labeled_items": len(labeled),
        "per_dimension": per_dimension,
        "large_disagreements": large_disagreements,
    }


# ---------------------------------------------------------------------------
# Langfuse plumbing
# ---------------------------------------------------------------------------

def make_langfuse_client():
    """Build a Langfuse client from env creds, or exit with instructions."""
    import os

    load_dotenv()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    host = os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"

    if not (secret_key and public_key):
        print(
            "ERROR: Langfuse credentials not found.\n"
            "Add to .env (repo root, gitignored):\n"
            "  LANGFUSE_SECRET_KEY=sk-lf-...\n"
            "  LANGFUSE_PUBLIC_KEY=pk-lf-...\n"
            "  LANGFUSE_HOST=https://cloud.langfuse.com"
        )
        sys.exit(1)

    from langfuse import Langfuse

    return Langfuse(secret_key=secret_key, public_key=public_key, host=host), host


def collect_candidates(
    langfuse: Any,
    datasets: List[str],
    run_prefix: str,
    max_runs_per_dataset: int = 8,
) -> List[Dict[str, Any]]:
    """
    Gather candidate (dataset, item, run, trace) tuples from matching
    dataset runs, interleaved newest-run-first across datasets.
    """
    per_dataset_runs: Dict[str, List[List[Dict[str, Any]]]] = {}

    for dataset_name in datasets:
        try:
            runs_page = langfuse.api.datasets.get_runs(dataset_name, limit=50)
        except Exception as e:
            logger.warning("dataset_runs_fetch_failed", dataset=dataset_name, error=str(e))
            print(f"  ! Skipping dataset {dataset_name}: {e}")
            continue

        matching = [r for r in runs_page.data if r.name.startswith(run_prefix)]
        matching.sort(key=lambda r: r.created_at, reverse=True)
        matching = matching[:max_runs_per_dataset]

        run_lists: List[List[Dict[str, Any]]] = []
        for run in matching:
            try:
                run_details = langfuse.api.datasets.get_run(dataset_name, run.name)
            except Exception as e:
                logger.warning(
                    "dataset_run_fetch_failed",
                    dataset=dataset_name, run_name=run.name, error=str(e),
                )
                print(f"  ! Skipping run {run.name}: {e}")
                continue
            run_lists.append([
                {
                    "dataset": dataset_name,
                    "item_id": item.dataset_item_id,
                    "run_name": run.name,
                    "trace_id": item.trace_id,
                }
                for item in run_details.dataset_run_items
                if item.trace_id
            ])
        per_dataset_runs[dataset_name] = run_lists
        print(f"  {dataset_name}: {len(matching)} matching runs")

    return interleave_by_run(per_dataset_runs)


def hydrate_candidate(
    langfuse: Any,
    host: str,
    candidate: Dict[str, Any],
    attempts: int = 3,
    retry_delays: tuple = (2, 5),
) -> Optional[Dict[str, Any]]:
    """
    Fetch the candidate's trace; return a manifest entry, or None (logged)
    if the trace has no itinerary output or can't be fetched.

    Trace fetches retry on failure — the Langfuse trace endpoint times out
    transiently under burst reads, and a dropped candidate here silently
    shrinks the labeling pack.
    """
    trace = None
    for attempt in range(attempts):
        try:
            trace = langfuse.api.trace.get(candidate["trace_id"])
            break
        except Exception as e:
            if attempt < attempts - 1:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                logger.warning(
                    "trace_fetch_retry",
                    trace_id=candidate["trace_id"],
                    attempt=attempt + 1,
                    delay=delay,
                )
                time.sleep(delay)
            else:
                logger.warning(
                    "trace_fetch_failed", trace_id=candidate["trace_id"], error=str(e)
                )
                print(
                    f"  ! Dropped {candidate['item_id']} ({candidate['run_name']}): "
                    f"trace fetch failed after {attempts} attempts: {e}"
                )
                return None

    output = trace.output
    if not isinstance(output, dict) or not output:
        print(
            f"  ! Dropped {candidate['item_id']} ({candidate['run_name']}): "
            "trace has no itinerary output"
        )
        return None

    trace_input = trace.input if isinstance(trace.input, dict) else {}
    preferences = trace_input.get("preferences") or None

    judge_scores: Dict[str, Optional[int]] = {}
    for score in trace.scores or []:
        if score.name in DIMENSIONS and getattr(score, "source", None) != "ANNOTATION":
            value = getattr(score, "value", None)
            if value is not None:
                judge_scores[score.name] = int(value)

    if not judge_scores:
        print(
            f"  ! Dropped {candidate['item_id']} ({candidate['run_name']}): "
            "no judge scores on trace"
        )
        return None

    return {
        **candidate,
        "book_title": trace_input.get("book_title") or "unknown",
        "author": trace_input.get("author") or "",
        "has_preferences": bool(preferences),
        "judge_scores": {d: judge_scores.get(d) for d in DIMENSIONS},
        "trace_url": f"{host}{trace.html_path}" if trace.html_path else None,
    }


def ensure_score_configs(langfuse: Any) -> Dict[str, str]:
    """
    Idempotently create the 6 human_<dimension> numeric score configs (1-5),
    each carrying the exact SCORING_CRITERIA rubric text so the criteria are
    visible in the Langfuse annotation drawer. Returns name -> config id.
    """
    existing: Dict[str, str] = {}
    try:
        page = langfuse.api.score_configs.get(limit=100)
        for config in page.data:
            existing[config.name] = config.id
    except Exception as e:
        logger.warning("score_configs_list_failed", error=str(e))

    config_ids: Dict[str, str] = {}
    for dimension in DIMENSIONS:
        name = f"{HUMAN_SCORE_PREFIX}{dimension}"
        if name in existing:
            config_ids[name] = existing[name]
            continue
        created = langfuse.api.score_configs.create(
            name=name,
            data_type="NUMERIC",
            min_value=1,
            max_value=5,
            description=SCORING_CRITERIA[dimension],
        )
        config_ids[name] = created.id
        print(f"  Created score config: {name}")
    return config_ids


def ensure_queue(langfuse: Any, queue_name: str, score_config_ids: List[str]) -> str:
    """Find the annotation queue by name or create it. Returns queue id."""
    try:
        page = langfuse.api.annotation_queues.list_queues(limit=100)
        for queue in page.data:
            if queue.name == queue_name:
                print(f"  Reusing existing queue: {queue_name} ({queue.id})")
                return queue.id
    except Exception as e:
        logger.warning("queue_list_failed", error=str(e))

    queue_obj = langfuse.api.annotation_queues.create_queue(
        name=queue_name,
        score_config_ids=score_config_ids,
        description=(
            "Judge-calibration labeling: hand-score each itinerary on the six "
            "1-5 dimensions using the rubric text on each score. Label from "
            "the trace OUTPUT (the itinerary) and the INPUT (book + "
            "preferences); do not look at the existing judge scores first."
        ),
    )
    print(f"  Created queue: {queue_name} ({queue_obj.id})")
    return queue_obj.id


def get_enqueued_trace_ids(langfuse: Any, queue_id: str) -> set:
    """Trace ids already in the queue — enqueue must be idempotent so a
    re-run (e.g. after transient trace-fetch drops) only backfills."""
    trace_ids = set()
    page = 1
    while True:
        try:
            response = langfuse.api.annotation_queues.list_queue_items(
                queue_id, page=page, limit=100
            )
        except Exception as e:
            logger.warning("queue_items_list_failed", queue_id=queue_id, error=str(e))
            break
        for item in response.data:
            trace_ids.add(item.object_id)
        if len(response.data) < 100:
            break
        page += 1
    return trace_ids


def merge_manifest_items(
    existing: List[Dict[str, Any]],
    new: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Union by trace_id, new entries winning — a re-run must never drop a
    previously manifested (and possibly already-labeled) item just because
    its trace fetch failed this time."""
    merged = {item["trace_id"]: item for item in existing}
    for item in new:
        merged[item["trace_id"]] = item
    return list(merged.values())


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_build_queue(args: argparse.Namespace) -> int:
    langfuse, host = make_langfuse_client()

    print(f"Collecting candidates from datasets: {args.datasets}")
    candidates = collect_candidates(langfuse, args.datasets, args.run_prefix)
    print(f"  {len(candidates)} candidate generations found")

    selected = select_candidates(candidates, target=args.target, max_per_case=2)
    print(f"  {len(selected)} selected (cap 2 generations per case, target {args.target})")

    print("Fetching traces...")
    entries = []
    for candidate in selected:
        entry = hydrate_candidate(langfuse, host, candidate)
        if entry:
            entries.append(entry)
    print(f"  {len(entries)} usable itineraries ({len(selected) - len(entries)} dropped, see above)")

    if not entries:
        print("ERROR: no usable itineraries found — nothing to enqueue.")
        return 1

    shape_counts = Counter(e["has_preferences"] for e in entries)
    dataset_counts = Counter(e["dataset"] for e in entries)
    print(
        f"  Shapes: {shape_counts[True]} with preferences, "
        f"{shape_counts[False]} without; by dataset: {dict(dataset_counts)}"
    )

    if args.dry_run:
        print("\n--dry-run: not writing to Langfuse or disk. Selected items:")
        for entry in entries:
            print(
                f"  {entry['dataset']}/{entry['item_id']} {entry['run_name']} "
                f"{entry['book_title']} prefs={entry['has_preferences']}"
            )
        return 0

    print("Ensuring score configs...")
    config_ids = ensure_score_configs(langfuse)

    print("Ensuring annotation queue...")
    queue_id = args.queue_id or ensure_queue(
        langfuse, args.queue_name, list(config_ids.values())
    )

    print("Enqueuing traces...")
    already_enqueued = get_enqueued_trace_ids(langfuse, queue_id)
    enqueued = 0
    skipped = 0
    for entry in entries:
        if entry["trace_id"] in already_enqueued:
            skipped += 1
            continue
        try:
            langfuse.api.annotation_queues.create_queue_item(
                queue_id, object_id=entry["trace_id"], object_type="TRACE"
            )
            enqueued += 1
        except Exception as e:
            logger.warning(
                "queue_item_create_failed", trace_id=entry["trace_id"], error=str(e)
            )
            print(f"  ! Failed to enqueue {entry['item_id']}: {e}")
    print(f"  {enqueued} enqueued, {skipped} already in queue")

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        with open(manifest_path) as f:
            previous = json.load(f)
        entries = merge_manifest_items(previous.get("items", []), entries)
        print(f"  Merged with existing manifest: {len(entries)} total items")
        shape_counts = Counter(e["has_preferences"] for e in entries)
        dataset_counts = Counter(e["dataset"] for e in entries)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "queue_name": args.queue_name,
        "queue_id": queue_id,
        "langfuse_host": host,
        "score_config_ids": config_ids,
        "dimensions": DIMENSIONS,
        "counts": {
            "total": len(entries),
            "with_preferences": shape_counts[True],
            "without_preferences": shape_counts[False],
            "by_dataset": dict(dataset_counts),
        },
        "items": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}")
    print(
        f"Label here: {host} → Annotation Queues → {args.queue_name} "
        f"({enqueued + skipped} items). Score from the trace output, judge scores untouched."
    )
    return 0


def fetch_human_scores(langfuse: Any, trace_id: str) -> Dict[str, int]:
    """Latest human_<dimension> label score per dimension.

    Labels are identified by the human_ name prefix, NOT by score source:
    labels entered through the annotation UI arrive as ANNOTATION, labels
    entered via the API (e.g. the Claude-labeled 2026-07 pack) as API. The
    prefix is the contract — judge scores are unprefixed.
    """
    try:
        response = langfuse.api.scores_v3.get_many_v3(trace_id=trace_id, limit=100)
    except Exception as e:
        logger.warning("label_scores_fetch_failed", trace_id=trace_id, error=str(e))
        return {}

    human_scores: Dict[str, int] = {}
    # API returns newest first; keep the first (latest) value per dimension.
    for score in response.data:
        name = getattr(score, "name", "") or ""
        if not name.startswith(HUMAN_SCORE_PREFIX):
            continue
        dimension = name[len(HUMAN_SCORE_PREFIX):]
        value = getattr(score, "value", None)
        if dimension in DIMENSIONS and dimension not in human_scores and value is not None:
            human_scores[dimension] = int(value)
    return human_scores


def cmd_agreement(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path} — run build-queue first.")
        return 1
    with open(manifest_path) as f:
        manifest = json.load(f)

    langfuse, _host = make_langfuse_client()

    items = manifest["items"]
    print(f"Fetching human labels for {len(items)} manifest items...")
    for item in items:
        item["human_scores"] = fetch_human_scores(langfuse, item["trace_id"])

    agreement = compute_agreement(items)
    if agreement["n_labeled_items"] == 0:
        print(
            "\nNo human labels found yet (0 of "
            f"{agreement['n_manifest_items']} items carry ANNOTATION scores).\n"
            f"Label the queue '{manifest.get('queue_name')}' in Langfuse first, "
            "then re-run this command."
        )
        return 0

    report_lines = [
        "# Judge vs human agreement",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Manifest:** {manifest_path} ({agreement['n_manifest_items']} items, "
        f"{agreement['n_labeled_items']} labeled)",
        "",
        "Bias = mean(judge − human): positive means the judge scores HIGHER "
        "than the human on that dimension.",
        "",
        "| Dimension | n | Mean abs diff | Bias | \\|Δ\\|≥2 |",
        "|-----------|---|---------------|------|--------|",
    ]
    for dimension, stats in agreement["per_dimension"].items():
        if stats["n"] == 0:
            report_lines.append(f"| {dimension} | 0 | — | — | — |")
        else:
            report_lines.append(
                f"| {dimension} | {stats['n']} | {stats['mean_abs_diff']} | "
                f"{stats['bias']:+.3f} | {stats['large_disagreements']} |"
            )

    if agreement["large_disagreements"]:
        report_lines.extend([
            "",
            "## Large disagreements (|Δ| ≥ 2)",
            "",
            "| Dimension | Case | Book | Judge | Human |",
            "|-----------|------|------|-------|-------|",
        ])
        for disagreement in agreement["large_disagreements"]:
            report_lines.append(
                f"| {disagreement['dimension']} | "
                f"{disagreement['dataset']}/{disagreement['item_id']} | "
                f"{disagreement['book_title']} | {disagreement['judge']} | "
                f"{disagreement['human']} |"
            )

    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    json_path = report_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(agreement, f, indent=2)

    print("\n".join(report_lines))
    print(f"\nReport written: {report_path} (+ {json_path})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-judge calibration tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-queue", help="Build the Langfuse human-labeling queue"
    )
    build.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    build.add_argument(
        "--run-prefix", default=DEFAULT_RUN_PREFIX,
        help="Only dataset runs whose name starts with this prefix",
    )
    build.add_argument("--target", type=int, default=30, help="Target item count")
    build.add_argument("--queue-name", default=DEFAULT_QUEUE_NAME)
    build.add_argument(
        "--queue-id", default=None,
        help="Use an existing queue id instead of finding/creating by name",
    )
    build.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    build.add_argument(
        "--dry-run", action="store_true",
        help="Select and print items without writing to Langfuse or disk",
    )
    build.set_defaults(func=cmd_build_queue)

    agreement = subparsers.add_parser(
        "agreement", help="Compute judge-vs-human agreement from labels"
    )
    agreement.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    agreement.add_argument(
        "--output", default=str(CALIBRATION_DIR / "agreement_report.md")
    )
    agreement.set_defaults(func=cmd_agreement)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
