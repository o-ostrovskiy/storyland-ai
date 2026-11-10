"""Run the place→book reverse-routing grounding eval against Langfuse.

The itinerary eval (``run_scheduled_eval.py``) composes book→place itineraries
and scores them with an LLM judge. The place→book capability is the *reverse*
(``PlaceToBookResolver``: a destination → grounded, labelled book candidates),
so it needs its own runner: a deterministic grounding check, not an LLM judge.

For each ``place_to_book_v1`` dataset item this script:
  1. resolves the place through the live ``PlaceToBookResolver``;
  2. checks the evalset's grounding expectations (a real place returns at least
     ``min_literal`` grounded ``literal`` candidates; a fictional/ungroundable
     place returns the clean not-found state with no fabricated list);
  3. logs a Langfuse dataset run with per-case scores, so the result shows up as
     a proper experiment under the ``place_to_book_v1`` dataset alongside the
     itinerary runs.

Usage:
    python evaluation/tools/run_place_to_book_eval.py [--max-cases N] [--no-register]
"""

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.tools.experiment_run import link_experiment_item  # noqa: E402

from common.config import load_config
from common.logging import configure_logging, get_logger

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    LANGFUSE_AVAILABLE = False

logger = get_logger("storyland.place_to_book_eval")

DATASET_NAME = "place_to_book_v1"
EVALSET_FILE = "evaluation/place_to_book_v1.evalset.json"


def _norm_title(title: Optional[str]) -> str:
    """Casefolded title without punctuation or a leading article, for comparison."""
    text = re.sub(r"[^\w\s]", " ", (title or "").casefold())
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^(the|a|an) ", "", text)


def _norm_author(author: Optional[str]) -> str:
    return re.sub(r"[^\w]", "", (author or "").casefold())


def _parse_example(example: str) -> Optional[tuple]:
    """``"Title (Author)"`` → (normalized title, normalized author)."""
    match = re.match(r"^(.*)\(([^()]*)\)\s*$", example or "")
    if not match:
        return None
    return _norm_title(match.group(1)), _norm_author(match.group(2))


def score_case(
    result,
    expect: str,
    min_literal: Optional[int],
    expected_examples: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Score one resolved place against its grounding expectation.

    Returns a dict with the boolean verdict plus the numeric signals logged to
    Langfuse. ``literal`` candidates that survive ``enforce_label_invariants``
    are, by construction, grounded (they name a real ``maps_to``); ``vibe``
    candidates carry ``maps_to=None``.
    """
    candidates = [
        {
            "title": c.title,
            "author": c.author,
            "match_type": c.match_type,
            "maps_to": c.maps_to,
        }
        for c in result.candidates
    ]
    n_literal_grounded = sum(
        1 for c in candidates
        if c["match_type"] == "literal" and (c["maps_to"] or "").strip()
    )
    # Invariant guard: literal => has maps_to; vibe => maps_to is null.
    grounding_clean = all(
        (c["match_type"] == "literal" and (c["maps_to"] or "").strip())
        or (c["match_type"] == "vibe" and not (c["maps_to"] or "").strip())
        for c in candidates
    )

    if expect == "found":
        passed = bool(result.found) and n_literal_grounded >= (min_literal or 1)
    else:  # not_found
        passed = (not result.found) and len(candidates) == 0

    found_classification = float(result.found == (expect == "found"))

    # ai-private#71: never the same book twice in one answer. A book is its
    # title + author: same-title books by different authors are not repeats.
    seen_books = [(_norm_title(c["title"]), _norm_author(c["author"])) for c in candidates]
    n_duplicate_titles = len(seen_books) - len(set(seen_books))
    # ai-private#71: when a returned book is one of the evalset's known
    # examples, its author must be that book's real author. Books the evalset
    # doesn't name can't be checked offline and are not counted either way.
    known = dict(filter(None, (_parse_example(e) for e in expected_examples or [])))
    n_author_mismatches = sum(
        1
        for c in candidates
        if _norm_title(c["title"]) in known
        and _norm_author(c["author"]) != known[_norm_title(c["title"])]
    )
    passed = passed and n_duplicate_titles == 0 and n_author_mismatches == 0

    return {
        "expect": expect,
        "found": bool(result.found),
        "n_candidates": len(candidates),
        "n_literal_grounded": n_literal_grounded,
        "min_literal": min_literal,
        "grounding_clean": bool(grounding_clean),
        "found_classification": found_classification,
        "n_duplicate_titles": n_duplicate_titles,
        "n_author_mismatches": n_author_mismatches,
        "pass": bool(passed),
        "candidates": candidates,
    }


def aggregate_metrics(evaluated: List[Dict[str, Any]]) -> Dict[str, float]:
    """Run-level means of every per-case signal, for before/after comparison."""
    n = len(evaluated)
    if not n:
        return {}

    def mean(fn) -> float:
        return round(sum(fn(c) for c in evaluated) / n, 3)

    return {
        "case_pass": mean(lambda c: float(c["pass"])),
        "found_classification": mean(lambda c: c["found_classification"]),
        "grounding_clean": mean(lambda c: float(c["grounding_clean"])),
        "no_duplicate_titles": mean(lambda c: float(c["n_duplicate_titles"] == 0)),
        "authors_correct": mean(lambda c: float(c["n_author_mismatches"] == 0)),
        "literal_grounded_mean": mean(lambda c: float(c["n_literal_grounded"])),
    }


async def run(max_cases: Optional[int], register: bool, output_dir: str) -> Dict[str, Any]:
    config = load_config()
    configure_logging(level=config.log_level, enable_adk_debug=config.enable_adk_debug)

    if not LANGFUSE_AVAILABLE:
        raise RuntimeError("langfuse package not available")
    if not (config.langfuse_public_key and config.langfuse_secret_key):
        raise RuntimeError(
            "Langfuse credentials missing — set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"
        )

    # Refresh the dataset items from the evalset so they carry the place + the
    # grounding expectations (the generic itinerary extractor stored neither).
    if register:
        from evaluation.tools.langfuse_eval import LangfuseEvalPipeline
        LangfuseEvalPipeline().create_dataset_from_evalset(
            EVALSET_FILE, dataset_name=DATASET_NAME
        )
        logger.info("place_to_book_dataset_refreshed", dataset=DATASET_NAME)

    from api import dependencies
    await dependencies.initialize()
    resolver = dependencies.get_place_to_book_resolver()

    langfuse = Langfuse(
        secret_key=config.langfuse_secret_key,
        public_key=config.langfuse_public_key,
        host=config.langfuse_host or "https://cloud.langfuse.com",
    )
    dataset = langfuse.get_dataset(DATASET_NAME)
    items = list(dataset.items)
    if max_cases is not None:
        items = items[:max_cases]

    run_name = f"p2b_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_metadata = {"dataset_name": DATASET_NAME, "evaluation_type": "place_to_book_grounding"}
    case_results: List[Dict[str, Any]] = []

    logger.info("starting_place_to_book_eval", dataset=DATASET_NAME, cases=len(items), run_name=run_name)

    for item in items:
        place = (item.input or {}).get("place")
        meta = item.metadata or {}
        expect = meta.get("expect", "found")
        min_literal = meta.get("min_literal")
        expected_examples = meta.get("expected_literal_examples") or []

        if not place:
            logger.warning("place_to_book_item_missing_place", item_id=item.id)
            case_results.append({"item_id": item.id, "status": "skipped", "reason": "no place in input"})
            continue

        with langfuse.start_as_current_observation(
            as_type="span", name=run_name, metadata={**run_metadata, "eval_id": item.id},
        ) as root_span:
            try:
                result = await resolver.resolve(place)
                scored = score_case(result, expect, min_literal, expected_examples)
                try:
                    root_span.update(input={"place": place}, output=scored)
                except Exception:  # pragma: no cover - richness only, never fatal
                    pass
                root_span.score_trace(
                    name="case_pass", value=float(scored["pass"]),
                    comment=f"{place}: expect={expect}, found={scored['found']}, "
                            f"literal_grounded={scored['n_literal_grounded']}/{min_literal}",
                )
                root_span.score_trace(
                    name="found_classification", value=scored["found_classification"],
                    comment="1.0 = found/not-found matches the expected grounding state",
                )
                root_span.score_trace(
                    name="literal_grounded", value=float(scored["n_literal_grounded"]),
                    comment="Count of grounded `literal` candidates (each names a real maps_to)",
                )
                root_span.score_trace(
                    name="grounding_clean", value=float(scored["grounding_clean"]),
                    comment="1.0 = every literal has a maps_to and every vibe has none (no fabrication)",
                )
                root_span.score_trace(
                    name="no_duplicate_titles", value=float(scored["n_duplicate_titles"] == 0),
                    comment="1.0 = no title appears twice in the answer (ai-private#71)",
                )
                root_span.score_trace(
                    name="authors_correct", value=float(scored["n_author_mismatches"] == 0),
                    comment="1.0 = every returned evalset-known book names its real author",
                )
                case_results.append({"item_id": item.id, "place": place, "run_name": run_name, **scored})
                logger.info(
                    "place_to_book_case_evaluated", item_id=item.id, place=place,
                    passed=scored["pass"], found=scored["found"],
                    literal_grounded=scored["n_literal_grounded"],
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
                    run_description=f"place→book grounding eval of {DATASET_NAME}",
                )

    langfuse.flush()

    evaluated = [c for c in case_results if "pass" in c]
    passed = sum(1 for c in evaluated if c["pass"])
    summary = {
        "dataset_name": DATASET_NAME,
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "total": len(case_results),
        "evaluated": len(evaluated),
        "passed": passed,
        "failed": len(evaluated) - passed,
        "metrics": aggregate_metrics(evaluated),
        "case_results": case_results,
    }

    out_path = Path(output_dir) / f"place_to_book_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print(f"place→book grounding eval — {DATASET_NAME}")
    print("=" * 60)
    for c in case_results:
        if "pass" in c:
            print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['item_id']:<24} "
                  f"expect={c['expect']:<9} found={c['found']} "
                  f"literal_grounded={c['n_literal_grounded']}")
        else:
            print(f"  [SKIP] {c['item_id']} ({c.get('reason')})")
    print("=" * 60)
    print(f"Result: {passed}/{len(evaluated)} PASS | Langfuse run: {run_name}")
    print(f"Metrics: {json.dumps(summary['metrics'])}")
    print(f"Results saved to {out_path}")
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the place→book grounding eval against Langfuse")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases (default: all)")
    parser.add_argument("--output-dir", default="evaluation/results", help="Directory for the local results JSON")
    parser.add_argument(
        "--no-register", action="store_true",
        help="Skip refreshing dataset items from the evalset (assume already correct)",
    )
    args = parser.parse_args()
    asyncio.run(run(max_cases=args.max_cases, register=not args.no_register, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
