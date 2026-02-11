"""
Scheduled evaluation runner for StoryLand AI.

This script runs automated evaluations against the Langfuse dataset and logs results
for tracking quality over time. Designed to be run via cron or GitHub Actions.
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logging import get_logger
from common.config import load_config
from tools.langfuse_eval import LangfuseEvalPipeline

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

logger = get_logger("storyland.scheduled_eval")


async def run_evaluation_on_dataset(
    dataset_name: str,
    config: Any,
    max_cases: int = 10,
) -> Dict[str, Any]:
    """
    Run evaluation on a Langfuse dataset.

    Args:
        dataset_name: Name of the Langfuse dataset
        config: Application configuration
        max_cases: Maximum number of test cases to evaluate

    Returns:
        Evaluation results summary
    """
    logger.info("starting_evaluation", dataset_name=dataset_name, max_cases=max_cases)

    if not LANGFUSE_AVAILABLE:
        logger.error("langfuse_not_available")
        return {"error": "Langfuse not installed"}

    # Initialize Langfuse client
    langfuse = Langfuse(
        secret_key=config.langfuse_secret_key,
        public_key=config.langfuse_public_key,
        host=config.langfuse_host or "https://cloud.langfuse.com",
    )

    try:
        # Get dataset items
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

    # Fetch dataset items
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

    total_cases = len(items)
    evaluated_cases = 0
    failed_cases = 0
    case_results = []

    logger.info(
        "dataset_loaded",
        dataset_name=dataset_name,
        total_cases=total_cases,
        max_cases=max_cases,
    )

    # Limit to max_cases
    items_to_evaluate = items[:max_cases]

    # Create run name for this evaluation batch
    run_name = f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_metadata = {
        "dataset_name": dataset_name,
        "evaluation_type": "scheduled",
        "max_cases": max_cases,
    }

    # Evaluate each dataset item
    for item in items_to_evaluate:
        try:
            # Extract input from dataset item
            input_data = item.input
            expected_output = item.expected_output

            logger.info(
                "evaluating_case",
                dataset_name=dataset_name,
                item_id=item.id,
            )

            # Create a Langfuse run for this dataset item
            # Using the context manager pattern from Langfuse SDK
            with item.run(
                run_name=run_name,
                run_description=f"Scheduled evaluation of {dataset_name}",
                run_metadata=run_metadata,
            ) as root_span:
                # Run the StoryLand workflow
                # Note: This is a simplified evaluation - full workflow requires human interaction
                # For automated evaluation, we would need to mock region selection
                result = await _run_evaluation_case(
                    input_data=input_data,
                    expected_output=expected_output,
                    config=config,
                    root_span=root_span,
                )

                # Score the trace if evaluation produced a result
                if result.get("status") == "evaluated":
                    # Placeholder score - real implementation would use LLM-as-judge
                    # See issue #96 for LLM-based scoring implementation
                    root_span.score_trace(
                        name="evaluation_status",
                        value=1.0,
                        comment="Placeholder - workflow execution needed (see issue #95)",
                    )

            case_results.append({
                "item_id": item.id,
                "status": "success",
                "run_name": run_name,
            })

            evaluated_cases += 1

            logger.info(
                "case_evaluated",
                dataset_name=dataset_name,
                item_id=item.id,
                run_name=run_name,
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

    # Flush Langfuse client to ensure all traces are sent
    langfuse.flush()

    results = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "total_cases": total_cases,
        "evaluated_cases": evaluated_cases,
        "failed_cases": failed_cases,
        "case_results": case_results,
    }

    logger.info(
        "evaluation_complete",
        dataset=dataset_name,
        evaluated=evaluated_cases,
        failed=failed_cases,
        total=total_cases,
        message=f"View results at {config.langfuse_host}",
    )

    return results


async def _run_evaluation_case(
    input_data: Dict[str, Any],
    expected_output: Any,
    config: Any,
    root_span: Any,
) -> Dict[str, Any]:
    """
    Run evaluation for a single test case.

    Note: This is a simplified evaluation runner. The full StoryLand workflow
    requires human interaction for region selection, which is not feasible
    in automated evaluations.

    For proper evaluation, you would need to:
    1. Mock region selection with a deterministic choice
    2. Run the full metadata -> discovery -> composition workflow
    3. Compare output against expected_output
    4. Score using LLM-as-judge (see issue #96)

    Args:
        input_data: Input from dataset item
        expected_output: Expected output (if available)
        config: Application configuration
        root_span: Langfuse root span (from item.run() context manager)

    Returns:
        Evaluation result
    """
    try:
        # Extract the book query from input
        text = input_data.get('text') or input_data.get('starting_prompt', '')

        if not text:
            logger.warning("no_input_text", input_data=input_data)
            return {"status": "skipped", "reason": "No input text"}

        # For automated evaluation, we would run a simplified workflow here
        # This is a placeholder - actual implementation would call the agent
        # See issue #95 for workflow execution implementation
        logger.info("would_evaluate", input_text=text)

        # Placeholder result - in a real implementation, this would run the agent
        result = {
            "status": "evaluated",
            "input": text,
            "note": (
                "This is a placeholder evaluation. Full workflow implementation "
                "requires mocking human interaction for region selection. "
                "See GitHub issues #95 (workflow execution), #96 (LLM scoring), "
                "#97 (automated region selection)."
            ),
        }

        return result

    except Exception as e:
        logger.error("evaluation_case_error", error=str(e))
        raise


async def run_all_evaluations(
    output_dir: str = "evaluation/results",
    max_cases_per_dataset: int = 10,
    dataset_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run evaluations on specified or all available datasets.

    Args:
        output_dir: Directory to save evaluation results
        max_cases_per_dataset: Maximum cases to evaluate per dataset
        dataset_names: List of dataset names to evaluate. If None, evaluates
                      datasets found in evaluation/langfuse_datasets.json or
                      defaults to ["single_test", "storyland_eval"]

    Returns:
        List of evaluation result summaries
    """
    config = load_config()

    # Check if Langfuse is configured
    if not all([config.langfuse_secret_key, config.langfuse_public_key]):
        logger.error(
            "langfuse_not_configured",
            message="Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to run evaluations",
        )
        return []

    # Determine which datasets to evaluate
    if dataset_names:
        # Use explicitly provided dataset names
        datasets = dataset_names
        logger.info("using_provided_datasets", datasets=datasets)
    else:
        # Try to load from langfuse_datasets.json (created by langfuse_eval.py)
        datasets_file = Path("evaluation/langfuse_datasets.json")
        if datasets_file.exists():
            try:
                with open(datasets_file, 'r') as f:
                    datasets_info = json.load(f)
                    datasets = [
                        d["dataset_name"]
                        for d in datasets_info.get("datasets", [])
                    ]
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
                # Fall back to defaults
                datasets = ["single_test", "storyland_eval"]
        else:
            # Default datasets if no config file exists
            datasets = ["single_test", "storyland_eval"]
            logger.info("using_default_datasets", datasets=datasets)

    results = []
    for dataset_name in datasets:
        try:
            result = await run_evaluation_on_dataset(
                dataset_name=dataset_name,
                config=config,
                max_cases=max_cases_per_dataset,
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

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)

    logger.info("all_evaluations_complete", output_file=output_file)
    return results


def main():
    """Main entry point for scheduled evaluations."""
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
        help='Specific datasets to evaluate (e.g., single_test storyland_eval). '
             'If not specified, evaluates all configured datasets.',
    )

    args = parser.parse_args()

    # Run evaluations
    results = asyncio.run(
        run_all_evaluations(
            output_dir=args.output_dir,
            max_cases_per_dataset=args.max_cases,
            dataset_names=args.datasets,
        )
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    for result in results:
        print(f"\nDataset: {result.get('dataset_name', 'unknown')}")
        print(f"Timestamp: {result.get('timestamp', 'N/A')}")
        print(f"Evaluated: {result.get('evaluated_cases', 0)} cases")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
