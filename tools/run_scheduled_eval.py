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
    evaluated_cases = 0  # Real workflow evaluations only
    placeholder_cases = 0  # Placeholder executions (not real evals)
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

                # Score the trace based on evaluation result
                # Only real workflow evaluations are scored
                case_status = result.get("status", "unknown")

                if case_status == "evaluated":
                    # Real evaluation - score it (LLM-as-judge in issue #96)
                    root_span.score_trace(
                        name="evaluation_status",
                        value=1.0,
                        comment="Evaluation completed - implement LLM scoring (see issue #96)",
                    )
                elif case_status == "placeholder":
                    # Placeholder - don't score to avoid polluting metrics
                    logger.info(
                        "placeholder_execution",
                        item_id=item.id,
                        note="Workflow execution not implemented (see issue #95)",
                    )

            # Track result status
            case_results.append({
                "item_id": item.id,
                "status": case_status,
                "run_name": run_name,
            })

            # Count by status type
            if case_status == "evaluated":
                # Real workflow evaluation
                evaluated_cases += 1
                logger.info(
                    "case_evaluated",
                    dataset_name=dataset_name,
                    item_id=item.id,
                    run_name=run_name,
                )
            elif case_status == "placeholder":
                # Placeholder execution (not a real eval)
                placeholder_cases += 1
                logger.info(
                    "placeholder_counted",
                    dataset_name=dataset_name,
                    item_id=item.id,
                )
            elif case_status == "skipped":
                logger.info(
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

    # Flush Langfuse client to ensure all traces are sent
    langfuse.flush()

    results = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "total_cases": total_cases,
        "evaluated_cases": evaluated_cases,  # Real evaluations only
        "placeholder_cases": placeholder_cases,  # Placeholders (not real)
        "failed_cases": failed_cases,
        "case_results": case_results,
    }

    logger.info(
        "evaluation_complete",
        dataset=dataset_name,
        evaluated=evaluated_cases,
        placeholders=placeholder_cases,
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
        logger.info("placeholder_execution", input_text=text)

        # Placeholder result - in a real implementation, this would run the agent
        # Status is "placeholder" not "evaluated" to avoid counting as real evaluation
        result = {
            "status": "placeholder",
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
        # Return error result instead of empty list to distinguish from legitimate empty results
        return [{
            "dataset_name": "config_error",
            "error": "Langfuse credentials not configured. Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY.",
            "timestamp": datetime.now().isoformat(),
            "total_cases": 0,
            "evaluated_cases": 0,
        }]

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

        # Show breakdown of evaluation types
        if evaluated > 0:
            print(f"Real evaluations: {evaluated} cases")
        if placeholders > 0:
            print(f"Placeholder runs: {placeholders} cases (workflow not implemented)")
        if evaluated == 0 and placeholders == 0 and total_cases > 0:
            print(f"Evaluated: 0 cases")

        total_evaluated += evaluated
        total_placeholders += placeholders

        # Check if this dataset failed
        # Failure = explicit error OR (has cases but all failed)
        # Empty dataset (total_cases == 0) is NOT a failure
        has_error = 'error' in result
        has_failures = total_cases > 0 and failed_cases == total_cases

        if has_error or has_failures:
            total_failed_datasets += 1
            if has_error:
                print(f"ERROR: {result['error']}")
            elif has_failures:
                print(f"ERROR: All {failed_cases} case(s) failed evaluation")
        elif total_cases == 0:
            # Empty dataset - informational, not an error
            total_skipped += 1
            print("INFO: Dataset is empty (no test cases)")

    print("\n" + "=" * 60)
    print(f"Total: {total_evaluated} real evaluation(s), {total_placeholders} placeholder(s)")
    if total_skipped > 0:
        print(f"Empty datasets: {total_skipped}")
    print(f"Failed datasets: {total_failed_datasets}")
    print("=" * 60)

    # Exit with error code if all evaluations failed
    # This ensures GitHub Actions workflow correctly reports failure
    # Note: Empty datasets don't count as failures
    actual_datasets = total_datasets - total_skipped

    if actual_datasets == 0 and total_failed_datasets > 0:
        # No actual datasets to evaluate, but config/setup errors exist
        print("\n❌ ERROR: Configuration or setup error - cannot run evaluations")
        sys.exit(1)
    elif actual_datasets > 0 and total_failed_datasets == actual_datasets:
        # All non-empty datasets failed
        print("\n❌ ERROR: All dataset evaluations failed")
        sys.exit(1)
    elif total_failed_datasets > 0:
        print(f"\n⚠️  WARNING: {total_failed_datasets}/{total_datasets} dataset(s) failed")
        # Still exit 0 if some succeeded - partial success
    elif actual_datasets == 0:
        # Only empty datasets
        print("\n⚠️  WARNING: No datasets to evaluate (all empty)")
        # Exit 0 - not an error, just nothing to do
    else:
        # Success - but distinguish real evals from placeholders
        if total_evaluated > 0:
            print("\n✅ All evaluations completed successfully")
        elif total_placeholders > 0:
            print("\n⚠️  Placeholder runs completed (no real workflow evaluation)")
            print("   Implement issues #95, #96, #97 for actual quality evaluation")
        else:
            print("\n✅ Execution completed (no cases to evaluate)")


if __name__ == '__main__':
    main()
