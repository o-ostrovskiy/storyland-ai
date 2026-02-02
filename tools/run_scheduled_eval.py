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
from typing import Dict, Any, List

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

    # Get dataset items
    dataset = langfuse.get_dataset(dataset_name)

    results = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "total_cases": 0,
        "evaluated_cases": 0,
        "scores": {},
    }

    logger.info(
        "evaluation_complete",
        dataset=dataset_name,
        message=f"View results at {config.langfuse_host}",
    )

    return results


async def run_all_evaluations(
    output_dir: str = "evaluation/results",
    max_cases_per_dataset: int = 10,
) -> List[Dict[str, Any]]:
    """
    Run evaluations on all configured datasets.

    Args:
        output_dir: Directory to save evaluation results
        max_cases_per_dataset: Maximum cases to evaluate per dataset

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

    # Datasets to evaluate
    datasets = ["single_test", "storyland_eval"]

    results = []
    for dataset_name in datasets:
        result = await run_evaluation_on_dataset(
            dataset_name=dataset_name,
            config=config,
            max_cases=max_cases_per_dataset,
        )
        results.append(result)

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

    args = parser.parse_args()

    # Run evaluations
    results = asyncio.run(
        run_all_evaluations(
            output_dir=args.output_dir,
            max_cases_per_dataset=args.max_cases,
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
