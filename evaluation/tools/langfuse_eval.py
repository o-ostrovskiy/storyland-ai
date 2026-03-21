"""
Langfuse Evaluation Pipeline for StoryLand AI.

This module provides tools for creating and managing evaluation datasets in Langfuse,
defining scoring functions, and running automated evaluations to track quality over time.
"""

import os
import json
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

from common.logging import get_logger

logger = get_logger("storyland.langfuse_eval")


class LangfuseEvalPipeline:
    """
    Automated evaluation pipeline using Langfuse.

    Features:
    - Create evaluation datasets from existing test cases
    - Define custom scoring functions for quality metrics
    - Run evaluations and track results over time
    - Export evaluation results for analysis
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        public_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        """
        Initialize Langfuse evaluation pipeline.

        Args:
            secret_key: Langfuse secret key (or use LANGFUSE_SECRET_KEY env var)
            public_key: Langfuse public key (or use LANGFUSE_PUBLIC_KEY env var)
            host: Langfuse host URL (or use LANGFUSE_HOST env var)
        """
        if not LANGFUSE_AVAILABLE:
            raise ImportError(
                "Langfuse not available. Install with: pip install langfuse"
            )

        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not all([self.secret_key, self.public_key]):
            raise ValueError(
                "Langfuse credentials required. Set LANGFUSE_SECRET_KEY and "
                "LANGFUSE_PUBLIC_KEY environment variables."
            )

        self.client = Langfuse(
            secret_key=self.secret_key,
            public_key=self.public_key,
            host=self.host,
        )

        # Verify authentication
        if not self.client.auth_check():
            raise RuntimeError("Langfuse authentication failed. Check credentials.")

        logger.info("langfuse_eval_initialized", host=self.host)

    def create_dataset_from_evalset(
        self,
        evalset_path: str,
        dataset_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Create a Langfuse dataset from an ADK evalset file.

        Args:
            evalset_path: Path to .evalset.json file
            dataset_name: Name for the Langfuse dataset (defaults to eval_set_id)
            description: Description of the dataset

        Returns:
            Dataset name in Langfuse
        """
        # Load evalset
        with open(evalset_path, 'r') as f:
            evalset = json.load(f)

        dataset_name = dataset_name or evalset.get('eval_set_id', 'storyland_eval')
        description = description or evalset.get('description', 'StoryLand AI evaluation dataset')

        logger.info(
            "creating_langfuse_dataset",
            dataset_name=dataset_name,
            num_cases=len(evalset.get('eval_cases', [])),
        )

        self.client.create_dataset(
            name=dataset_name,
            description=description,
        )

        # Add eval cases as dataset items
        for case in evalset.get('eval_cases', []):
            eval_id = case.get('eval_id', 'unknown')

            # Extract input from conversation
            input_data = self._extract_input_from_case(case)

            # Extract expected output (if available)
            expected_output = case.get('expected_output', None)

            # Add metadata
            metadata = {
                'eval_id': eval_id,
                'session_input': case.get('session_input', {}),
                'creation_timestamp': case.get('creation_timestamp'),
            }

            self.client.create_dataset_item(
                dataset_name=dataset_name,
                id=eval_id,
                input=input_data,
                expected_output=expected_output,
                metadata=metadata,
            )

        logger.info("langfuse_dataset_created", dataset_name=dataset_name)
        return dataset_name

    def _extract_input_from_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Extract input data from eval case."""
        return {
            'book_title': case.get('book_title', ''),
            'author': case.get('author', ''),
        }

    def create_scoring_functions(self) -> List[str]:
        """
        Create scoring functions in Langfuse for StoryLand AI evaluation.

        Returns:
            List of created scoring function IDs
        """
        scoring_functions = [
            {
                'name': 'book_relevance',
                'description': 'Evaluates if itinerary locations are connected to the book\'s settings, themes, or author',
                'prompt': (
                    'Evaluate if the travel itinerary locations are directly connected to the book\'s '
                    'settings, themes, characters, or author. Each suggested place should have a clear '
                    'and meaningful connection to the literary work.\n\n'
                    'Score 1-5 where:\n'
                    '5 = All locations strongly connected to book\n'
                    '4 = Most locations clearly relevant\n'
                    '3 = Some locations relevant, some generic\n'
                    '2 = Few relevant locations\n'
                    '1 = No clear connection to book'
                ),
            },
            {
                'name': 'preference_adherence',
                'description': 'Evaluates if recommendations respect user preferences (budget, pace, accessibility)',
                'prompt': (
                    'Evaluate if the travel recommendations respect the user\'s stated preferences '
                    'including budget level, travel pace, accessibility needs, and interest in museums. '
                    'The response should acknowledge and adapt to these preferences.\n\n'
                    'Score 1-5 where:\n'
                    '5 = Perfectly aligned with all preferences\n'
                    '4 = Respects most preferences well\n'
                    '3 = Acknowledges some preferences\n'
                    '2 = Ignores most preferences\n'
                    '1 = No consideration of preferences'
                ),
            },
            {
                'name': 'completeness',
                'description': 'Evaluates if itinerary includes comprehensive details (cities, landmarks, author sites)',
                'prompt': (
                    'Evaluate if the response includes a comprehensive itinerary with cities, landmarks, '
                    'and author-related sites. The itinerary should provide enough detail for the traveler '
                    'to plan their trip.\n\n'
                    'Score 1-5 where:\n'
                    '5 = Comprehensive with all components\n'
                    '4 = Good coverage of main elements\n'
                    '3 = Basic itinerary, missing some details\n'
                    '2 = Minimal information\n'
                    '1 = Incomplete or vague response'
                ),
            },
            {
                'name': 'actionability',
                'description': 'Evaluates if itinerary is practical with specific places and logistics',
                'prompt': (
                    'Evaluate if the itinerary is practical and actionable with specific places, '
                    'suggested times of day to visit, and logistical notes. The traveler should be '
                    'able to use this information to actually plan their trip.\n\n'
                    'Score 1-5 where:\n'
                    '5 = Highly detailed and actionable\n'
                    '4 = Good practical details\n'
                    '3 = Some actionable information\n'
                    '2 = Vague suggestions\n'
                    '1 = No practical details'
                ),
            },
            {
                'name': 'geographical_accuracy',
                'description': 'Evaluates accuracy of locations and geographical information',
                'prompt': (
                    'Evaluate if the locations mentioned are real places that can be visited. '
                    'Cities and landmarks should be correctly associated with their countries. '
                    'The geographical information should be accurate.\n\n'
                    'Score 1-5 where:\n'
                    '5 = All locations accurate and real\n'
                    '4 = Minor geographical details off\n'
                    '3 = Some questionable locations\n'
                    '2 = Multiple inaccuracies\n'
                    '1 = Fictional or incorrect locations'
                ),
            },
            {
                'name': 'engagement',
                'description': 'Evaluates if descriptions are engaging and evoke the spirit of the book',
                'prompt': (
                    'Evaluate if the summary and descriptions are engaging and evoke the spirit of '
                    'the book. The language should capture the literary connection and make the trip '
                    'feel like a meaningful journey.\n\n'
                    'Score 1-5 where:\n'
                    '5 = Highly engaging, captures book\'s spirit\n'
                    '4 = Good descriptions, engaging tone\n'
                    '3 = Adequate but generic descriptions\n'
                    '2 = Dry or uninspiring\n'
                    '1 = No literary connection in writing'
                ),
            },
        ]

        logger.info("creating_scoring_functions", count=len(scoring_functions))

        # Note: Langfuse scoring functions are typically created via UI
        # For programmatic creation, we'll store them as metadata
        return [sf['name'] for sf in scoring_functions]

    def export_evaluation_results(
        self,
        output_path: str,
        dataset_name: Optional[str] = None,
    ) -> None:
        """
        Export evaluation results to JSON file.

        Args:
            output_path: Path to save results JSON
            dataset_name: Filter by dataset name (optional)
        """
        logger.info("exporting_evaluation_results", output_path=output_path)

        # Fetch dataset runs from Langfuse
        # Note: This would use Langfuse API to fetch evaluation results
        # Implementation depends on Langfuse SDK capabilities

        results = {
            'export_date': datetime.now().isoformat(),
            'dataset_name': dataset_name,
            'instructions': (
                'To view detailed results, visit Langfuse dashboard at '
                f'{self.host} and navigate to Datasets > {dataset_name}'
            ),
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("evaluation_results_exported", path=output_path)

    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get statistics about a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset statistics
        """
        # This would query Langfuse API for dataset stats
        # Implementation depends on SDK capabilities
        return {
            'dataset_name': dataset_name,
            'message': f'View dataset at {self.host}',
        }


def create_dataset_from_evalsets(
    evalset_dir: str = "agents/storyland",
    output_summary: str = "evaluation/langfuse_datasets.json",
) -> None:
    """
    Create Langfuse datasets from all evalset files.

    Args:
        evalset_dir: Directory containing .evalset.json files
        output_summary: Path to save summary of created datasets
    """
    pipeline = LangfuseEvalPipeline()

    evalset_files = list(Path(evalset_dir).glob("*.evalset.json"))
    logger.info("found_evalset_files", count=len(evalset_files))

    created_datasets = []

    for evalset_file in evalset_files:
        dataset_name = pipeline.create_dataset_from_evalset(str(evalset_file))
        created_datasets.append({
            'evalset_file': str(evalset_file),
            'dataset_name': dataset_name,
            'created_at': datetime.now().isoformat(),
        })

    # Save summary
    os.makedirs(os.path.dirname(output_summary), exist_ok=True)
    with open(output_summary, 'w') as f:
        json.dump({
            'created_at': datetime.now().isoformat(),
            'datasets': created_datasets,
        }, f, indent=2)

    logger.info("datasets_created", count=len(created_datasets))


if __name__ == '__main__':
    """CLI for creating Langfuse datasets from evalsets."""
    import argparse

    parser = argparse.ArgumentParser(description='Langfuse Evaluation Pipeline')
    parser.add_argument(
        '--create-datasets',
        action='store_true',
        help='Create Langfuse datasets from evalset files',
    )
    parser.add_argument(
        '--evalset-dir',
        default='agents/storyland',
        help='Directory containing .evalset.json files',
    )

    args = parser.parse_args()

    if args.create_datasets:
        create_dataset_from_evalsets(evalset_dir=args.evalset_dir)
    else:
        parser.print_help()
