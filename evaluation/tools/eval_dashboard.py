"""
Evaluation Dashboard for StoryLand AI.

Creates visualizations and reports for tracking evaluation metrics over time.
Can be run locally or integrated into monitoring systems.
"""

import os
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logging import get_logger

logger = get_logger("storyland.eval_dashboard")


def make_langfuse_annotation_checker():
    """
    Build a checker that reports whether a trace has human annotation scores.

    Returns a callable trace_id -> Optional[bool]: True if any
    ANNOTATION-source score exists on the trace, False if none, None if the
    status can't be determined (no credentials, API error). Returns None
    directly (no callable) when Langfuse credentials are not configured —
    the weekly CI job has them, a bare local checkout may not.
    """
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    if not (secret_key and public_key):
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        return None

    langfuse = Langfuse(
        secret_key=secret_key,
        public_key=public_key,
        host=os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com",
    )

    def checker(trace_id: str) -> Optional[bool]:
        try:
            response = langfuse.api.scores_v3.get_many_v3(
                trace_id=trace_id, source="ANNOTATION", limit=1
            )
            return bool(response.data)
        except Exception as e:
            logger.warning(
                "annotation_check_failed", trace_id=trace_id, error=str(e)
            )
            return None

    return checker


def build_spot_check_section(
    results: List[Dict[str, Any]],
    annotation_checker=None,
    now: Optional[datetime] = None,
) -> List[str]:
    """
    Render the "Human spot-checks" report section from loaded results files.

    Every flagged case in the window gets a row; a case stays listed as
    PENDING (with its age) until an ANNOTATION-source score exists on its
    trace, so skipped reviews accumulate visibly instead of disappearing.

    Args:
        results: Loaded results-file dicts (each may carry human_spot_check)
        annotation_checker: Optional trace_id -> Optional[bool] (True =
            reviewed). None means review status can't be verified.
        now: Clock override for age computation (tests)

    Returns:
        Markdown lines (empty list if no run in the window flagged cases)
    """
    now = now or datetime.now()
    rows = []
    pending_count = 0

    for result in results:
        spot_check = result.get("human_spot_check")
        if not spot_check or not spot_check.get("selected"):
            continue

        selected_at = spot_check.get("selected_at", "")
        age_days = None
        try:
            age_days = (now - datetime.fromisoformat(selected_at)).days
        except (ValueError, TypeError):
            pass

        for case in spot_check["selected"]:
            trace_id = case.get("trace_id")
            reviewed = (
                annotation_checker(trace_id)
                if annotation_checker and trace_id
                else None
            )
            if reviewed is True:
                status = "✅ Reviewed"
            elif reviewed is False:
                pending_count += 1
                age = f", {age_days}d old" if age_days is not None else ""
                status = f"⏳ PENDING{age}"
            else:
                status = "❓ Unknown (no Langfuse credentials)"

            date = selected_at.split("T")[0] if "T" in selected_at else selected_at
            rows.append(
                f"| {date} | {case.get('dataset', '?')} | "
                f"{case.get('item_id', '?')} | {case.get('book_title', '?')} | "
                f"{status} |"
            )

    if not rows:
        return []

    lines = [
        "## Human spot-checks",
        "",
        "Randomly flagged cases awaiting hand review (judge-calibration "
        "spot-checks). Pending rows persist until the trace carries a "
        "human annotation score in Langfuse.",
        "",
        "| Flagged | Dataset | Case | Book | Review status |",
        "|---------|---------|------|------|---------------|",
        *rows,
        "",
    ]
    if pending_count:
        lines.extend([
            f"**⚠️ {pending_count} spot-check(s) awaiting human review.**",
            "",
        ])
    return lines


class EvalDashboard:
    """
    Dashboard for tracking evaluation quality metrics over time.

    Features:
    - Load evaluation results from JSON files
    - Generate trend reports
    - Export metrics for external dashboards (Grafana, etc.)
    """

    def __init__(self, results_dir: str = "evaluation/results"):
        """
        Initialize evaluation dashboard.

        Args:
            results_dir: Directory containing evaluation result JSON files
        """
        self.results_dir = results_dir
        logger.info("eval_dashboard_initialized", results_dir=results_dir)

    def load_all_results(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Load all evaluation results from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of evaluation results sorted by timestamp
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        results = []
        pattern = os.path.join(self.results_dir, "eval_results_*.json")

        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                timestamp_str = data.get('timestamp', '')
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp >= cutoff_date:
                        results.append(data)
            except Exception as e:
                logger.warning("failed_to_load_result", filepath=filepath, error=str(e))

        # Sort by timestamp
        results.sort(key=lambda x: x.get('timestamp', ''))

        logger.info("loaded_results", count=len(results), days=days)
        return results

    def generate_trend_report(
        self,
        output_path: str = "evaluation/trend_report.md",
        days: int = 30,
    ) -> None:
        """
        Generate a markdown report showing evaluation trends.

        Args:
            output_path: Path to save the report
            days: Number of days to include in the report
        """
        results = self.load_all_results(days=days)

        report_lines = [
            "# StoryLand AI - Evaluation Trends",
            "",
            f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Period:** Last {days} days",
            f"**Total Evaluation Runs:** {len(results)}",
            "",
            "## Overview",
            "",
        ]

        if not results:
            report_lines.extend([
                "No evaluation results found for the specified period.",
                "",
                "To run evaluations:",
                "```bash",
                "python evaluation/tools/run_scheduled_eval.py",
                "```",
            ])
        else:
            # Calculate summary stats
            total_cases = sum(
                sum(r.get('evaluated_cases', 0) for r in result.get('results', []))
                for result in results
            )

            report_lines.extend([
                f"- **Total Test Cases Evaluated:** {total_cases}",
                f"- **Latest Evaluation:** {results[-1].get('timestamp', 'N/A')}",
                "",
                "## Recent Evaluations",
                "",
                "| Date | Dataset | Cases | Status |",
                "|------|---------|-------|--------|",
            ])

            # Show last 10 evaluation runs
            for result in results[-10:]:
                timestamp = result.get('timestamp', 'N/A')
                date = timestamp.split('T')[0] if 'T' in timestamp else timestamp

                for dataset_result in result.get('results', []):
                    dataset_name = dataset_result.get('dataset_name', 'unknown')
                    cases = dataset_result.get('evaluated_cases', 0)
                    status = "✅ Complete" if cases > 0 else "⚠️ No data"

                    report_lines.append(f"| {date} | {dataset_name} | {cases} | {status} |")

            spot_check_lines = build_spot_check_section(
                results, annotation_checker=make_langfuse_annotation_checker()
            )
            if spot_check_lines:
                report_lines.append("")
                report_lines.extend(spot_check_lines)

            report_lines.extend([
                "",
                "## Viewing Results",
                "",
                "To view detailed evaluation results:",
                "",
                "1. **Langfuse Dashboard**: Visit your Langfuse instance and navigate to Datasets",
                "2. **Local Results**: Check `evaluation/results/` directory for JSON files",
                "",
                "## Next Steps",
                "",
                "- Review failed test cases in Langfuse",
                "- Update scoring thresholds if needed",
                "- Add new test scenarios to evalsets",
                "- Monitor trends for quality regression",
            ])

        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info("trend_report_generated", output_path=output_path)
        print(f"\n✅ Trend report saved to: {output_path}")

    def export_metrics(
        self,
        output_path: str = "evaluation/metrics.json",
        days: int = 30,
    ) -> None:
        """
        Export evaluation metrics in a format suitable for external dashboards.

        Args:
            output_path: Path to save metrics JSON
            days: Number of days to include
        """
        results = self.load_all_results(days=days)

        metrics = {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "total_runs": len(results),
            "runs": [],
        }

        for result in results:
            run_data = {
                "timestamp": result.get('timestamp'),
                "datasets": [],
            }

            for dataset_result in result.get('results', []):
                run_data["datasets"].append({
                    "name": dataset_result.get('dataset_name'),
                    "evaluated_cases": dataset_result.get('evaluated_cases', 0),
                    "total_cases": dataset_result.get('total_cases', 0),
                })

            metrics["runs"].append(run_data)

        # Save metrics
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info("metrics_exported", output_path=output_path)
        print(f"\n✅ Metrics exported to: {output_path}")

    def print_summary(self, days: int = 7) -> None:
        """
        Print a summary of recent evaluations to console.

        Args:
            days: Number of days to include
        """
        results = self.load_all_results(days=days)

        print("\n" + "=" * 60)
        print("StoryLand AI - Evaluation Summary")
        print("=" * 60)
        print(f"Period: Last {days} days")
        print(f"Total evaluation runs: {len(results)}")

        if results:
            latest = results[-1]
            print(f"Latest run: {latest.get('timestamp', 'N/A')}")

            print("\nRecent results:")
            for result in results[-5:]:
                timestamp = result.get('timestamp', 'N/A')
                date = timestamp.split('T')[0] if 'T' in timestamp else timestamp
                print(f"  - {date}: {len(result.get('results', []))} dataset(s)")
        else:
            print("\nNo evaluation results found.")
            print("\nRun evaluations with:")
            print("  python evaluation/tools/run_scheduled_eval.py")

        print("=" * 60 + "\n")


def main():
    """Main entry point for dashboard CLI."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluation Dashboard')
    parser.add_argument(
        '--action',
        choices=['summary', 'report', 'export'],
        default='summary',
        help='Action to perform',
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to include',
    )
    parser.add_argument(
        '--results-dir',
        default='evaluation/results',
        help='Directory containing evaluation results',
    )

    args = parser.parse_args()

    dashboard = EvalDashboard(results_dir=args.results_dir)

    if args.action == 'summary':
        dashboard.print_summary(days=args.days)
    elif args.action == 'report':
        dashboard.generate_trend_report(days=args.days)
    elif args.action == 'export':
        dashboard.export_metrics(days=args.days)


if __name__ == '__main__':
    main()
