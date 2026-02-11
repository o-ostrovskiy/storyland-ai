# Evaluation Results

This directory contains automated evaluation results from the Langfuse evaluation pipeline.

## Directory Structure

```
evaluation/
├── README.md              # This file
├── results/               # Evaluation run results (JSON files)
├── trend_report.md        # Generated trend report
└── metrics.json           # Exported metrics for dashboards
```

## Quick Start

### Run Evaluations

```bash
# Using Makefile
make eval-run

# Or directly
python tools/run_scheduled_eval.py
```

### View Results

```bash
# Show summary
make eval-summary

# Generate trend report
make eval-report

# Export metrics
make eval-export
```

## Result Files

### Evaluation Results (`results/`)

Files are named: `eval_results_YYYYMMDD_HHMMSS.json`

**Structure**:
```json
{
  "timestamp": "2026-02-02T12:00:00",
  "results": [
    {
      "dataset_name": "storyland_eval",
      "evaluated_cases": 8,
      "total_cases": 8
    }
  ]
}
```

### Trend Report (`trend_report.md`)

Markdown report showing:
- Total evaluation runs
- Recent evaluation results
- Status of each dataset
- Next steps and recommendations

### Metrics Export (`metrics.json`)

JSON format suitable for external dashboards (Grafana, Datadog, etc.)

## Viewing in Langfuse

For detailed evaluation results with scores:

1. Log in to [cloud.langfuse.com](https://cloud.langfuse.com)
2. Navigate to **Datasets**
3. Select a dataset (e.g., `storyland_eval`)
4. View evaluation runs, scores, and individual test cases

## Automated Evaluations

Evaluations run automatically via GitHub Actions:

- **Schedule**: Every Monday at 9 AM UTC
- **Workflow**: `.github/workflows/scheduled-eval.yml`
- **Results**: Uploaded as workflow artifacts

Manual trigger: Go to Actions tab → Scheduled Evaluation → Run workflow

## Documentation

Full documentation: [docs/langfuse-evaluation.md](../docs/langfuse-evaluation.md)

## Need Help?

- Check [troubleshooting.md](../docs/troubleshooting.md)
- View Langfuse docs: [langfuse.com/docs](https://langfuse.com/docs)
- Open an issue on GitHub
