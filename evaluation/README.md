# Evaluation Pipeline

Automated evaluation system for StoryLand AI using Langfuse to track quality over time.

## Overview

The evaluation pipeline runs the complete StoryLand workflow (metadata → discovery → composition) on test datasets and tracks results in Langfuse. This enables:
- Continuous quality monitoring
- Regression detection before releases
- Cost/token tracking per evaluation
- Trend analysis over time

## Directory Structure

```
evaluation/
├── README.md                      # This file
├── tools/                         # Evaluation scripts
│   ├── run_scheduled_eval.py      # Main evaluation runner
│   ├── eval_dashboard.py          # Dashboard and reporting
│   ├── langfuse_eval.py           # Dataset creation pipeline
│   └── setup_langfuse_eval.sh     # Setup script
├── single_test.evalset.json       # Test dataset (1 case: Pride & Prejudice)
├── storyland_eval.evalset.json    # Main dataset (8 diverse books)
├── langfuse_datasets.json         # Dataset registry (gitignored)
├── results/                       # Evaluation run results (gitignored)
├── trend_report.md                # Generated trend report (tracked)
└── metrics.json                   # Exported metrics (gitignored)
```

## Setup

### Prerequisites

1. **Langfuse account**: Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
2. **API credentials**: Add to `.env`:

```bash
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_HOST=https://cloud.langfuse.com
GOOGLE_API_KEY=your-google-api-key
```

### Initialize Datasets

Create Langfuse datasets from evalset files (one-time):

```bash
make eval-setup
```

This creates two datasets in Langfuse:
- `single_test` - 1 test case (Pride & Prejudice) for quick validation
- `storyland_eval` - 8 diverse scenarios for comprehensive testing

## Quick Start

### Run Evaluations

```bash
# Using Makefile
make eval-run

# Or directly
python evaluation/tools/run_scheduled_eval.py
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

## Quality Metrics

Each evaluation is scored on 6 dimensions (1-5 scale):

1. **Book Relevance** - Are locations connected to the book's settings/themes/author?
2. **Preference Adherence** - Does itinerary respect user preferences (budget, pace, etc.)?
3. **Completeness** - Includes cities, landmarks, and author sites?
4. **Actionability** - Specific places with practical trip-planning details?
5. **Geographical Accuracy** - Real locations correctly associated with countries?
6. **Engagement** - Descriptions capture the book's spirit?

**Note**: LLM-as-judge scoring is not yet implemented (see Issue #96). Currently evaluations just run the workflow and track success/failure.

## Troubleshooting

### "Langfuse authentication failed"
Check credentials in `.env`:
```bash
python -c "from langfuse import Langfuse; print(Langfuse().auth_check())"
```

### "Dataset not found"
Run setup to create datasets:
```bash
make eval-setup
```

### No evaluation results
Run an evaluation first:
```bash
make eval-run
```

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse Datasets Guide](https://langfuse.com/docs/datasets)
- Token tracking: [docs/langfuse-integration.md](../docs/langfuse-integration.md)
