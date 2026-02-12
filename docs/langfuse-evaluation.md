# Langfuse Evaluation Pipeline

Automated evaluation pipeline for StoryLand AI using Langfuse datasets, scoring functions, and scheduled evaluation runs.

## Overview

The Langfuse evaluation pipeline enables continuous quality monitoring by:
- Creating evaluation datasets from test cases
- Defining custom scoring functions for quality metrics
- Running automated evaluations on a schedule
- Tracking quality trends over time via dashboards

This complements the existing [Langfuse integration](langfuse-integration.md) which tracks token usage and costs.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Evaluation Pipeline                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Dataset Creation                                     │
│     ├─ Load .evalset.json files                         │
│     ├─ Convert to Langfuse datasets                     │
│     └─ Upload to Langfuse                               │
│                                                          │
│  2. Scoring Functions (6 metrics)                       │
│     ├─ book_relevance                                   │
│     ├─ preference_adherence                             │
│     ├─ completeness                                     │
│     ├─ actionability                                    │
│     ├─ geographical_accuracy                            │
│     └─ engagement                                       │
│                                                          │
│  3. Scheduled Evaluations                               │
│     ├─ GitHub Actions (weekly)                          │
│     ├─ Manual trigger                                   │
│     └─ Results archived                                 │
│                                                          │
│  4. Quality Dashboard                                   │
│     ├─ Trend reports                                    │
│     ├─ Metrics export                                   │
│     └─ Langfuse UI                                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites

1. **Langfuse Account**: Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
2. **API Keys**: Get credentials from project settings
3. **Environment Variables**: Configure in `.env`:

```bash
# Langfuse Configuration
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_HOST=https://cloud.langfuse.com

# Required for running evaluations
GOOGLE_API_KEY=your-google-api-key
```

### Installation

The evaluation pipeline is included in the standard installation:

```bash
pip install -e .
```

## Usage

### 1. Create Evaluation Datasets

Convert ADK evalset files to Langfuse datasets:

```bash
# Create datasets from all .evalset.json files
python evaluation/tools/langfuse_eval.py --create-datasets --evalset-dir evaluation
```

This creates Langfuse datasets from:
- `evaluation/single_test.evalset.json` → `single_test` dataset
- `evaluation/storyland_eval.evalset.json` → `storyland_eval` dataset

### 2. View Datasets in Langfuse

1. Log in to [cloud.langfuse.com](https://cloud.langfuse.com)
2. Navigate to **Datasets**
3. Select a dataset (e.g., `storyland_eval`)
4. Review test cases and their inputs

### 3. Run Scheduled Evaluations

#### Manually

```bash
# Run evaluations on all datasets
python evaluation/tools/run_scheduled_eval.py --output-dir evaluation/results --max-cases 10
```

#### Automated (GitHub Actions)

The pipeline includes a GitHub Actions workflow that runs weekly:

- **Schedule**: Every Monday at 9 AM UTC
- **Workflow**: `.github/workflows/scheduled-eval.yml`
- **Manual Trigger**: Available via GitHub Actions UI

**Setup GitHub Secrets:**

Add these secrets in your repository settings:
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_HOST`
- `GOOGLE_API_KEY`

### 4. View Results

#### Dashboard (CLI)

```bash
# Print summary of recent evaluations
python evaluation/tools/eval_dashboard.py --action summary --days 7

# Generate markdown trend report
python evaluation/tools/eval_dashboard.py --action report --days 30

# Export metrics for external tools
python evaluation/tools/eval_dashboard.py --action export --days 30
```

#### Langfuse UI

1. Go to **Datasets** in Langfuse
2. Select a dataset
3. View evaluation runs and scores
4. Inspect individual test case results

## Scoring Functions

The pipeline includes 6 quality metrics aligned with StoryLand AI's goals:

### 1. Book Relevance
**What it measures**: Connection between itinerary locations and the book's settings, themes, or author

**Scoring (1-5)**:
- **5**: All locations strongly connected to book
- **4**: Most locations clearly relevant
- **3**: Some locations relevant, some generic
- **2**: Few relevant locations
- **1**: No clear connection to book

### 2. Preference Adherence
**What it measures**: Respect for user preferences (budget, pace, accessibility, museums)

**Scoring (1-5)**:
- **5**: Perfectly aligned with all preferences
- **4**: Respects most preferences well
- **3**: Acknowledges some preferences
- **2**: Ignores most preferences
- **1**: No consideration of preferences

### 3. Completeness
**What it measures**: Coverage of cities, landmarks, and author sites

**Scoring (1-5)**:
- **5**: Comprehensive with all components
- **4**: Good coverage of main elements
- **3**: Basic itinerary, missing some details
- **2**: Minimal information
- **1**: Incomplete or vague response

### 4. Actionability
**What it measures**: Practical details for trip planning (specific places, times, logistics)

**Scoring (1-5)**:
- **5**: Highly detailed and actionable
- **4**: Good practical details
- **3**: Some actionable information
- **2**: Vague suggestions
- **1**: No practical details

### 5. Geographical Accuracy
**What it measures**: Accuracy of locations and geographical information

**Scoring (1-5)**:
- **5**: All locations accurate and real
- **4**: Minor geographical details off
- **3**: Some questionable locations
- **2**: Multiple inaccuracies
- **1**: Fictional or incorrect locations

### 6. Engagement
**What it measures**: Quality of writing and literary connection

**Scoring (1-5)**:
- **5**: Highly engaging, captures book's spirit
- **4**: Good descriptions, engaging tone
- **3**: Adequate but generic descriptions
- **2**: Dry or uninspiring
- **1**: No literary connection in writing

## Evaluation Datasets

### `single_test` Dataset
- **Purpose**: Quick smoke test
- **Cases**: 1 (Pride and Prejudice)
- **Use**: Development and CI/CD

### `storyland_eval` Dataset
- **Purpose**: Comprehensive quality assessment
- **Cases**: 8 diverse scenarios
- **Books**: Pride and Prejudice, The Great Gatsby, Harry Potter, The Da Vinci Code, Hemingway, The Girl with the Dragon Tattoo, Agatha Christie, Under the Tuscan Sun
- **Preferences**: Varied (luxury, budget, family-friendly, fast-paced, etc.)

## Dashboard & Reporting

### Trend Report

Generate a markdown report showing evaluation trends:

```bash
python evaluation/tools/eval_dashboard.py --action report --days 30
```

Output: `evaluation/trend_report.md`

**Includes**:
- Total evaluation runs
- Recent evaluation results
- Status of each dataset
- Recommendations for next steps

### Metrics Export

Export metrics for external dashboards (Grafana, Datadog, etc.):

```bash
python evaluation/tools/eval_dashboard.py --action export --days 30
```

Output: `evaluation/metrics.json`

**Format**:
```json
{
  "generated_at": "2026-02-02T12:00:00",
  "period_days": 30,
  "total_runs": 12,
  "runs": [
    {
      "timestamp": "2026-02-01T09:00:00",
      "datasets": [
        {
          "name": "storyland_eval",
          "evaluated_cases": 8,
          "total_cases": 8
        }
      ]
    }
  ]
}
```

## Best Practices

### Dataset Management

1. **Keep datasets small**: Start with 5-10 high-quality test cases
2. **Update regularly**: Add edge cases as you discover them
3. **Version control**: Store `.evalset.json` files in git

### Evaluation Frequency

- **Development**: Run manually before releases
- **Production**: Weekly automated runs
- **After changes**: Run when modifying agents or prompts

### Interpreting Results

- **Score < 3.0**: Requires immediate attention
- **Score 3.0-4.0**: Good, may need minor improvements
- **Score > 4.0**: Excellent quality

### Quality Regression

Set up alerts for:
- Average score drops below threshold
- Individual metric consistently underperforming
- Increased variance in scores

## Troubleshooting

### Authentication Failed

```
Error: Langfuse authentication failed
```

**Solution**: Verify credentials in `.env`:
```bash
# Test authentication
python -c "from langfuse import Langfuse; print(Langfuse().auth_check())"
```

### No Evaluation Results

```
No evaluation results found
```

**Solution**: Run an evaluation first:
```bash
python evaluation/tools/run_scheduled_eval.py
```

### Dataset Not Found

```
Error: Dataset 'storyland_eval' not found
```

**Solution**: Create datasets from evalsets:
```bash
python evaluation/tools/langfuse_eval.py --create-datasets
```

## Integration with Existing Tools

### ADK Evaluation

The Langfuse pipeline complements ADK's built-in evaluation:

```bash
# ADK evaluation (rubric-based, CLI)
adk eval agents/storyland single_test \
  --config_file_path tests/evaluation/eval_config.json

# Langfuse evaluation (tracking over time)
python evaluation/tools/run_scheduled_eval.py
```

### Token Tracking

Langfuse tracks both quality metrics AND token usage:
- **Quality**: Via evaluation pipeline (this doc)
- **Cost**: Via plugin (see [langfuse-integration.md](langfuse-integration.md))

## Next Steps

1. **Set up scheduled runs**: Configure GitHub Actions with secrets
2. **Create custom scorers**: Add domain-specific metrics in Langfuse UI
3. **Monitor trends**: Review weekly trend reports
4. **Expand datasets**: Add more diverse test scenarios
5. **Set quality gates**: Fail CI if scores drop below threshold

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse Datasets Guide](https://langfuse.com/docs/datasets)
- [Google ADK Evaluation](https://github.com/googleapis/python-genai)
- [Testing Guide](testing.md)

---

**Need help?** Check [troubleshooting.md](troubleshooting.md) or open an issue on GitHub.
