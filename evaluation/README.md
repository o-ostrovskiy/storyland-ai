# Evaluation & Observability

Automated evaluation system and LLM observability for StoryLand AI, powered by [Langfuse](https://langfuse.com).

## Overview

This module provides two capabilities:

1. **Quality Evaluation** — Run the complete StoryLand workflow on test datasets and track results over time for regression detection
2. **Token & Cost Tracking** — Monitor token usage, cost, and trace hierarchy for every Gemini API call via the Langfuse plugin

Both require a Langfuse account. Token tracking is optional and auto-disables if credentials are missing.

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

**To get credentials:**
1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a new project
3. Copy the API keys from project settings

### Initialize Datasets

Create Langfuse datasets from evalset files (one-time):

```bash
make eval-setup
```

This creates two datasets in Langfuse:
- `single_test` — 1 test case (Pride & Prejudice) for quick validation
- `storyland_eval` — 8 diverse scenarios for comprehensive testing

## Quality Evaluation

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

### Quality Metrics

Each evaluation is scored on 6 dimensions (1-5 scale):

1. **Book Relevance** — Are locations connected to the book's settings/themes/author?
2. **Preference Adherence** — Does itinerary respect user preferences (budget, pace, etc.)?
3. **Completeness** — Includes cities, landmarks, and author sites?
4. **Actionability** — Specific places with practical trip-planning details?
5. **Geographical Accuracy** — Real locations correctly associated with countries?
6. **Engagement** — Descriptions capture the book's spirit?

LLM-as-judge scoring is implemented in `evaluation/tools/llm_scorer.py` using Gemini to evaluate itineraries against these criteria.

### Automated Evaluations

Evaluations run automatically via GitHub Actions:

- **Schedule**: Every Monday at 9 AM UTC
- **Workflow**: `.github/workflows/scheduled-eval.yml`
- **Results**: Uploaded as workflow artifacts

Manual trigger: Go to Actions tab → Scheduled Evaluation → Run workflow

### Prompt Versioning

Use `--prompt-version <label>` to tag a run in Langfuse before merging a prompt change. Compare against previous runs by filtering on `prompt_version` in the Langfuse Runs view. Prompt change history is in [`evaluation/PROMPT_CHANGELOG.md`](PROMPT_CHANGELOG.md).

## Token & Cost Tracking (Langfuse Plugin)

### Features

- **Input/Output Tokens**: Accurate counts from Gemini API responses
- **Per-Agent Metrics**: Track token usage for each agent call
- **Real-time Monitoring**: View token consumption as workflows execute
- **Cost Calculation**: Automatic cost estimation based on Gemini pricing
- **Trace Hierarchy**: Top-level traces for workflows, spans for nested agents
- **Tool Tracking**: Monitor tool execution within agent calls

### Current Gemini Pricing

Gemini 2.0 Flash (as of Jan 2026):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

### Enabling the Plugin

The plugin is automatically initialized if credentials are provided:

```python
from plugins.langfuse_plugin import LangfusePlugin
from common.config import load_config

config = load_config()

langfuse_plugin = LangfusePlugin(
    secret_key=config.langfuse_secret_key,
    public_key=config.langfuse_public_key,
    host=config.langfuse_host,
)

runner = Runner(
    agent=my_agent,
    plugins=[LoggingPlugin(), langfuse_plugin],
)
```

Auto-disables if credentials are missing.

### CLI Output

Token stats are displayed at the end of each run:

```
📊 Context: 45 events, ~12000 tokens
💰 Token Usage: 8,432 tokens (input: 5,123, output: 3,309)
   Estimated cost: $0.001378
```

### Programmatic Access

```python
stats = langfuse_plugin.get_session_stats()
print(f"Total tokens: {stats['total_tokens']}")
print(f"Cost: ${stats['cost_usd']:.6f}")

langfuse_plugin.reset_stats()
await langfuse_plugin.flush()
```

### Architecture

```
ADK Runner
    ↓
Plugin.on_event()
    ↓
AgentStartEvent → Create Langfuse trace/span
ModelRequestEvent → Start generation tracking
ModelResponseEvent → Extract token usage, calculate cost
AgentCompleteEvent → Finalize trace with totals
```

Token extraction from Gemini responses:
```python
response.usage_metadata.prompt_token_count      # Input tokens
response.usage_metadata.candidates_token_count  # Output tokens
response.usage_metadata.total_token_count       # Total tokens
```

### Disabling the Plugin

**Option 1:** Omit from Runner plugins:
```python
runner = Runner(agent=my_agent, plugins=[LoggingPlugin()])
```

**Option 2:** Remove credentials from `.env`.

## Viewing Data in Langfuse

1. Log in to [cloud.langfuse.com](https://cloud.langfuse.com)
2. Select your project
3. **Traces** — One per workflow execution (metadata_extraction, discovery, composition)
4. **Generations** — Individual LLM calls with token counts and costs
5. **Spans** — Tool executions and nested agent calls
6. **Datasets** — Evaluation runs, scores, and individual test cases

## Directory Structure

```
evaluation/
├── README.md                      # This file
├── tools/                         # Evaluation scripts
│   ├── run_scheduled_eval.py      # Main evaluation runner
│   ├── eval_dashboard.py          # Dashboard and reporting
│   ├── langfuse_eval.py           # Dataset creation pipeline
│   ├── llm_scorer.py              # LLM-as-judge quality scoring
│   └── setup_langfuse_eval.sh     # Setup script
├── single_test.evalset.json       # Test dataset (1 case: Pride & Prejudice)
├── storyland_eval.evalset.json    # Main dataset (8 diverse books)
├── langfuse_datasets.json         # Dataset registry (gitignored)
├── results/                       # Evaluation run results (gitignored)
├── trend_report.md                # Generated trend report (tracked)
└── metrics.json                   # Exported metrics (gitignored)
```

### Result Files

- **`results/eval_results_YYYYMMDD_HHMMSS.json`** — Per-run evaluation results
- **`trend_report.md`** — Markdown report with recent evaluation status
- **`metrics.json`** — JSON export for external dashboards (Grafana, Datadog, etc.)

## Troubleshooting

### Langfuse authentication failed
```bash
python -c "from langfuse import Langfuse; print(Langfuse().auth_check())"
```

### Dataset not found
```bash
make eval-setup
```

### Plugin not tracking tokens
- Check credentials are set in `.env`
- Look for `langfuse_enabled` or `langfuse_disabled` log messages
- Verify installation: `pip list | grep langfuse`

### Zero cost displayed
Token usage is tracked, but cost may show $0.000000 for very small usage. Verify pricing constants in `plugins/langfuse_plugin.py`.

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse Datasets Guide](https://langfuse.com/docs/datasets)
- [Gemini API Pricing](https://ai.google.dev/pricing)
- [Google ADK Plugins Guide](https://github.com/google/adk-plugins)
