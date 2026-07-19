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
- `storyland_eval` — 8 diverse scenarios for comprehensive testing
- `books_v1` — 10 cases with expected outputs and book-specific scoring criteria
- `place_to_book_v1` — 11 cases for the **place→book reverse-routing** grounding gate (literal/vibe labelling + not-found state). Register with `make eval-setup-one EVALSET_FILE=evaluation/place_to_book_v1.evalset.json`.

## Quality Evaluation

### Run Evaluations

```bash
# Itinerary (book→place) — LLM-as-judge scoring on the storyland_eval / books_v1 datasets
make eval-run
# Or directly:
python evaluation/tools/run_scheduled_eval.py
```

### Place→Book grounding eval (reverse routing)

The `place_to_book_v1` dataset is **not** scored by the itinerary judge — it is a
deterministic grounding gate for the reverse-routing capability (`PlaceToBookResolver`):
a real destination must return at least `min_literal` grounded `literal` candidates
(each naming a real `maps_to`), and a fictional/ungroundable place must return the clean
not-found state with no fabricated list. It has its own runner:

```bash
make eval-place-to-book
# Or directly (re-registers the dataset items from the evalset, then runs all cases):
python evaluation/tools/run_place_to_book_eval.py [--max-cases N] [--no-register]
```

Each case logs a Langfuse dataset run (`p2b_eval_YYYYMMDD_HHMMSS`) under `place_to_book_v1`
with scores: `case_pass` (0/1), `found_classification` (0/1), `literal_grounded` (count),
and `grounding_clean` (0/1 — every `literal` has a `maps_to`, every `vibe` has none).

> Note: `_extract_input_from_case` stores `{place}` (not `{book_title, author}`) for
> place→book cases, and carries the grounding expectations (`expect`, `min_literal`,
> `expected_literal_examples`) into item metadata so the runner has a target to score.

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
- **Per-Agent Metrics**: Token usage per agent call
- **Cost Calculation**: Automatic cost estimation based on Gemini pricing
- **Trace Hierarchy**: Nested spans for agents, generations, and tools
- **Fast Preview**: Enable the "Fast (Preview)" toggle in Langfuse for real-time ingestion (requires Langfuse SDK v4)

### Current Gemini Pricing

Pricing is model-aware (as of May 2026, standard non-batch rates):

| Model | Input / 1M | Output / 1M |
|---|---|---|
| gemini-2.5-flash | $0.30 | $2.50 |
| gemini-2.5-flash-lite | $0.10 | $0.40 |
| gemini-2.0-flash | $0.10 | $0.40 |
| gemini-1.5-flash | $0.075 | $0.30 |
| gemini-1.5-pro | $1.25 | $5.00 |

Unknown models fall back to gemini-2.5-flash rates. Source: [Google AI Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing).

### Enabling the Plugin

The plugin is automatically initialized if credentials are provided. It auto-disables if credentials are missing:

```python
from plugins.langfuse_plugin import LangfusePlugin

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

### Programmatic Access

```python
stats = langfuse_plugin.get_session_stats()
print(f"Total tokens: {stats['total_tokens']}")
print(f"Cost: ${stats['cost_usd']:.6f}")

await langfuse_plugin.flush()
```

### Architecture

The plugin uses Google ADK's callback system to hook into the agent lifecycle. It runs alongside the ADK runner and receives events at each stage:

```
ADK Runner (runner.run_async)
    ↓
on_user_message_callback    → opens root span, sets user_id/session_id via
                              propagate_attributes() (Langfuse v4 OTel context)
before_agent_callback       → opens child span per agent
before_model_callback       → opens generation span (LLM call)
after_model_callback        → records token counts + cost via usage_details/cost_details
after_tool_callback         → closes tool span
after_agent_callback        → closes agent span
after_run_callback          → closes root span, flushes stats
```

Token extraction from Gemini responses:
```python
response.usage_metadata.prompt_token_count      # Input tokens
response.usage_metadata.candidates_token_count  # Output tokens
response.usage_metadata.total_token_count       # Total tokens
```

### Trace Structure in Langfuse

A complete evaluation run produces this span hierarchy:

```
eval_run_YYYYMMDD_v2  (dataset run root — from start_as_current_observation)
├── book_to_place_discovery_invocation  (plugin root span)
│   └── book_to_place_discovery
│       ├── book_context_researcher
│       │   └── gemini-2.5-flash_call  (generation, with token counts)
│       ├── book_context_formatter
│       │   └── gemini-2.5-flash_call
│       ├── city/landmark/author researcher→formatter  (parallel graph branches)
│       │   └── ...
│       └── region_analyzer
│           └── gemini-2.5-flash_call
├── book_to_place_composition_invocation  (plugin root span)
│   └── ...
└── llm_score_itinerary  (generation — from @observe in llm_scorer.py)
    model: gemini-2.5-flash-lite, input/output tokens, scores as output
```

**Two tracing mechanisms used:**
- **ADK plugin callbacks** — everything inside `runner.run_async()`: agents, model calls, tools. This is the only way to hook into ADK's internal lifecycle.
- **`@observe` decorator** — `score_itinerary()` in `llm_scorer.py`. Used for standalone LLM calls outside the ADK runner. Note: `@observe` is not used on `discover()`/`compose()` because those are async generators — OTel context is lost after each `yield` (Langfuse issue [#8447](https://github.com/langfuse/langfuse/issues/8447)).

### Disabling the Plugin

**Option 1:** Omit from Runner plugins:
```python
runner = Runner(agent=my_agent, plugins=[LoggingPlugin()])
```

**Option 2:** Remove credentials from `.env`.

## Viewing Data in Langfuse

1. Log in to [cloud.langfuse.com](https://cloud.langfuse.com)
2. Select your project
3. Enable **"Fast (Preview)"** toggle (top-right) for real-time ingestion — requires Langfuse SDK v4 ✓
4. **Traces** — One trace per eval case; contains the full span tree above
5. **Observations** — Filterable list of all spans and generations across all traces
6. **Datasets** — Open `storyland_eval` or `books_v1`, select a Run to see per-item scores and trace links
7. **Scores** — Quality scores (book_relevance, completeness, etc.) attached to each trace

## Directory Structure

```
evaluation/
├── README.md                      # This file
├── tools/                         # Evaluation scripts
│   ├── run_scheduled_eval.py      # Itinerary (book→place) eval runner — LLM-as-judge
│   ├── run_place_to_book_eval.py  # Place→book reverse-routing grounding eval runner
│   ├── eval_dashboard.py          # Dashboard and reporting
│   ├── langfuse_eval.py           # Dataset creation pipeline
│   ├── llm_scorer.py              # LLM-as-judge quality scoring
│   └── setup_langfuse_eval.sh     # Setup script
├── storyland_eval.evalset.json    # Dataset (8 diverse books)
├── books_v1.evalset.json          # Dataset (10 books with expected output + criteria)
├── place_to_book_v1.evalset.json  # Dataset (11 place→book reverse-routing grounding cases)
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
