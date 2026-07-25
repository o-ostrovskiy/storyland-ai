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

This creates the Langfuse datasets:
- `storyland_eval` — 8 diverse scenarios for comprehensive testing
- `books_v1` — 10 cases with expected outputs and book-specific scoring criteria
- `place_to_book_v1` — 11 cases for the **place→book reverse-routing** grounding gate (literal/vibe labelling + not-found state). Register with `make eval-setup-one EVALSET_FILE=evaluation/place_to_book_v1.evalset.json`.
- `local_atmosphere_v1` — 8 cases for the **local-atmosphere** ("book near me") flow: mixed preference shapes (4 with / 4 without `user:preferences`), LLM judge + deterministic envelope/radius gate.
- `expansion_v1` — 5 cases for the **expansion** (suggestion-chip) flow: deterministic two-step gate (compose → expand), no judge.

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

### Local-atmosphere eval ("book near me")

The `local_atmosphere_v1` dataset evaluates the single-phase local-atmosphere flow
(`WorkflowExecutor.local_atmosphere` — the same code path as
`POST /itinerary/local-atmosphere`): given a book plus a user location
(`location_label`, `lat`, `lng`) and a `radius_km`, compose a nearby itinerary whose
mood evokes the book. Two layers of scoring:

- **Deterministic gate** — the result must validate as a `ComposerEnvelope`, plus an
  opportunistic haversine radius check over any geo fields in the payload (`CityStop`
  carries no coordinates today, so this normally reports `no_geo_fields` and radius
  adherence is asserted via each case's `geographical_accuracy` judge criterion).
- **LLM judge** — the standard `llm_scorer.score_itinerary` path (6 dimensions).

```bash
make eval-local-atmosphere
# Or directly (re-registers the dataset items from the evalset, then runs all cases):
python evaluation/tools/run_local_atmosphere_eval.py [--max-cases N] [--no-register]
```

Each case logs a Langfuse dataset run (`la_eval_YYYYMMDD_HHMMSS`) with scores:
`case_pass` (0/1), `envelope_valid` (0/1), `radius_within` (0/1 — only when geo fields
were found), the judge dimensions, and `judge_average`. Per the eval protocol,
`preference_adherence` is reported **only for cases that carry `user:preferences`**,
and the report includes per-preference-shape aggregation plus a `mechanism` section.

### Expansion eval (suggestion chips)

The `expansion_v1` dataset is a **deterministic two-step gate** for the expansion flow
(no LLM judge — like the place→book gate). Each case drives the live
`WorkflowExecutor` end to end: `discover` → `compose` (first region) → pick a
suggestion chip (`chip_keyword` substring match on label/action_prompt, else the first
chip) → `expand`. The harness deliberately uses the executor rather than the raw
workflow: the invariants under test — `source="expansion"` stamping, the dedupe
post-filter, chip-id stamping, trusted-chip resolution (MYS-167) — are executor logic.

Checks per case: at least `min_new_places` new stops; every new stop stamped
`source="expansion"`; no duplicates against the base itinerary (case-insensitive name
match); every new stop validates as a `CityStop`; follow-up chips (≤4) carry non-empty
unique server-stamped ids. A compose that legally returns zero chips is recorded as
`no_chips` and counted skipped, never failed.

```bash
make eval-expansion
# Or directly:
python evaluation/tools/run_expansion_eval.py [--max-cases N] [--no-register]
```

Each case logs a Langfuse dataset run (`exp_eval_YYYYMMDD_HHMMSS`) with scores:
`case_pass`, `no_duplicates`, `source_stamped`, `places_schema_valid`,
`chip_ids_stamped` (all 0/1), `new_places_count`, and `chips_available`.

### Which model runs where

Two models participate in every judged eval run — keep them straight when reading results:

| Role | Model | Where it's set |
|------|-------|----------------|
| **System under test** — the agent workflows being evaluated (discovery, composition, local-atmosphere, expansion, place→book) | `MODEL_NAME` env var (repo default `gemini-3.1-flash-lite`, `common/config.py`; the CI eval workflow sets the same) | Stamped as `model_under_test` in each dedicated runner's Langfuse run metadata and results JSON — baselines are model-bound, and a model lift must read as a re-baseline, not a regression. |
| **LLM-as-judge** — scores the outputs | `gemini-2.5-flash-lite`, fixed | Default in `llm_scorer.score_itinerary`, passed explicitly by `run_scheduled_eval.py`. Deliberately decoupled from `MODEL_NAME`: a fixed judge keeps scores comparable across system-model lifts (changing both at once would confound every delta). |

Production reads `MODEL_NAME` from `.env.prod` on the box (deploys never overwrite it), so the prod value can drift from the repo default — check recent `production`-env generations in Langfuse for the authoritative answer (the plugin records the configured model string per generation).

### Eval protocol

Rules that apply to every evalset and eval report in this repo:

1. **`flow` routing.** Every evalset declares a top-level `"flow"` field
   (`itinerary` [default when absent], `place_to_book`, `local_atmosphere`,
   `expansion`). The dataset sync records it in `evaluation/langfuse_datasets.json`,
   and `run_scheduled_eval.py`'s auto-discovery only consumes `flow: itinerary`
   datasets — others are logged as routed to their dedicated runner (INFO, not a
   failure). This prevents the scheduled itinerary runner from either skip-failing on
   format-incompatible datasets or silently running them through the wrong flow.
   A new evalset for a non-itinerary flow **must** carry the field (pinned by
   `tests/unit/test_eval_dataset_routing.py`).
2. **Per-shape aggregation.** Wherever preferences are involved, reports aggregate
   judge scores separately for cases with and without `user:preferences`. The
   preference-free aggregate excludes `preference_adherence` (the judge always emits
   all 6 dimensions, but with no preferences it is judged against nothing), and
   preference-free cases must not carry a `preference_adherence` quality criterion
   (the dataset sync warns on violations).
3. **`mechanism` section.** Every eval report (results JSON + printed summary) carries
   a `mechanism` block stating how scores were produced: which checks are
   deterministic, which are LLM-judged and by what model, and the pass rule.

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

For books_v1, the dataset's book-specific `quality_criteria` are **not** injected
into the quality judge — they are scored separately as **`criteria_coverage`**
(1–5, compliance with the listed criteria only, never blended into the quality
average). The MYS-586 ablation showed the injection made the judge grade
compliance-with-specifics rather than quality (divergence vs a second model's
reading: −0.95 with injection, +0.07 without). One number carrying two meanings
means a reader can't tell which half moved. The local-atmosphere runner still
injects its criteria into the judge deliberately (radius adherence rides on it).

### Gate status & caveats (post-calibration, 2026-07)

Full evidence: `evaluation/calibration/agreement_report.md` and
`evaluation/calibration/rebaseline_2026-07.md`. The short version every reader
of eval numbers needs:

- **Both dataset gates are catastrophe detectors, not quality floors.** The
  conservative derivation (pooled − 2×historical maxΔ) puts storyland_eval at
  ≥ 4.03 against a pooled 4.37 and books_v1 at ≥ 2.14 against 2.94 — far below
  their means by construction. Passing a gate means "no catastrophe," nothing
  more.
- **The storyland_eval judge is trustworthy** (±0.3 of a careful independent
  reading; stable across the MYS-586 prompt changes). Trusting the judge and
  trusting the gate are different claims — the gate is still loose.
- **books_v1 quality deltas are believable post-split** (re-baselined
  2026-07-21, pooled 3.90: `rebaseline_split_2026-07.md`). The split-config
  judge matched the second-model reading at +0.07 bias / 0.45 MAD; pre-split
  numbers (≤2.94-era) were similarity-dominated and are not comparable.
  `criteria_coverage` is the separate compliance read (~2.3–2.4 currently) —
  never blend it back into quality when quoting either.
- **The judge cannot fact-check geography.** `geographical_accuracy` scored 5
  on an itinerary placing "Portland Observatory" in Oregon (it's in Maine) and
  1–2 on verifiably correct St. Petersburg addresses. Deterministic checks
  (e.g. the local-atmosphere radius gate) are the instrument for geographic
  claims; never cite this dimension alone as evidence of a geographic error.
- **The 2026-07 calibration labels are model labels** (Claude Fable 5,
  Olga-approved), not human ground truth: agreement numbers are
  judge-vs-second-model divergence. Weekly spot-checks are the channel where
  true human anchors accumulate — a flagged case stays `pending_review` until
  a real `human_*` score lands, so the absence of human anchors stays visible.

### Judge Calibration (human labels)

The judge has never been anchored to human labels, and its run-to-run noise is
large (books_v1 ±0.40, storyland_eval ±0.17 on identical code) — so judge
scores gate only catastrophes until calibrated. `evaluation/tools/judge_calibration.py`
manages the calibration loop against Langfuse annotation queues:

```bash
# 1. Build the labeling queue: pulls ~30 judge-scored itineraries from recent
#    dataset runs, creates human_<dimension> score configs (1-5, exact rubric
#    text from SCORING_CRITERIA), enqueues the traces, writes a manifest to
#    evaluation/calibration/. Needs LANGFUSE_* creds in .env.
python evaluation/tools/judge_calibration.py build-queue          # --dry-run to preview

# 2. Hand-label in Langfuse: Annotation Queues → judge-calibration-2026-07.
#    Score from the trace OUTPUT (itinerary) + INPUT (book, preferences);
#    don't look at the judge's existing scores first.

# 3. Compute judge-vs-human agreement (per-dimension mean absolute difference
#    and direction bias, |Δ|≥2 disagreement list):
python evaluation/tools/judge_calibration.py agreement
```

The manifest (`evaluation/calibration/queue_manifest_*.json`) is the join key
between judge scores and human labels — commit it. Rubric re-anchoring edits
to `SCORING_CRITERIA` and gate-threshold recomputation are driven by the
agreement report.

### Weekly Human Spot-Check

Each scheduled eval run randomly flags up to 2 scored cases for human review
(seeded by run date, so a same-day re-run flags the same cases). The selection
is recorded in the results JSON under `human_spot_check` with
`status: pending_review`, and the trend report renders a "Human spot-checks"
section where unreviewed cases stay listed with their age — skipped reviews
are visible, not silent. A case counts as reviewed once its trace carries any
`human_*` label score in Langfuse (entered via the annotation UI or the API).

Set `LANGFUSE_REVIEW_QUEUE_ID` (e.g. as a GitHub Actions variable) to have
flagged traces auto-enqueued into an annotation queue; enqueue failures are
non-fatal and the JSON record remains the source of truth.

Results JSONs also now carry the full `itinerary` payload, the `preferences`
the judge saw, and the Langfuse `trace_id` per case, so review and calibration
work from artifacts alone.

### Automated Evaluations

Evaluations run automatically via GitHub Actions:

- **Schedule**: Every Monday at 9 AM UTC
- **Workflow**: `.github/workflows/scheduled-eval.yml`
- **Results**: Uploaded as workflow artifacts

Manual trigger: Go to Actions tab → Evaluation (manual dispatch) → Run workflow

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

Pricing is model-aware (as of Jul 2026, standard non-batch rates):

| Model | Input / 1M | Output / 1M |
|---|---|---|
| gemini-3.1-flash-lite | $0.25 | $1.50 |
| gemini-2.5-flash | $0.30 | $2.50 |
| gemini-2.5-flash-lite | $0.10 | $0.40 |
| gemini-2.0-flash | $0.10 | $0.40 |
| gemini-1.5-flash | $0.075 | $0.30 |
| gemini-1.5-pro | $1.25 | $5.00 |

Matching is by substring with longest-key-first (so a `-preview` alias bills at its base model's rates). Unknown models fall back to gemini-2.5-flash rates. Source: [Google AI Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing).

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
6. **Datasets** — Open `storyland_eval`, `books_v1`, `place_to_book_v1`, `local_atmosphere_v1`, or `expansion_v1`, select a Run to see per-item scores and trace links
7. **Scores** — Quality scores (book_relevance, completeness, etc.) attached to each trace

## Directory Structure

```
evaluation/
├── README.md                      # This file
├── tools/                         # Evaluation scripts
│   ├── run_scheduled_eval.py      # Itinerary (book→place) eval runner — LLM-as-judge
│   ├── run_place_to_book_eval.py  # Place→book reverse-routing grounding eval runner
│   ├── run_local_atmosphere_eval.py  # Local-atmosphere eval runner (judge + deterministic gate)
│   ├── run_expansion_eval.py      # Expansion (suggestion-chip) deterministic eval runner
│   ├── eval_dashboard.py          # Dashboard and reporting
│   ├── langfuse_eval.py           # Dataset creation pipeline
│   ├── llm_scorer.py              # LLM-as-judge quality scoring
│   ├── judge_calibration.py       # Human-label queue + judge-vs-human agreement
│   └── setup_langfuse_eval.sh     # Setup script
├── storyland_eval.evalset.json    # Dataset (8 diverse books)
├── books_v1.evalset.json          # Dataset (10 books with expected output + criteria)
├── place_to_book_v1.evalset.json  # Dataset (11 place→book reverse-routing grounding cases)
├── local_atmosphere_v1.evalset.json  # Dataset (8 local-atmosphere cases, mixed preference shapes)
├── expansion_v1.evalset.json      # Dataset (5 expansion two-step deterministic cases)
├── langfuse_datasets.json         # Dataset registry incl. per-dataset flow (gitignored)
├── calibration/                   # Judge-calibration manifests + agreement reports (tracked)
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
