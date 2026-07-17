# StoryLand AI

> Turn your favorite books into meaningful travel experiences

**We're live! Try it at [mystoryland.ai](https://mystoryland.ai/)**

## Overview

StoryLand AI transforms the worlds within beloved books into real, actionable travel plans. When readers finish a book they love, they often want to explore the places that inspired it—but turning that impulse into reality requires navigating scattered information across countless sources. StoryLand AI solves this by using a multi-agent system that automatically researches, discovers, and composes personalized travel itineraries based on any book.

## The Problem

When someone finishes a book they love, they want to stay in that world a little longer. They want to:
- Walk the same streets the characters walked
- Feel the atmosphere of the city where the story unfolded
- Visit the real places that inspired the author

But the moment they try, the magic disappears.

### Current Reality

Readers face:
- Endless Google searches across scattered blogs, tourist sites, and forums
- Contradictory or incomplete information
- Too many tabs, too much noise, too many missing pieces
- Frustration that kills the initial inspiration

**Result:** Most people never follow through on exploring the world behind the story.

## The Solution: Multi-Agent Architecture

StoryLand AI uses specialized AI agents that work together to solve this complex problem. Each agent is an expert in a specific part of the journey from book to travel plan.

### Why Agents?

Creating a meaningful travel plan from a book requires:
- **Multiple information sources:** Settings, real locations, cultural context, maps, routes, historical notes, travel details
- **Parallel research:** Looking up landmarks, author sites, filming locations, museums simultaneously
- **Intelligent coordination:** Merging disparate information into a coherent journey
- **Personalization:** Remembering preferences across multiple books

A single LLM prompt can't handle this complexity. Agents can.

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd storyland-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Optional: Configure Langfuse for LLM observability
# Add your Langfuse credentials to .env:
# - LANGFUSE_SECRET_KEY
# - LANGFUSE_PUBLIC_KEY
# - LANGFUSE_HOST
```

### Run

```bash
# Start the FastAPI SSE API server
make run-api
# curl http://localhost:8080/api/v1/health
```

**What you get:**
- Region options to choose from
- Cities to visit with suggested days
- Landmarks tied to the book's setting
- Author-related sites (birthplace, museums, etc.)
- Practical travel details and tips

See [Configuration](#configuration) for environment variables.

## Architecture

StoryLand AI uses a two-phase workflow with human-in-the-loop region selection. Book title and author are pre-confirmed by the upstream service (Backend → storyland-ai):

```mermaid
flowchart TB
    subgraph Input["Input (pre-confirmed)"]
        DTO[book_title + author<br/>📖 from Backend]
    end

    subgraph Phase1["Phase 1: Discovery"]
        BC[book_context_pipeline<br/>🔍 Setting & themes]
        RP[reader_profile_agent<br/>👤 User preferences]

        subgraph PD["Parallel Discovery ⚡"]
            CP[City Agent<br/>🏙️]
            LP[Landmark Agent<br/>🏛️]
            AP[Author Agent<br/>✍️]
        end

        BC --> PD
        RP --> PD
        PD --> RA[region_analyzer<br/>🌍 Geographic grouping]
    end

    subgraph HITL["Human Selection"]
        Select[Choose region to explore]
    end

    subgraph Phase2["Phase 2: Composition"]
        TC[trip_composer<br/>🗺️ Personalized itinerary]
    end

    DTO --> BC
    RA --> Select
    Select --> TC
```

**Two-phase design:**
1. **Phase 1 (Discovery)**: Finds all locations and groups them into practical travel regions
2. **Human Selection**: User chooses which region(s) to explore (prevents impractical multi-continent itineraries)
3. **Phase 2 (Composition)**: Creates detailed itinerary for selected region(s) only

### Key Components

- **Book Context Agent** - Researches setting, themes, and time period
- **Discovery Agents** - Parallel search for cities, landmarks, and author sites
- **Region Analyzer** - Groups cities by geographic proximity using LLM world knowledge
- **Trip Composer** - Creates personalized itinerary based on user preferences
- **Reader Profile Agent** - Accesses user preferences from session state
- **Local Atmosphere Pipeline** - Single-phase researcher → formatter that finds nearby places (within ~80 km of the user) whose mood and aesthetic evoke the book, for readers who can't travel to the actual setting
- **Expansion Pipeline** - On-demand researcher → formatter that adds 3-5 new places to an existing city when the user clicks a suggestion chip
- **Book Recommendation Agent** - On-demand single agent that recommends 5 books based on the source book, destination cities, and themes (triggered by the "Find books like this" chip)

**Session State Keys:**

| Key | Phase | Description |
|-----|-------|-------------|
| `book_metadata` | 1 | Exact title and author (from request DTO) |
| `book_context` | 1 | Setting, themes, time period |
| `city_discovery` | 1 | Cities with literary connections |
| `landmark_discovery` | 1 | Specific landmarks and sites |
| `author_sites` | 1 | Author-related locations |
| `region_analysis` | 1 | Geographic region grouping |
| `user:preferences` | All | User travel preferences (persists across sessions) |
| `final_itinerary` | 2 | Complete travel plan |
| `last_suggestions` | 2 | Current set of suggestion chips |
| `expansion_count` | Post-2 | Number of place-expansion requests made |
| `book_recommendation_chip_id` | 2 | ID of the server-stamped "Find books like this" chip |
| `book_recommendation_count` | Post-2 | Number of book recommendation requests made |
| `last_book_recommendations` | Post-2 | Most recent BookRecommendationsResult |
| `job_failed` | All | Terminal failure marker (set on error, cleared on compose retry) |

**Storage:** In-memory (default) or SQLite persistence. Multi-user support with isolated data. See [Configuration](#configuration) for options.

**Technology Stack:**
- [Google Agent Development Kit (ADK)](https://github.com/google/adk-python)
- Google Gemini 2.0/2.5 models (configurable)
- Pydantic for data validation
- SQLite for persistence

### API (FastAPI SSE)

StoryLand AI includes a FastAPI server with Server-Sent Events streaming for web/mobile clients. The API offers a two-phase travel-planning flow plus a single-phase **local-atmosphere** mode for users who can't travel:

```bash
# Start the API server
make run-api   # http://localhost:8080

# Health check (always open — no secret required)
curl http://localhost:8080/api/v1/health

# Discover locations (phase 1) — streams progress events, returns regions
# Add -H "X-Internal-Secret: <value>" if INTERNAL_API_SECRET is set in .env
curl -N -X POST http://localhost:8080/api/v1/itinerary/discover \
  -H "Content-Type: application/json" \
  -d '{"book_title": "1984", "author": "George Orwell"}'

# Compose itinerary (phase 2) — streams final itinerary
curl -N -X POST http://localhost:8080/api/v1/itinerary/{job_id}/compose \
  -H "Content-Type: application/json" \
  -d '{"region_ids": [1]}'

# Local atmosphere (single-phase) — places near the user that match the book's mood
curl -N -X POST http://localhost:8080/api/v1/itinerary/local-atmosphere \
  -H "Content-Type: application/json" \
  -d '{
        "book_title": "Wuthering Heights",
        "author": "Emily Brontë",
        "user_location": {"label": "New York, NY", "lat": 40.7128, "lng": -74.0060},
        "radius_km": 80
      }'

# Check job status
curl http://localhost:8080/api/v1/itinerary/{job_id}/status
```

SSE event types: `progress`, `metadata`, `regions`, `itinerary`, `expansion`, `book_recommendations`, `error`, `done`. The local-atmosphere endpoint emits `progress → metadata → itinerary → done` (no `regions` event). The `/expand` endpoint emits `progress → expansion → done`. The `/recommend-books` endpoint emits `progress → book_recommendations → done`.

**Job status states** (`GET /api/v1/itinerary/{job_id}/status`):

| Status | Meaning |
|--------|---------|
| `searching` | Phase 1 (Discovery) in progress |
| `discovering` | Discovery running, collecting locations |
| `regions_ready` | Discovery complete, awaiting region selection |
| `composing` | Phase 2 (Composition) in progress |
| `completed` | Itinerary ready |
| `failed` | Terminal error — book not found, timeout, or cancellation (takes priority over all other states) |

Failed jobs can be retried by calling `/compose` again with the same `job_id`; the failure marker is cleared automatically when a new compose attempt begins.

### AI Models

StoryLand AI uses Google Gemini models (default: `gemini-2.5-flash-lite`) for all agents, chosen for native ADK integration, fast parallel execution (sub-2s response times), and excellent structured output adherence across 16 Pydantic data models. The complete workflow takes 60-100 seconds end-to-end with parallel discovery providing 3x speedup over sequential execution.

## Configuration

All configuration is via environment variables in `.env`. Copy `.env.example` to get started.

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | **Required.** Google AI API key from [AI Studio](https://aistudio.google.com/app/apikey) | — |
| `MODEL_NAME` | Gemini model to use | `gemini-2.5-flash-lite` |
| `USE_DATABASE` | Enable SQLite persistence | `false` |
| `DATABASE_URL` | SQLite database path | `sqlite+aiosqlite:///storyland_sessions.db` |
| `SESSION_MAX_EVENTS` | Max events in session | `20` |
| `MAX_CONTEXT_TOKENS` | Max tokens for context window | `30000` |
| `WORKFLOW_TIMEOUT` | Max seconds for entire workflow | `300` |
| `AGENT_TIMEOUT` | Max seconds per agent | `60` |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `ENABLE_ADK_DEBUG` | Enable ADK internal debug logging | `false` |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key (optional) | — |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key (optional) | — |
| `LANGFUSE_HOST` | Langfuse host URL (optional) | — |
| `SENTRY_DSN` | Sentry error-tracking DSN (optional) — unset means Sentry is fully disabled (local/CI). Errors only by default; performance tracing stays off since agent runs are already traced in Langfuse. | — |
| `SENTRY_ENABLE_LOGS` | Ship INFO+ logs to Sentry Logs (searchable/alertable). Kill switch — only takes effect when `SENTRY_DSN` is set. structlog error/critical become Sentry events, info/warning become Sentry logs, debug stays local (LLM prompts can appear there). | `true` |
| `SENTRY_TRACES_SAMPLE_RATE` | Sentry performance-tracing sample rate (0.0–1.0) | `0.0` |
| `CORS_ORIGINS` | Allowed CORS origins for API (comma-separated) | `*` |
| `INTERNAL_API_SECRET` | Shared secret with the gateway service — when set, all itinerary endpoints require an `X-Internal-Secret` header with this value. The health endpoint (`/health`) is always open. Leave empty for standalone/dev use. | — |

Free tier includes 15 RPM and 200 requests/day — sufficient for development.

## Development

### Project Structure

```
storyland-ai/
├── models/              # Pydantic data models
│   ├── book.py          # BookMetadata, BookContext, BookInfo
│   ├── discovery.py     # CityDiscovery, LandmarkDiscovery, AuthorSites,
│   │                    # RegionCity, RegionOption, RegionAnalysis
│   ├── itinerary.py     # TripItinerary, CityPlan, CityStop
│   ├── place_to_book.py # PlaceBookCandidate/PlaceToBookCandidates/Result (reverse discovery)
│   └── preferences.py   # TravelPreferences
│
├── tools/               # ADK tool integrations
│   └── preferences.py   # Session state preferences tool
│
├── agents/              # AI agent definitions
│   ├── book_context_agent.py     # Book setting research
│   ├── discovery_agents.py       # City/landmark/author discovery
│   ├── trip_composer_agent.py    # Itinerary composition
│   ├── local_atmosphere_agent.py    # Researcher+formatter for "near me" mode
│   ├── expansion_agent.py           # Researcher+formatter for place expansion chips
│   ├── book_recommendation_agent.py # Researcher+formatter for "Find books like this" chip
│   ├── place_to_book_agent.py        # Researcher+formatter for place→book reverse discovery
│   ├── reader_profile_agent.py      # Preferences-based personalization
│   ├── region_analyzer_agent.py     # Geographic region grouping
│   ├── orchestrator.py              # Two-phase + local-atmosphere + expansion + book-rec + place→book workflows
│   ├── prompts.py                   # AgentPrompts dataclass + versioned loader
│   └── prompts/                  # Versioned prompt sets
│       ├── v1.json               # Original prompts (git ref 4c6fdc9)
│       └── v2.json               # Current prompts (PR #63)
│
├── core/                # Transport-agnostic orchestration & SDK
│   ├── executor.py      # WorkflowExecutor (discover/compose/expand/recommend)
│   ├── place_to_book.py # PlaceToBookResolver (place→book reverse routing, isolated capability)
│   ├── cache.py         # In-process TTL/LRU result cache
│   └── session_state.py # Typed session-state accessor
│
├── api/                 # FastAPI SSE streaming API
│   ├── app.py           # Application factory with lifespan
│   ├── routes.py        # HTTP endpoints (discover, compose, expand, recommend-books, local-atmosphere, status, health)
│   ├── streaming.py     # SSE async generators wrapping ADK Runner
│   ├── models.py        # Request/response/SSE event Pydantic models
│   └── dependencies.py  # Shared app state & dependency injection
│
├── services/            # Core services
│   └── session_service.py   # Session management (InMemory/SQLite)
│
├── common/              # Shared utilities
│   ├── config.py        # Configuration management
│   └── logging.py       # Structured logging (structlog)
│
├── plugins/             # ADK runner plugins
│   └── langfuse_plugin.py  # Langfuse observability & token tracking
│
├── tests/               # Test suite
│   ├── unit/            # Unit tests (no API calls)
│   └── integration/     # Integration tests (VCR cassettes)
│
├── pyproject.toml       # Dependencies & pytest config
└── .env.example         # Environment template
```

### Extending the Codebase

**Adding a new agent** — create in `agents/`, import in `orchestrator.py`:
```python
from google.adk.agents import LlmAgent
def create_my_agent(model):
    return LlmAgent(name="my_agent", model=model, instruction="...")
```

**Adding a new tool** — create in `tools/`:
```python
from google.adk.tools import FunctionTool
def my_function(query: str) -> str:
    return result
my_tool = FunctionTool(my_function)
```

**Adding a new model** — create in `models/`:
```python
from pydantic import BaseModel, Field
class MyModel(BaseModel):
    field1: str = Field(description="Description")
```

### Prompt Engineering

Agent prompts include reliability improvements:

- **Anti-hallucination:** `"If the research found no cities, return an empty list - do not hallucinate."`
- **Error handling:** `"If the tool returns an error, report it clearly and explain what went wrong"`
- **Disambiguation:** Book title and author injected into search queries to avoid confusion with similarly-named books

**Versioned prompts** — all agent instructions live in `agents/prompts/v2.json` (and `v1.json` for history). To add a new prompt version, create `agents/prompts/v3.json` and pass `--prompt-version v3` to the eval runner. The current version (`v2`) is controlled by `CURRENT_PROMPT_VERSION` in `agents/prompts.py`. Prompt change history is in [`evaluation/PROMPT_CHANGELOG.md`](evaluation/PROMPT_CHANGELOG.md).

## Deployment

Production runs on a single self-hosted Lightsail box via docker compose (see
`storyland-infrastructure/deploy/`). Releases go through the **G5-gated
`Deploy AI (prod)`** GitHub Action (`.github/workflows/deploy-ai-prod.yml`):

- It is `workflow_dispatch`-only, so it never fires automatically.
- The run pauses on the `production` GitHub Environment for the required
  reviewer's (Olga's) approval — that pause **is** the G5 gate.
- After approval it calls the reusable SSH-deploy workflow in
  `storyland-infrastructure` (`deploy-service.yml`, `service: storyland-ai`),
  which rsyncs the source to the box and rebuilds/restarts the `storyland-ai`
  service.

On-box runtime config (`.env.prod`: `GOOGLE_API_KEY`, `INTERNAL_API_SECRET`,
`LANGFUSE_*`, `CACHE_TTL_SECONDS`, `CACHE_MAX_ENTRIES`) is the source of truth and
is excluded from the rsync, so a rebuild never clobbers it. The release action is
post-merge only and never triggers an eval (eval = real Gemini spend). Manual
fallback: `KEY=~/.ssh/storyland.pem storyland-infrastructure/deploy/deploy.sh storyland-ai`.

One-time setup (out-of-band, not in the repo): add the `LIGHTSAIL_SSH_KEY` and
`BOX_IP` repo secrets and create a `production` Environment with a required
reviewer.

## Testing

```bash
make test                  # Unit tests (442 tests)
make test-integration      # Integration tests with VCR cassettes (excludes real_api)
make test-integration-live # Live tests that hit real APIs (real_api marker; uses quota)
make test-all              # Both
make test-cov              # With coverage
```

| Module | Tests | Description |
|--------|-------|-------------|
| `test_models.py` | 79 | Pydantic model validation (incl. BookRecommendation) |
| `test_tools.py` | 4 | Preferences tool |
| `test_agents.py` | 49 | Agent factory functions (incl. book recommendation pipeline) |
| `test_services.py` | 10 | Session service |
| `test_llm_scorer.py` | 23 | LLM scoring models and prompts |
| `test_cache.py` | 14 | Discovery result cache (TTL/LRU, key normalization, cache-hit short-circuit) |
| `test_core.py` | 89 | Events, session state, extraction, regions, prompts (incl. book recs) |
| `test_api.py` | 108 | API models, endpoints, SSE streaming (incl. book recommendations) |
| `test_place_to_book.py` | 33 | Place→book reverse routing: normalization, grounding filter, label invariants, resolver |

Integration tests use [VCR.py](https://vcrpy.readthedocs.io/) to record/replay HTTP interactions. For quality evaluation, see [evaluation/README.md](evaluation/README.md).

### Live cache verification (`real_api`)

`tests/integration/test_cache_real_api.py` is the live counterpart to the mocked
cache-hit unit test. The Discovery result cache is always on, so it calls
`discover()` twice with an identical book/author and meters
`Gemini.generate_content_async`
directly (call count + summed `usage_metadata.total_token_count`) to prove the
**second call is a cache hit that makes zero new Gemini calls and consumes zero
new Gemini tokens** while returning the same regions. It is marked `real_api`, so
it is excluded from the default `make test-integration` run and only fires via
`make test-integration-live`. It hits the live API (no VCR — recorded responses
would defeat token counting), so it needs `GOOGLE_API_KEY` and uses a small
amount of quota; it skips when the key is absent.

## Troubleshooting

- **Rate limits (429):** Retry logic handles automatically. Wait ~60s between books.
- **API key issues:** `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OK' if os.getenv('GOOGLE_API_KEY') else 'MISSING')"`
- **Database issues:** `rm storyland_sessions.db` to start fresh
- **Virtual env not activated:** `source .venv/bin/activate`
- **Missing dependencies:** `pip install -e ".[dev]"`
- **Timeout errors:** Increase `WORKFLOW_TIMEOUT` in `.env` (default: 300s)

## Database Reference

When `USE_DATABASE=true`, ADK's `DatabaseSessionService` creates a `sessions` table:

```sql
CREATE TABLE sessions (
    app_name VARCHAR(128) NOT NULL,
    user_id VARCHAR(128) NOT NULL,
    id VARCHAR(128) NOT NULL,          -- note: "id", not "session_id"
    state TEXT NOT NULL,               -- JSON-encoded session state
    create_time DATETIME NOT NULL,     -- note: "create_time", not "created_at"
    update_time DATETIME NOT NULL,
    PRIMARY KEY (app_name, user_id, id)
)
```

**State scopes:** No prefix = session-scoped (ephemeral). `user:` prefix = persists across sessions. `app:` prefix = global.

**Common queries:**
```sql
-- User's book history
SELECT id, json_extract(state, '$.book_title') as book, create_time
FROM sessions WHERE app_name = 'storyland' AND user_id = 'alice'
ORDER BY create_time DESC;

-- User preferences (keys with colons must be quoted)
SELECT json_extract(state, '$."user:preferences"') as preferences
FROM sessions WHERE app_name = 'storyland' AND user_id = 'alice'
ORDER BY create_time DESC LIMIT 1;
```

**Maintenance:**
```bash
rm storyland_sessions.db                    # Start fresh
sqlite3 storyland_sessions.db "VACUUM;"     # Reclaim space
cp storyland_sessions.db backup_$(date +%Y%m%d).db  # Backup
```

## Documentation

- **[Architecture Decisions](docs/ARCHITECTURE.md)** - ADRs explaining key design choices
- **[Evaluation & Observability](evaluation/README.md)** - Quality evaluation pipeline and Langfuse token/cost tracking

## Why StoryLand AI?

**For Readers:**
- Turn literary inspiration into real travel experiences
- Save hours of research and planning
- Discover places you'd never find through conventional searches
- Get personalized recommendations that match your travel style

**For Developers:**
- Demonstrates real-world multi-agent coordination
- Shows parallel vs. sequential agent patterns
- Implements session-based personalization
- Production-ready with database persistence and error handling

## Vision

A great book doesn't just end—it opens a door. StoryLand AI helps readers step through it.

We believe that every story deserves to be experienced beyond the page, and every reader should be able to walk through the worlds they love without the friction of scattered information and endless research.

**A single prompt can describe a world—but only agents can build a bridge between that world and real places someone can actually visit.**

---

*Built with Google Agent Development Kit*
