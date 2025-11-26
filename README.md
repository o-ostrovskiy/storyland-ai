# StoryLand AI

> Turn your favorite books into meaningful travel experiences

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

## Table of Contents

- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [CLI Usage](#cli-usage)
- [Observability](#observability)
- [Testing & Evaluation](#testing--evaluation)
- [Using Saved Data](#using-saved-data)
- [Development](#development)
- [Database Reference](#database-reference)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A [Google AI Studio API key](https://aistudio.google.com/app/apikey) (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd storyland-ai
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up your API key**

   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your Google API key:
   ```env
   # Required
   GOOGLE_API_KEY=your-actual-api-key-here

   # Optional - Enable database persistence
   USE_DATABASE=false
   DATABASE_URL=sqlite:///storyland_sessions.db
   ```

## Quick Start

### Option 1: CLI (Production Ready)

The fastest way to get a travel itinerary:

```bash
# Basic usage
python main.py "Gone with the Wind"

# With author
python main.py "The Nightingale" --author "Kristin Hannah"

# With database persistence
python main.py "1984" --database

# With preferences
python main.py "Pride and Prejudice" --budget luxury --pace relaxed --museums

# Family trip
python main.py "Harry Potter" --with-kids --budget moderate

# Verbose logging (DEBUG level)
python main.py "1984" -v

# Full example
python main.py "The Great Gatsby" --author "F. Scott Fitzgerald" --user-id charlie --database --budget luxury
```

### Option 2: ADK Web UI (Development)

For interactive development and debugging:

```bash
python main.py --dev
```

This launches the ADK Web UI at http://localhost:8000 with full DEBUG logging.

### Option 3: Jupyter Notebook (Interactive Demo)

For exploration and experimentation:

```bash
# Start Jupyter
jupyter notebook

# Open the comprehensive showcase notebook:
# - storyland_showcase.ipynb (complete feature showcase with all capabilities)
```

**What's in the showcase notebook:**
- Multi-agent architecture (sequential, parallel, three-phase workflow)
- Custom tools (Google Books API, preferences)
- Sessions & state management (InMemory + SQLite)
- User preferences & personalization
- Context engineering & compaction
- Observability & structured logging
- Agent evaluation with rubrics
- Human-in-the-loop region selection
- Full end-to-end demo

### Option 4: Programmatic Usage

Import and use the modular components with the three-phase workflow:

```python
from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner

from services.session_service import create_session_service
from tools.google_books import google_books_tool
from agents.orchestrator import (
    create_metadata_stage,
    create_discovery_workflow,
    create_composition_workflow,
)

# Configure model
model = Gemini(model="gemini-2.0-flash-lite", api_key="your-key")
session_service = create_session_service(use_database=True)

# Phase 1: Extract book metadata
metadata_stage = create_metadata_stage(model, google_books_tool)
metadata_runner = Runner(agent=metadata_stage, app_name="storyland", session_service=session_service)

async with metadata_runner:
    async for event in metadata_runner.run_async(user_id="alice", session_id="session1", new_message=...):
        pass

# Get exact title/author from session state
session = await session_service.get_session(app_name="storyland", user_id="alice", session_id="session1")
book_metadata = session.state.get("book_metadata", {})
exact_title = book_metadata.get("book_title")
exact_author = book_metadata.get("author")

# Phase 2: Discovery - find locations and analyze regions
discovery_workflow = create_discovery_workflow(model, book_title=exact_title, author=exact_author)
discovery_runner = Runner(agent=discovery_workflow, app_name="storyland", session_service=session_service)

async with discovery_runner:
    async for event in discovery_runner.run_async(user_id="alice", session_id="session1", new_message=...):
        pass

# Get region analysis and let user select
session = await session_service.get_session(app_name="storyland", user_id="alice", session_id="session1")
region_analysis = session.state.get("region_analysis", {})
selected_regions = region_analysis.get("regions", [])[:1]  # Select first region

# Phase 3: Composition - create itinerary for selected region(s)
composition_workflow = create_composition_workflow(model)
composition_runner = Runner(agent=composition_workflow, app_name="storyland", session_service=session_service)

async with composition_runner:
    async for event in composition_runner.run_async(user_id="alice", session_id="session1", new_message=...):
        print(event)
```

### What You'll Get

The system generates a structured travel plan including:
- **Region options** - Choose which geographic region(s) to explore
- **Cities to visit** with suggested number of days
- **Landmarks and experiences** tied to the book's setting
- **Author-related sites** (birthplace, museums, etc.)
- **Practical details** (time of day, visiting tips)
- **Thematic connections** explaining why each location matters

### Human-in-the-Loop Region Selection

When a book spans multiple geographic regions (e.g., different countries or distant cities), the system presents region options:

```
======================================================================
TRAVEL REGION OPTIONS
======================================================================

Cities grouped by geographic proximity for practical travel planning.

[1] Southern England
    Cities: Bath, London
    Duration: ~4 days
    Travel: Train connections between cities (1-2 hours)
    Highlights: Regency-era architecture, Jane Austen Centre

[2] Peak District, England
    Cities: Bakewell, Chatsworth
    Duration: ~2 days
    Travel: All accessible by car within 30 minutes
    Highlights: Pemberley inspiration, scenic countryside

Enter region number(s) separated by commas (e.g., '1,2' for multiple regions)
Or press Enter to select all regions
Which region(s) would you like to explore? [1/2]:
```

This prevents impractical itineraries spanning disconnected locations (e.g., mixing East Coast and West Coast USA cities).

## Features

### 1. Sessions & State Management

- **In-Memory (Default)**: Fast, perfect for development and testing
- **SQLite-Backed**: Persistent sessions across restarts
- **Multi-scoped State**: Session, user, app, and temporary scopes
- **User Preferences**: Persist across sessions with `user:` prefix

```python
# Preferences persist automatically
state["user:preferences"] = {
    "prefers_museums": True,
    "budget": "moderate",
    "favorite_genres": ["classics"]
}
```

### 2. Context Engineering

- **Sliding Window**: Keep recent N events
- **Token Estimation**: Track conversation size
- **Compaction Detection**: Know when to reduce context
- **ADK Integration**: Use built-in `GetSessionConfig`

```python
from services.context_manager import ContextManager

context_manager = ContextManager(max_events=20)
if context_manager.should_compact(session.events):
    session.events = context_manager.limit_events(session.events)
```

### 3. Personalized Itineraries

The trip composer uses user preferences to tailor recommendations:

```python
# Set preferences in session state
state["user:preferences"] = {
    "budget": "luxury",           # budget, moderate, luxury
    "preferred_pace": "relaxed",  # relaxed, moderate, fast-paced
    "prefers_museums": True,
    "travels_with_kids": False,
    "dietary_restrictions": ["vegetarian"]
}
```

**How preferences affect itineraries:**

| Preference | Impact |
|------------|--------|
| `budget: "budget"` | Free museums, walking tours, affordable cafes |
| `budget: "luxury"` | Premium experiences, fine dining, private tours |
| `preferred_pace: "relaxed"` | 2-3 stops/day, longer breaks |
| `preferred_pace: "fast-paced"` | 5+ stops/day, efficient routing |
| `prefers_museums: true` | Prioritize literary museums, archives |
| `travels_with_kids: true` | Family-friendly activities, avoid long queues |

### 4. Multi-User Support

Each user has isolated sessions and preferences:

```bash
python main.py "Pride and Prejudice" --user-id alice --database
python main.py "Dune" --user-id bob --database
```

### 5. Automated Research

- Agents handle complexity of gathering scattered information
- **Parallel Processing**: Multiple discovery agents work simultaneously
- **Structured Output**: Clear, actionable travel plans

## Architecture

StoryLand AI uses a modular architecture with Google's Agent Development Kit (ADK).

```mermaid
graph LR
    subgraph "User Interface"
        CLI[CLI<br/>main.py]
        NB[Notebooks<br/>*.ipynb]
    end

    subgraph "Services"
        SS[SessionService<br/>InMemory / SQLite]
        CM[ContextManager<br/>Token Management]
    end

    subgraph "Tools"
        GB[google_books_tool<br/>📚 Book Search]
        GS[google_search<br/>🔍 Web Search]
        PT[get_preferences_tool<br/>👤 State Access]
    end

    subgraph "Agents"
        WF[Workflow<br/>SequentialAgent]
    end

    CLI --> SS
    CLI --> WF
    NB --> SS
    NB --> WF
    WF --> GB
    WF --> GS
    WF --> PT
    PT -.-> SS
    SS --> DB[(SQLite DB)]
```

### Project Structure

```
storyland-ai/
├── models/              # Pydantic data models
│   ├── book.py          # BookMetadata, BookContext, BookInfo
│   ├── discovery.py     # CityDiscovery, LandmarkDiscovery, AuthorSites,
│   │                    # RegionCity, RegionOption, RegionAnalysis
│   ├── itinerary.py     # TripItinerary, CityPlan, CityStop
│   └── preferences.py   # TravelPreferences
│
├── tools/               # External API integrations
│   ├── google_books.py  # Google Books search tool
│   └── preferences.py   # Session state preferences tool
│
├── agents/              # AI agent definitions
│   ├── book_metadata_agent.py    # Book metadata extraction
│   ├── book_context_agent.py     # Book setting research (accepts title/author)
│   ├── discovery_agents.py       # City/landmark/author discovery
│   ├── trip_composer_agent.py    # Itinerary composition
│   ├── reader_profile_agent.py   # Preferences-based personalization
│   ├── region_analyzer_agent.py  # Geographic region grouping (LLM-based)
│   ├── orchestrator.py           # Three-phase workflow coordination
│   │                             # - create_metadata_stage()
│   │                             # - create_discovery_workflow(title, author)
│   │                             # - create_composition_workflow()
│   │                             # - create_eval_workflow() (for ADK evals)
│   └── storyland/                # ADK Web UI agent
│       └── agent.py              # root_agent for adk web
│
├── services/            # Core services
│   ├── session_service.py   # Session management (InMemory/SQLite)
│   └── context_manager.py   # Context engineering
│
├── common/              # Shared utilities
│   ├── config.py        # Configuration management
│   └── logging.py       # Structured logging (structlog)
│
├── tests/               # Test suite
│   ├── conftest.py      # Shared pytest fixtures
│   ├── unit/            # Unit tests (no API calls)
│   │   ├── test_models.py    # Pydantic model tests
│   │   ├── test_tools.py     # Tool function tests
│   │   ├── test_agents.py    # Agent factory tests
│   │   └── test_services.py  # Service tests
│   └── evaluation/      # ADK evaluation config
│       └── eval_config.json  # Rubric definitions
│
├── main.py              # CLI entry point
├── *.ipynb              # Demo notebooks
├── pyproject.toml       # Dependencies & pytest config
└── .env.example         # Environment template
```

### Multi-Agent Workflow (Three-Phase Architecture)

The workflow uses a three-phase architecture with human-in-the-loop region selection:

```mermaid
flowchart TB
    subgraph Input
        CLI[CLI Args] --> |"--budget, --pace, etc"| State
        State[Session State<br/>user:preferences]
    end

    subgraph Phase1["Phase 1: Metadata Stage"]
        BM[book_metadata_pipeline<br/>📚 Google Books API] --> |state.book_metadata| Extract
        Extract[Extract exact<br/>title & author]
    end

    subgraph Phase2["Phase 2: Discovery Workflow"]
        direction TB

        BC[book_context_pipeline<br/>🔍 Uses exact title/author] --> |state.book_context| RP

        RP[reader_profile_agent<br/>👤 get_preferences_tool] --> |state.reader_profile| PD

        subgraph PD["ParallelAgent ⚡"]
            direction LR
            CP[city_pipeline<br/>🏙️]
            LP[landmark_pipeline<br/>🏛️]
            AP[author_pipeline<br/>✍️]
        end

        PD --> |discoveries| RA
        RA[region_analyzer<br/>🌍 Group by geography]
    end

    subgraph HITL["Human-in-the-Loop"]
        Select[User selects<br/>region(s) to explore]
    end

    subgraph Phase3["Phase 3: Composition Workflow"]
        TC[trip_composer<br/>🗺️ Personalized Itinerary]
    end

    Extract --> |"title, author"| BC
    State -.-> |ToolContext.state| RP
    RA --> |region_analysis| Select
    Select --> |selected_regions| TC
    TC --> Output[TripItinerary<br/>JSON Output]
```

**Why three phases?**
1. **Phase 1 (Metadata)**: Books with common titles (e.g., "The Nightingale") can match multiple books. Phase 1 resolves the exact book.
2. **Phase 2 (Discovery)**: Finds all relevant locations and groups them into practical travel regions using LLM-based geographic analysis.
3. **Human Selection**: User chooses which region(s) to explore, preventing impractical itineraries spanning distant locations.
4. **Phase 3 (Composition)**: Creates a detailed itinerary for only the selected region(s).

### Data Flow

```mermaid
sequenceDiagram
    participant CLI as main.py
    participant SS as SessionService
    participant P1 as Phase 1: metadata_stage
    participant P2 as Phase 2: discovery_workflow
    participant RA as region_analyzer
    participant User as User
    participant P3 as Phase 3: composition_workflow
    participant TC as trip_composer

    CLI->>SS: create_session(state={user:preferences})

    Note over CLI,P1: Phase 1: Extract exact metadata
    CLI->>P1: run_async("Find book metadata...")
    P1->>P1: book_metadata_pipeline
    P1-->>SS: state["book_metadata"] = {title, author}
    CLI->>SS: get_session() → extract exact_title, exact_author

    Note over CLI,P2: Phase 2: Discovery + Region Analysis
    CLI->>P2: create_discovery_workflow(exact_title, exact_author)
    CLI->>P2: run_async(prompt)

    P2->>P2: book_context_pipeline (searches: "title author setting")
    P2->>P2: reader_profile_agent
    P2->>P2: parallel_discovery (city, landmark, author)
    P2->>RA: Analyze geographic regions
    RA-->>SS: state["region_analysis"] = {regions: [...]}

    Note over CLI,User: Human-in-the-Loop
    CLI->>User: Display region options
    User-->>CLI: Select region(s)

    Note over CLI,P3: Phase 3: Create itinerary for selected region(s)
    CLI->>P3: create_composition_workflow()
    CLI->>P3: run_async(prompt with selected_regions)

    P3->>TC: Execute with selected_regions in prompt
    TC-->>P3: TripItinerary (for selected region only!)

    P3-->>CLI: Final response
```

The three-phase approach ensures:
1. Accurate book identification (Phase 1)
2. Comprehensive location discovery with geographic grouping (Phase 2)
3. Practical itineraries focused on user-selected regions (Phase 3)

### Two-Stage Agent Pattern

Each pipeline uses a two-stage approach:
1. **Stage 1 (Researcher)**: Uses tools to gather data
2. **Stage 2 (Formatter)**: Validates with Pydantic schema

This pattern ensures type safety and clean data flow between agents.

### Prompt Engineering

Agent prompts include several reliability improvements:

**Error Handling Guidance:**
```
If the tool returns an error, report it clearly and explain what went wrong
If no results found, acknowledge this and suggest the user verify the title/author
```

**Anti-Hallucination Instructions:**
```
IMPORTANT: If the research found no cities, return an empty list - do not hallucinate.
Include only cities actually mentioned in the research.
```

**Disambiguation (Book Context):**
```
BOOK: "{book_title}" by {author}

Search queries to use:
- "{book_title} {author} setting location"
```

### Core Components

#### 1. Book Metadata Agent
- Extracts book information from Google Books API
- Uses both `title` and `author` parameters when available for accurate matching
- Includes error handling guidance for failed searches
- Validates with `BookMetadata` Pydantic model
- Saves to `state["book_metadata"]`

#### 2. Book Context Agent
- Receives exact `book_title` and `author` from Phase 1
- Searches with precise queries (e.g., "The Nightingale Kristin Hannah setting")
- Researches setting, time period, and themes
- Uses Google Search for deep context
- Validates with `BookContext` Pydantic model

#### 3. Discovery Agents (Parallel)
- **City Agent**: Finds cities to visit
- **Landmark Agent**: Discovers specific places
- **Author Agent**: Locates author-related sites
- All run in parallel for efficiency

#### 4. Region Analyzer Agent
- Analyzes discovered cities and groups them into practical travel regions
- Uses LLM world knowledge for geographic proximity analysis
- Rules for grouping:
  - Same country, close proximity (~500km) → ONE region
  - Cross-border accessible (train/short flight) → can be ONE region
  - Large countries split into regions (USA: East/West Coast, etc.)
  - Never combines cities requiring intercontinental flights
- Validates with `RegionAnalysis` Pydantic model
- Saves to `state["region_analysis"]`

#### 5. Trip Composer Agent
- Synthesizes discoveries into coherent itinerary for **selected region(s) only**
- Groups by city, suggests timing
- **Uses user preferences** for personalization:
  - Budget level (budget/moderate/luxury)
  - Pace (relaxed/moderate/fast-paced)
  - Museum preference
  - Traveling with kids
  - Dietary restrictions
- Validates with `TripItinerary` Pydantic model

#### 6. Reader Profile Agent
- Uses `get_preferences_tool` to access `user:preferences` from session state
- Tool reads from `ToolContext.state` (ADK's mechanism for state access)
- Summarizes preferences for trip composer
- Provides personalization context for itinerary generation

## CLI Usage

### Basic Commands

```bash
# Simple query (default user: user1)
python main.py "Gone with the Wind"

# With author
python main.py "The Nightingale" --author "Kristin Hannah"

# Custom user
python main.py "1984" --user-id alice
```

### Advanced Usage

```bash
# Enable SQLite persistence
python main.py "Pride and Prejudice" --database

# With preferences (via CLI flags)
python main.py "The Great Gatsby" --budget luxury --pace relaxed

# Family trip with database
python main.py "Harry Potter" --database --with-kids --budget moderate --museums
```

### Multi-User Support

```bash
# User "alice" explores Pride and Prejudice
python main.py "Pride and Prejudice" --user-id alice --database

# User "bob" explores Dune (completely separate data)
python main.py "Dune" --user-id bob --database

# Alice explores another book (her preferences persist!)
python main.py "Emma" --user-id alice --database
```

**Check user data:**

```bash
# See all users
sqlite3 storyland_sessions.db "SELECT DISTINCT user_id FROM sessions;"

# Count sessions per user
sqlite3 storyland_sessions.db "
SELECT user_id, COUNT(*) as session_count
FROM sessions
GROUP BY user_id;"

# View specific user's sessions
sqlite3 storyland_sessions.db "
SELECT id,
       json_extract(state, '$.book_title') as book,
       create_time
FROM sessions
WHERE user_id = 'alice';"
```

### Help

```bash
python main.py --help
```

## Observability

StoryLand AI provides two modes for observability:

### Development: ADK Web UI

For interactive development and debugging, use the built-in ADK Web UI:

```bash
# Launch via CLI
python main.py --dev

# Or directly
.venv/bin/adk web agents/
```

The Web UI provides:
- Visual agent execution flow
- Request/response inspection
- Built-in DEBUG logging
- Interactive testing

> **Note:** Plugins are NOT supported in ADK web mode.

### Production: ADK LoggingPlugin

For production runs, ADK's built-in `LoggingPlugin` automatically logs agent, tool, and model activity:

```bash
# Normal run (INFO level logging)
python main.py "1984" --author "George Orwell"

# Verbose mode (DEBUG level)
python main.py "1984" -v
```

**Sample output:**
```
🤖 AGENT STARTING
   Agent Name: workflow
   Invocation ID: e54b...
🤖 AGENT STARTING
   Agent Name: book_metadata_pipeline
🔧 TOOL STARTING
   Tool Name: search_book
🔧 TOOL COMPLETED
✅ AGENT COMPLETED
   Agent Name: book_metadata_pipeline
```

**What's logged:**
| Event | Description |
|-------|-------------|
| `🤖 AGENT STARTING/COMPLETED` | Agent execution |
| `🔧 TOOL STARTING/COMPLETED` | Tool invocations |
| `🧠 MODEL REQUEST/RESPONSE` | LLM calls |
| `❌ ERROR` | Failures |

### Configuration

```env
# .env
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR
ENABLE_ADK_DEBUG=false         # Enable DEBUG for ADK internal logs
```

| Mode | Logging | Plugin | Use Case |
|------|---------|--------|----------|
| `--dev` | DEBUG (ADK Web) | None | Development, debugging |
| Default | INFO | LoggingPlugin | Production |
| `-v` | DEBUG | LoggingPlugin | Troubleshooting |

## Testing & Evaluation

StoryLand AI includes comprehensive testing with pytest unit tests and ADK evaluation framework.

### Unit Tests (pytest)

Run unit tests locally without any API calls:

```bash
# Run all unit tests
.venv/bin/pytest tests/unit/ -v

# Run with coverage
.venv/bin/pytest tests/unit/ --cov=. --cov-report=term-missing

# Run specific test file
.venv/bin/pytest tests/unit/test_models.py -v

# Run specific test class
.venv/bin/pytest tests/unit/test_agents.py::TestEvalWorkflow -v
```

**Test coverage (125 tests total):**
| Module | Tests | Description |
|--------|-------|-------------|
| `test_models.py` | 46 | Pydantic model validation (incl. RegionAnalysis) |
| `test_tools.py` | 16 | Google Books, preferences tools |
| `test_agents.py` | 41 | Agent factory functions (three-phase & eval workflows) |
| `test_services.py` | 16 | Session service, context manager |
| `test_workflow_timeout.py` | 6 | Workflow timeout behavior |

### ADK Evaluation (CLI)

Run agent evaluation with rubric-based scoring:

```bash
# Run single test scenario (fast)
.venv/bin/adk eval agents/storyland single_test \
  --config_file_path tests/evaluation/eval_config.json \
  --print_detailed_results
```

**Note:** The eval workflow (`create_eval_workflow`) includes region analysis but auto-selects all regions since human-in-the-loop interaction is not possible in automated evaluations. The workflow still validates that region grouping works correctly.

**Evaluation rubrics:**
| Rubric | Description |
|--------|-------------|
| `book_relevance` | Locations connected to book's settings, themes, characters |
| `preference_adherence` | Respects user's budget, pace, accessibility preferences |
| `completeness` | Comprehensive itinerary with cities, landmarks, author sites |
| `actionability` | Practical details: times, transport, booking tips |
| `geographical_accuracy` | Real places correctly associated with countries |
| `engagement` | Engaging descriptions capturing literary spirit |

**Configuration files:**
- `tests/evaluation/eval_config.json` - Rubric definitions and judge model settings
- `agents/storyland/single_test.evalset.json` - Test scenario for Pride and Prejudice

### ADK Web UI Evaluation

For interactive evaluation with visual results:

```bash
# Launch ADK Web UI
.venv/bin/adk web agents/

# Open browser to http://localhost:8000
```

**Using the Eval tab:**

1. Navigate to the **Eval** tab in the Web UI
2. Select an eval set from the dropdown (e.g., `single_test`)
3. Click **Run Eval** to execute all test cases
4. View results with:
   - Pass/fail status per test case
   - Rubric scores with rationale
   - Agent response inspection
   - Turn-by-turn execution trace

**Creating custom eval sets:**

Create `.evalset.json` files in `agents/storyland/`:

```json
{
  "eval_set_id": "my_custom_eval",
  "name": "My Custom Evaluation",
  "description": "Test specific scenarios",
  "eval_cases": [
    {
      "eval_id": "test_case_1",
      "conversation": [
        {
          "invocation_id": "turn_1",
          "user_content": {
            "parts": [{"text": "Create a travel itinerary for..."}]
          }
        }
      ],
      "session_input": {
        "app_name": "storyland",
        "user_id": "eval_user",
        "state": {
          "user:preferences": {
            "budget": "moderate",
            "preferred_pace": "relaxed"
          }
        }
      }
    }
  ]
}
```

### Rate Limits

The Gemini free tier has limits (15 RPM, 200 requests/day). For evaluation:
- Use `gemini-2.5-flash-lite` for judge model (lower cost)
- Run single tests during development
- Run full suite when quota is available

## Using Saved Data in Future Runs

When you enable database persistence (`--database` or `USE_DATABASE=true`), all data is saved and can be reused.

### 1. User Preferences Persist

```python
# First session - set preferences
await session_service.create_session(
    app_name="storyland",
    user_id="alice",
    session_id="session-1",
    state={
        "book_title": "Pride and Prejudice",
        "user:preferences": {
            "prefers_museums": True,
            "budget": "moderate"
        }
    }
)

# Later session - preferences auto-loaded!
new_session = await session_service.create_session(
    app_name="storyland",
    user_id="alice",  # Same user
    session_id="session-2",
    state={"book_title": "Emma"}
)

prefs = new_session.state["user:preferences"]  # Available!
```

### 2. Resume Previous Sessions

```python
# Load existing session
session = await session_service.get_session(
    app_name="storyland",
    user_id="alice",
    session_id="previous-session-id"
)

# Access previous itinerary
itinerary = session.state.get("final_itinerary")
```

### 3. Query User History

```python
import sqlite3

def get_user_books(user_id: str):
    conn = sqlite3.connect('storyland_sessions.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT json_extract(state, '$.book_title') as book,
               create_time
        FROM sessions
        WHERE user_id = ?
        ORDER BY create_time DESC
    """, (user_id,))
    return cursor.fetchall()
```

### 4. Practical Demo

See **[sessions_memory_demo.ipynb](sessions_memory_demo.ipynb)** for 11 complete scenarios covering sessions, preferences, and context management.

## Development

### Adding New Agents

1. Create agent file in `agents/`:

```python
# agents/my_new_agent.py
from google.adk.agents import LlmAgent

def create_my_agent(model):
    return LlmAgent(
        name="my_agent",
        model=model,
        instruction="Do something..."
    )
```

2. Import in orchestrator and add to workflow

### Adding New Tools

1. Create tool file in `tools/`:

```python
# tools/my_tool.py
from google.adk.tools import FunctionTool

def my_function(query: str) -> str:
    return result

my_tool = FunctionTool(my_function)
```

2. Use in agents

### Adding New Models

```python
# models/my_model.py
from pydantic import BaseModel, Field

class MyModel(BaseModel):
    field1: str = Field(description="Description")
    field2: int = Field(description="Description")
```

## Database Reference

### Schema

ADK's `DatabaseSessionService` creates:

```sql
CREATE TABLE sessions (
    app_name VARCHAR(128) NOT NULL,
    user_id VARCHAR(128) NOT NULL,
    id VARCHAR(128) NOT NULL,
    state TEXT NOT NULL,
    create_time DATETIME NOT NULL,
    update_time DATETIME NOT NULL,
    PRIMARY KEY (app_name, user_id, id)
)
```

**Important:** Column names are:
- `id` (not `session_id`)
- `create_time` (not `created_at`)
- `update_time` (not `updated_at`)

See **[DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)** for complete reference.

### Inspecting Database

```bash
# Open database
sqlite3 storyland_sessions.db

# Inside sqlite3:
.tables                    # List tables
.schema                    # Show structures
.mode column              # Better formatting
.headers on

# View sessions
SELECT * FROM sessions;

# View latest
SELECT * FROM sessions ORDER BY create_time DESC LIMIT 1;

# Exit
.quit
```

**Quick one-liners:**

```bash
# List tables
sqlite3 storyland_sessions.db ".tables"

# Count sessions
sqlite3 storyland_sessions.db "SELECT COUNT(*) FROM sessions;"

# Latest session
sqlite3 -column -header storyland_sessions.db \
  "SELECT id, user_id, create_time FROM sessions ORDER BY create_time DESC LIMIT 1;"
```

## Troubleshooting

### Rate Limits (429 Errors)

The workflow makes ~11 API calls per book, which can hit free tier limits (15 RPM).

**Solution:** The retry logic will handle it automatically. Wait ~60 seconds between books.

### Database Issues

```bash
# Delete database to start fresh
rm storyland_sessions.db

# Check database exists
ls -lh *.db
```

### Import Errors

```bash
# Ensure you're in project root
pwd

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"
```

### API Key Issues

```bash
# Check API key is set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OK' if os.getenv('GOOGLE_API_KEY') else 'MISSING')"
```

## Configuration

All configuration via environment variables in `.env`:

```env
# Required
GOOGLE_API_KEY=your-google-ai-api-key-here

# Database (optional)
USE_DATABASE=false
DATABASE_URL=sqlite:///storyland_sessions.db

# Session (optional)
SESSION_MAX_EVENTS=20

# Context (optional)
MAX_CONTEXT_TOKENS=30000

# Model (optional)
MODEL_NAME=gemini-2.0-flash-lite

# Workflow execution (optional)
WORKFLOW_TIMEOUT=300    # Max seconds for workflow (default: 300)
AGENT_TIMEOUT=60        # Max seconds per agent (default: 60)

# Logging (optional)
LOG_LEVEL=INFO
ENABLE_ADK_DEBUG=false  # Enable DEBUG for ADK internal logs
```

## Technology Stack

- **Framework:** Google Agent Development Kit (ADK)
- **LLM:** Google Gemini (gemini-2.0-flash-lite)
- **APIs:** Google Books API, Google Search
- **Database:** SQLite (via ADK's DatabaseSessionService)
- **Data Validation:** Pydantic models
- **Agent Patterns:** Sequential and parallel orchestration

## Why StoryLand AI?

**For Readers:**
- Turn literary inspiration into real travel experiences
- Save hours of research and planning
- Discover places you'd never find through conventional searches
- Get personalized recommendations that match your travel style

**For Developers:**
- Demonstrates real-world multi-agent coordination
- Shows parallel vs. sequential agent patterns
- Implements session-based personalization via custom tools
- Shows how tools access session state via `ToolContext`
- Modular architecture with clean separation of concerns
- Production-ready with database persistence and error handling

## Vision

A great book doesn't just end—it opens a door. StoryLand AI helps readers step through it.

We believe that every story deserves to be experienced beyond the page, and every reader should be able to walk through the worlds they love without the friction of scattered information and endless research.

**A single prompt can describe a world—but only agents can build a bridge between that world and real places someone can actually visit.**

## Additional Resources

- **[DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)** - Complete database reference
- **[sessions_memory_demo.ipynb](sessions_memory_demo.ipynb)** - 11 demo scenarios covering sessions, preferences, and context management
- **[tests/evaluation/eval_config.json](tests/evaluation/eval_config.json)** - Evaluation rubric configuration
- **[agents/storyland/single_test.evalset.json](agents/storyland/single_test.evalset.json)** - Evaluation test scenario

---

## License

[Add your license information here]

## Contact

[Add your contact information here]

---

*Built with Google Agent Development Kit*
