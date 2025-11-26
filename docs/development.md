# Development Guide

## Project Structure

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

## Adding New Agents

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

## Adding New Tools

1. Create tool file in `tools/`:

```python
# tools/my_tool.py
from google.adk.tools import FunctionTool

def my_function(query: str) -> str:
    return result

my_tool = FunctionTool(my_function)
```

2. Use in agents

## Adding New Models

```python
# models/my_model.py
from pydantic import BaseModel, Field

class MyModel(BaseModel):
    field1: str = Field(description="Description")
    field2: int = Field(description="Description")
```

## Two-Stage Agent Pattern

Each pipeline uses a two-stage approach:
1. **Stage 1 (Researcher)**: Uses tools to gather data
2. **Stage 2 (Formatter)**: Validates with Pydantic schema

This pattern ensures type safety and clean data flow between agents.

## Prompt Engineering

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

## Core Components

### 1. Book Metadata Agent
- Extracts book information from Google Books API
- Uses both `title` and `author` parameters when available for accurate matching
- Includes error handling guidance for failed searches
- Validates with `BookMetadata` Pydantic model
- Saves to `state["book_metadata"]`

### 2. Book Context Agent
- Receives exact `book_title` and `author` from Phase 1
- Searches with precise queries (e.g., "The Nightingale Kristin Hannah setting")
- Researches setting, time period, and themes
- Uses Google Search for deep context
- Validates with `BookContext` Pydantic model

### 3. Discovery Agents (Parallel)
- **City Agent**: Finds cities to visit
- **Landmark Agent**: Discovers specific places
- **Author Agent**: Locates author-related sites
- All run in parallel for efficiency

### 4. Region Analyzer Agent
- Analyzes discovered cities and groups them into practical travel regions
- Uses LLM world knowledge for geographic proximity analysis
- Rules for grouping:
  - Same country, close proximity (~500km) → ONE region
  - Cross-border accessible (train/short flight) → can be ONE region
  - Large countries split into regions (USA: East/West Coast, etc.)
  - Never combines cities requiring intercontinental flights
- Validates with `RegionAnalysis` Pydantic model
- Saves to `state["region_analysis"]`

### 5. Trip Composer Agent
- Synthesizes discoveries into coherent itinerary for **selected region(s) only**
- Groups by city, suggests timing
- **Uses user preferences** for personalization:
  - Budget level (budget/moderate/luxury)
  - Pace (relaxed/moderate/fast-paced)
  - Museum preference
  - Traveling with kids
  - Dietary restrictions
- Validates with `TripItinerary` Pydantic model

### 6. Reader Profile Agent
- Uses `get_preferences_tool` to access `user:preferences` from session state
- Tool reads from `ToolContext.state` (ADK's mechanism for state access)
- Summarizes preferences for trip composer
- Provides personalization context for itinerary generation

## Technology Stack

- **Framework:** Google Agent Development Kit (ADK)
- **LLM:** Google Gemini (gemini-2.0-flash-lite)
- **APIs:** Google Books API, Google Search
- **Database:** SQLite (via ADK's DatabaseSessionService)
- **Data Validation:** Pydantic models
- **Agent Patterns:** Sequential and parallel orchestration
