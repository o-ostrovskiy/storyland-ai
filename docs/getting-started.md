# Getting Started

## Prerequisites

- Python 3.10 or higher
- A [Google AI Studio API key](https://aistudio.google.com/app/apikey) (free tier available)

## Installation

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

## What You'll Get

The system generates a structured travel plan including:
- **Region options** - Choose which geographic region(s) to explore
- **Cities to visit** with suggested number of days
- **Landmarks and experiences** tied to the book's setting
- **Author-related sites** (birthplace, museums, etc.)
- **Practical details** (time of day, visiting tips)
- **Thematic connections** explaining why each location matters

## Human-in-the-Loop Region Selection

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
