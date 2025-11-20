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
   pip install -r requirements.txt
   ```

4. **Set up your API key**

   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=your-actual-api-key-here
   ```

### Quick Start

**Option 1: Run the Jupyter Notebook (Recommended)**

The easiest way to get started is with the demo notebook:

```bash
jupyter notebook storyland_ai_demo.ipynb
```

This notebook demonstrates:
- Complete multi-agent workflow (sequential + parallel agents)
- Book metadata extraction using Google Books API
- Location discovery with parallel agent execution
- Travel itinerary composition
- Full execution tracing and observability

Simply configure your book in the notebook:
```python
BOOK_TITLE = "nightingale"  # Change to any book
AUTHOR = ""                  # Optional
```

Then run all cells to generate your literary travel itinerary!

**Option 2: Run the Agent System Programmatically**

```python
from google.adk.runners import Runner
from agents.storyland_orchestrator_agent import create_orchestrator

# Create the orchestrator
orchestrator = create_orchestrator()

# Create runner
runner = Runner(agent=orchestrator, app_name="storyland")

# Run with your book
response = runner.run(
    user_id="user1",
    session_id="session1",
    new_message="Create a travel itinerary for 'The Nightingale'"
)
```

### What You'll Get

The system generates a structured travel plan including:
- **Cities to visit** with suggested number of days
- **Landmarks and experiences** tied to the book's setting
- **Author-related sites** (birthplace, museums, etc.)
- **Practical details** (time of day, visiting tips)
- **Thematic connections** explaining why each location matters

**Example Output Structure:**
```json
{
  "itinerary": {
    "cities": [
      {
        "name": "Paris",
        "country": "France",
        "days_suggested": 2,
        "stops": [
          {
            "name": "Musée de l'Homme",
            "type": "museum",
            "reason": "Featured in The Nightingale as resistance headquarters",
            "time_of_day": "morning"
          }
        ]
      }
    ],
    "summary_text": "A journey through WWII France..."
  }
}
```

## Architecture

StoryLand AI is built on Google's Agent Development Kit (ADK) and uses the **`storyland_orchestrator_agent`** as the central coordinator of a specialized agent ecosystem.

### Core Components

#### 1. **Book Context Specialist** (`book_info_agent`)

**Purpose:** Understand the book at a high level—setting, key locations, themes

**Tools:**
- Google Books API (metadata, descriptions, categories)
- Google Search (setting queries, location research)

**Output:** Structured data about primary locations, time period, and author-related places

**Pattern:** Sequential agent (runs early, provides foundation for other agents)

---

#### 2. **Location Discovery Team** (Parallel Agents)

Once book context is established, these agents run **in parallel** to discover different aspects:

##### **City & Neighborhood Agent** (`city_discovery_agent`)
- Identifies real cities, districts, and neighborhoods
- Finds filming locations and setting-aligned places
- Returns candidate locations with reference URLs

##### **Landmarks & Experiences Agent** (`landmark_discovery_agent`)
- Focuses on museums, walking routes, viewpoints
- Searches for book-themed experiences (literary walks, tours)
- Identifies places travelers can actually visit

##### **Author & History Agent** (`author_context_agent`)
- Locates author's hometown, houses, statues
- Finds literary sites and historical context
- Adds depth: "where the story came from," not just "where it happens"

**Pattern:** Parallel execution reduces latency and demonstrates concurrent agent workflows

---

#### 3. **Trip Designer** (`trip_composer_agent`)

**Purpose:** Transform research into a human-readable travel plan

**Capabilities:**
- Groups places by region or city
- Suggests 1-day or weekend itineraries
- Explains why each location matters for the book
- Fills in practical details (hours, visiting order)

**Pattern:** Sequential agent (runs near end of pipeline, consumes discovery outputs)

---

#### 4. **Preferences & Memory Agent** (`reader_profile_agent`)

**Purpose:** Personalize experiences across multiple uses

**Manages:**
- Session-level state (current book, in-progress results)
- Long-term preferences ("travels with kids", "likes Europe", "prefers museums")

**Implementation:** Session service / memory bank for adaptive planning

---

### Essential Tools

All agents share access to:

- **Google Search** – Open-web information about settings, locations, travel details
- **Google Books API** – Official metadata, descriptions, and categories
- **Evaluation UI** – Trace agent calls, inspect outputs, compare configurations

## Features

- **Automated Research:** Agents handle the complexity of gathering scattered information
- **Parallel Processing:** Multiple discovery agents work simultaneously for faster results
- **Personalization:** Remembers your preferences across books
- **Structured Output:** Clear, actionable travel plans, not random search results
- **Observability:** Full tracing and evaluation capabilities for continuous improvement

## How It Works

1. **Input:** User provides a book title
2. **Book Analysis:** `book_info_agent` extracts setting, themes, and context
3. **Parallel Discovery:** Three agents simultaneously research cities, landmarks, and author connections
4. **Plan Composition:** `trip_composer_agent` merges findings into a coherent itinerary
5. **Personalization:** `reader_profile_agent` adapts the plan to user preferences
6. **Output:** A complete, book-themed travel inspiration plan

## Technology Stack

- **Framework:** Google Agent Development Kit (ADK)
- **LLM:** Google Gemini
- **APIs:** Google Books API, Google Search
- **Agent Patterns:** Sequential and parallel agent orchestration
- **Evaluation:** Google evaluation UI and logging tools

## Project Structure

```
storyland-ai/
├── agents/
│   ├── storyland_orchestrator_agent.py  # Central coordinator
│   ├── book_info_agent.py               # Book context specialist
│   ├── city_discovery_agent.py          # City & neighborhood discovery
│   ├── landmark_discovery_agent.py      # Landmarks & experiences
│   ├── author_context_agent.py          # Author & history research
│   ├── trip_composer_agent.py           # Travel plan composer
│   └── reader_profile_agent.py          # Preferences & memory
├── tools/
│   ├── google_search.py                 # Google Search integration
│   └── google_books.py                  # Google Books API wrapper
├── common/
│   └── logging.py                       # Logging utilities
└── README.md
```

## Why StoryLand AI?

**For Readers:**
- Turn literary inspiration into real travel experiences
- Save hours of research and planning
- Discover places you'd never find through conventional searches
- Get personalized recommendations that match your travel style

**For Developers:**
- Demonstrates real-world multi-agent coordination
- Shows parallel vs. sequential agent patterns
- Implements memory and personalization in agent systems
- Built with production-ready evaluation and observability

## Vision

A great book doesn't just end—it opens a door. StoryLand AI helps readers step through it.

We believe that every story deserves to be experienced beyond the page, and every reader should be able to walk through the worlds they love without the friction of scattered information and endless research.

**A single prompt can describe a world—but only agents can build a bridge between that world and real places someone can actually visit.**

---

## License

[Add your license information here]

## Contact

[Add your contact information here]

---

*Built with Google Agent Development Kit*
