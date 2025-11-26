# StoryLand AI: Transforming Books into Travel Adventures

## Problem Statement

When readers finish a beloved book, they want to stay in that world longer—walk the same streets, feel the atmosphere, visit places that inspired the author. But turning this impulse into reality is frustratingly difficult.

The current process requires endless Google searches across scattered travel blogs, tourist sites, historical forums, and film location databases. Information is contradictory, incomplete, and buried under commercial noise. A reader researching Pride and Prejudice might find a blog about Chawton cottage, Wikipedia on filming locations, forum debates about whether Pemberley was Chatsworth or Lyme Park, and generic Bath tourism sites—but nothing connects these into a coherent journey. The result: too many tabs, too much noise, too many missing pieces. The initial inspiration dies under research friction.

Literary tourism is booming (Harry Potter and Game of Thrones locations see massive visitor increases), yet planning remains painfully manual. Literature makes places meaningful in ways conventional tourism cannot—a street corner becomes a portal into narrative worlds. But readers lack tools to bridge the gap between page and passport.

**The core problem is information architecture**: the knowledge exists but is fragmented across incompatible sources, requiring expertise in both literature and travel planning that few possess.

## Why Agents?

A single LLM prompt cannot solve this problem.

Creating meaningful travel plans from books requires **coordination across multiple domains**: literary analysis (setting, themes, historical context), geographic research (real cities and landmarks), author biography (birthplaces, museums, archives), and travel logistics (grouping locations into regions, estimating times, personalizing). It also needs **parallel processing**—researching cities, landmarks, and author sites simultaneously rather than sequentially.

Agents enable **intelligent specialization**:
- **Book metadata agent** queries Google Books API and handles ambiguous titles (multiple books called "The Nightingale")
- **City discovery agent** searches for cities with literary connections
- **Landmark agent** finds museums, cafes, filming locations
- **Author agent** locates biographical sites
- **Region analyzer agent** applies geographic intelligence to group cities into practical travel regions
- **Trip composer agent** synthesizes everything into personalized itineraries

Each agent uses specialized tools (Google Books API, Google Search, session state access) and validates output against Pydantic schemas for type safety. This separation produces better results than any monolithic prompt.

Most critically, agents enable **stateful workflows with human-in-the-loop interaction**. StoryLand AI's three-phase architecture builds state progressively, letting users select geographic regions between phases—preventing impractical itineraries mixing continents. This orchestration is impossible with a single LLM call.

**Agents are the right solution because the problem is inherently multi-faceted**: research, synthesis, geographic reasoning, personalization, and human judgment working together.

## What I Created: Architecture Overview

StoryLand AI is a production-ready multi-agent system built on Google's Agent Development Kit (ADK) with a **three-phase workflow**:

### Phase 1: Metadata Extraction
Resolves book ambiguity by querying Google Books API for exact metadata—title, author, publication date, description. This prevents downstream confusion (distinguishing Kristin Hannah's "The Nightingale" from the 1950s French novel).

**Architecture**: Sequential pipeline where `book_metadata_researcher` calls Google Books tool and `book_metadata_formatter` validates against `BookMetadata` Pydantic schema.

### Phase 2: Discovery & Region Analysis
With exact book info, the system executes comprehensive discovery using **parallel agents**:

```
SequentialAgent (discovery_workflow)
├─ book_context_pipeline → researches setting, themes, period
├─ reader_profile_agent → loads user preferences
├─ ParallelAgent (concurrent ⚡)
│  ├─ city_pipeline → finds cities
│  ├─ landmark_pipeline → discovers landmarks
│  └─ author_pipeline → locates author sites
└─ region_analyzer_agent → groups cities geographically
```

The **parallel agent** runs three discovery agents simultaneously, dramatically reducing latency. Each follows a two-stage pattern: researcher gathers data via Google Search, formatter validates with Pydantic models.

The **region analyzer agent** demonstrates true intelligence, applying geographic reasoning to group cities into practical regions. For Pride and Prejudice, it identifies five regions:
- Hampshire (Chawton, Winchester) - Austen's heartlands, 3 days, 45-min drives
- Peak District (Chatsworth) - Pemberley inspiration
- Southern England (Bath, Lacock) - Regency architecture
- London & Kent - filming locations
- East Midlands (Stamford) - Rosings Park

Each region includes estimated days, travel notes, and highlights, preventing impractical East-Coast-to-West-Coast itineraries.

### Phase 3: Composition
After user selects region(s), `trip_composer_agent` creates personalized itineraries using the new `get_discovery_data_tool` for efficient structured data access. It applies user preferences (budget, pace, museum interest, kids) and generates:
- Cities with suggested days
- Specific stops (landmarks, museums, cafes)
- Time recommendations (morning/afternoon/full day)
- Thematic connections explaining significance
- Practical notes (fees, booking, accessibility)

Output validates against `TripItinerary` Pydantic model.

### Data Flow & State Management
The system uses ADK's `DatabaseSessionService` backed by SQLite. Each agent saves results to session state:
- `state["book_metadata"]` → exact book info
- `state["book_context"]` → setting and themes
- `state["city_discovery"]`, `["landmark_discovery"]`, `["author_sites"]` → discoveries
- `state["region_analysis"]` → geographic grouping
- `state["user:preferences"]` → **persists across sessions** (user-scoped)
- `state["final_itinerary"]` → complete plan

The `user:` prefix enables true personalization—preferences persist across all user sessions.

### Context Engineering
`ContextManager` monitors conversation size, estimates tokens, and applies sliding window compaction when needed, keeping recent events while preserving system prompts.

## Demo: How It Works

Running `python main.py "Pride and Prejudice" --author "Jane Austen" --budget moderate --pace relaxed --museums`:

**Phase 1** (5 sec): Google Books API confirms exact book—Pride and Prejudice by Jane Austen, 1813, Fiction/Romance.

**Phase 2** (30 sec): Parallel discovery finds 12 cities, 15 landmarks, 8 author sites. Book context identifies Hertfordshire/Derbyshire Regency setting. Reader profile loads preferences. Region analyzer creates 5 geographic regions.

**Human Selection** (30 sec): User views region options and selects "Hampshire - Jane Austen's Heartlands" (3 days, all within 45 min by car).

**Phase 3** (15 sec): Trip composer generates detailed 3-day itinerary:
- **Chawton**: Jane Austen's House Museum (full day), Chawton House, St. Nicholas Church
- **Steventon**: Birthplace site, Steventon Rectory location
- **Winchester**: Cathedral (her grave), historic city exploration

Each stop explains significance, notes admission fees (moderate budget), and suggests relaxed timing (2-3 stops/day, museum-focused).

**Total runtime**: ~50 seconds for a print-ready, personalized plan that would take hours manually.

## The Build: Technologies & Tools

**Framework**: Google Agent Development Kit (ADK) - orchestration, session management, tool integration, observability

**Model**: Google Gemini 2.5 Flash - fast, cost-effective, strong geographic reasoning

**Agent Patterns**:
- `SequentialAgent` - ordered workflows (metadata → discovery → composition)
- `ParallelAgent` - concurrent operations (city + landmark + author research)
- `LlmAgent` - individual agents with tools and schemas

**Tools**:
- Custom `google_books_tool` - Google Books API with Pydantic validation
- Built-in `google_search` - web research
- Custom `get_preferences_tool` - reads preferences from `ToolContext.state`
- Custom `get_discovery_data_tool` - efficient structured data access from session state

**Data Validation**: Nine Pydantic models ensure type-safe communication: `BookMetadata`, `BookContext`, `CityDiscovery`, `LandmarkDiscovery`, `AuthorSites`, `RegionAnalysis`, `TripItinerary`, `CityPlan`, `CityStop`, `TravelPreferences`.

**State Management**: ADK's `DatabaseSessionService` with SQLite—sessions survive restarts, user preferences persist, multi-user isolation.

**Observability**: Three layers—structured logging (`structlog`), ADK's `LoggingPlugin`, and custom event monitoring via `runner.run_async()`.

**Testing**: 125 unit tests (pytest) plus ADK evaluation framework with custom rubrics (book relevance, preference adherence, completeness, actionability, geographical accuracy, engagement).

**Development Tools**: CLI (`python main.py`), Jupyter notebooks (`storyland_showcase.ipynb`), ADK Web UI (`adk web agents/`), Makefile.

**Infrastructure**: Serverless—runs locally, requires only free Google AI Studio API key, SQLite storage (no external database).

## If I Had More Time

**Feature Enhancements**:
- **Multi-book itineraries**: Combine multiple books into one journey (e.g., Jane Austen tour covering Pride and Prejudice, Emma, Sense and Sensibility)
- **Route optimization**: Integrate mapping APIs for actual drive times, optimal stop ordering, smart accommodation placement
- **Seasonal intelligence**: Recommend visiting estates in spring/summer, include festival calendars (Bath's Jane Austen Festival)
- **Budget breakdown**: Cost estimates per stop, running totals, free alternative suggestions
- **Export formats**: PDFs with maps, Google Maps links, calendar .ics files, booking platform integration

**Technical Improvements**:
- **Caching layer**: Redis for Google Books/Search results—books don't change, reduce API calls
- **Streaming responses**: Real-time progress updates ("Researching cities... Found 3 so far...")
- **Evaluation expansion**: More eval sets (different genres, regions, edge cases like fictional worlds)
- **Error recovery**: Sophisticated fallbacks when APIs fail—cached results, LLM knowledge, user prompts
- **Rate limit handling**: Smarter retry logic with exponential backoff and quota tracking

**User Experience**:
- **Mobile app**: Web/native app with offline itinerary viewing while traveling
- **Photo integration**: Wikimedia Commons/Google Images for visual engagement
- **Community features**: Share itineraries, rate locations, upload trip photos, contribute updates
- **Reverse flow**: Input destinations, get book recommendations ("Visiting Paris? Read these 15 novels first")

**Scale & Performance**:
- **Async optimization**: More `asyncio.gather()`, pipeline phases
- **Database optimization**: Indexes, connection pooling, PostgreSQL for scale
- **Cost monitoring**: Track API usage per request, provide quota transparency
- **Multi-model support**: Let users choose models (Gemini Flash/Pro, Claude, GPT) for speed/quality tradeoffs

---

**StoryLand AI demonstrates that complex, multi-step research tasks are ideal for agent-based solutions.** By coordinating specialized agents, maintaining state across phases, and integrating human judgment at decision points, it transforms a 3-hour manual process into a 1-minute automated workflow—producing better, more personalized results. The magic of a great book doesn't have to end when you close the cover. With agents, it can become a real journey.
