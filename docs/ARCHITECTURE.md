# Architecture Decision Records

This document explains key design decisions in the StoryLand AI project and the rationale behind them.

---

## 1. Three-Phase Workflow with Human-in-the-Loop

### Decision
Split itinerary generation into three separate workflow phases with human region selection between phases 2 and 3.

### Architecture
```
Phase 1: Metadata Extraction
    ↓
Phase 2: Discovery & Region Analysis
    ↓
[USER SELECTS REGION(S)] ← Human-in-the-Loop
    ↓
Phase 3: Itinerary Composition
```

### Rationale

**Problem:** Books often span multiple distant locations. Examples:
- "Gone with the Wind" → Atlanta (USA) + Civil War sites across Georgia
- "Pride and Prejudice" → England (Derbyshire, Bath) + author sites in Hampshire
- "The Nightingale" → Multiple regions of France during WWII

Auto-generating itineraries for ALL discovered regions would create:
- Impractical multi-continent trip plans
- Wasted tokens on regions the user doesn't want to visit
- Loss of user agency and personalization

**Solution:** After discovering and analyzing all locations, present region options to the user:
```
[1] England (Derbyshire, Bath) - 7 days
[2] Scotland (Edinburgh) - 3 days
[3] Author Sites (Hampshire) - 2 days

Which region(s) would you like to explore? [1/2/3]:
```

User selects regions of interest, and Phase 3 only generates itineraries for those regions.

### Trade-offs

**Benefits:**
- User control over trip scope (single region vs. multi-region)
- Prevents impractical itineraries (no forced England → Scotland → France trips)
- Token efficiency (only generate itineraries for wanted regions)
- Better personalization (user implicitly expresses preferences through selection)

**Costs:**
- Not fully autonomous (requires human input)
- Cannot use in eval mode without modification (see ADR #5)
- Adds UX complexity (users must understand region grouping)

### Alternatives Considered

1. **Single workflow, auto-select all regions**
   - Rejected: Creates impractical multi-continent itineraries
   - Used in eval_workflow variant for ADK evals

2. **Post-generation filtering**
   - Rejected: Wastes tokens generating unwanted itineraries
   - Still requires human input, but after expensive generation

3. **Upfront preferences**
   - Considered: "Which countries are you willing to visit?"
   - Rejected: Requires users to know geography before seeing options

---

## 2. Two-Stage Pipeline Pattern

### Decision
Every research agent follows a two-stage sequential architecture:
1. **Researcher Agent** (LLM + tools)
2. **Formatter Agent** (LLM + Pydantic output_schema)

### Implementation
```python
# Stage 1: Researcher with tools
researcher = LlmAgent(
    tools=[google_search],
    instruction="Search for X and return ALL findings"
)

# Stage 2: Formatter with Pydantic
formatter = LlmAgent(
    output_schema=XDiscovery,  # Pydantic model
    output_key="x_discovery",
    instruction="Format into XDiscovery. Do NOT hallucinate."
)

pipeline = SequentialAgent(sub_agents=[researcher, formatter])
```

### Rationale

**Problem:** LLMs can hallucinate data, especially when combining research with formatting. Single-agent approaches risk:
- Fabricating book metadata when Google Books returns empty results
- Inventing city names that sound plausible but don't exist
- Making up landmarks that "feel right" for the book's setting

**Anti-Hallucination Mechanism:**

1. **Stage 1 (Researcher):**
   - Has access to tools (Google Search, Google Books API)
   - Gathers **real data** from authoritative external sources
   - Returns raw, unprocessed results
   - No output schema constraints

2. **Stage 2 (Formatter):**
   - **Has NO tools** (cannot generate new data)
   - Only processes previous agent's conversation history
   - Validates against strict Pydantic schema
   - Instruction: "If researcher found nothing, return empty fields - do NOT hallucinate"

**Why it works:** Formatter cannot fabricate data because it has no tools. It can only structure what the researcher found. Empty research → Empty output.

### Trade-offs

**Benefits:**
- **Data integrity:** All output comes from real API calls, not LLM imagination
- **Type safety:** Pydantic validation catches malformed data early
- **Separation of concerns:** Research logic separate from formatting logic
- **Debuggability:** Can see raw research results vs. formatted output
- **Reusability:** Same pattern used 5+ times (metadata, context, city, landmark, author)

**Costs:**
- **2x LLM calls** per pipeline (higher cost)
  - Example: book_metadata = researcher call + formatter call
- **Increased latency:** Sequential execution adds ~2-3 seconds per pipeline
- **Code duplication:** Similar two-stage pattern across multiple agent files

### Alternatives Considered

1. **Single agent with tools + output_schema**
   - Rejected: ADK agents can have tools OR output_schema, not both (in our usage)
   - Even if possible, single agent more prone to hallucination

2. **Tools with Pydantic return types**
   - Considered: Make Google Books tool return BookMetadata directly
   - Rejected: Tools handle API calls, not complex data structuring
   - LLM better at parsing varied API responses into consistent schemas

3. **Post-processing validation**
   - Considered: Single agent, then Python validates output
   - Rejected: LLM already generated hallucinated data, validation just catches it
   - Two-stage prevents hallucination at generation time

### Usage Across Codebase
This pattern appears in:
- `book_metadata_agent.py` → BookMetadata
- `book_context_agent.py` → BookContext
- `discovery_agents.py` → CityDiscovery, LandmarkDiscovery, AuthorSites

---

## 3. Parallel Discovery Execution

### Decision
Run city, landmark, and author discovery agents concurrently using `ParallelAgent`.

### Implementation
```python
parallel_discovery = ParallelAgent(
    name="parallel_discovery",
    sub_agents=[
        city_pipeline,      # Searches for cities in book's setting
        landmark_pipeline,  # Searches for landmarks mentioned in book
        author_pipeline,    # Searches for author birthplace, museums, etc.
    ],
)
```

### Rationale

**Performance Analysis:**
- **Sequential execution:** ~45-60 seconds (3 agents × 15-20s each)
- **Parallel execution:** ~15-20 seconds (max of 3 concurrent operations)
- **Speedup:** 3x faster with same token usage

**Why safe to parallelize:**
- Each agent makes independent Google Search queries
- No data dependencies between city/landmark/author discovery
- All three need the same input (book_title, author, book_context)
- Results are aggregated later by region_analyzer

### Trade-offs

**Benefits:**
- **3x speedup:** User sees results in ~20s instead of ~60s
- **No additional cost:** Same number of LLM calls and tokens
- **Better UX:** Reduced waiting time improves perceived performance
- **Resource efficiency:** Better utilization of API rate limits

**Costs:**
- **Higher peak concurrency:** 3 simultaneous Google Search API calls
- **Rate limit risk:** More likely to hit 15 RPM limit on free tier
  - Mitigation: Exponential backoff retry (see ADR #4)
- **Debugging complexity:** Harder to trace issues when agents run concurrently

### Why NOT Parallel Everywhere?

Other parts of the workflow remain **sequential** because they have dependencies:

```python
# Sequential workflow (dependencies exist)
sub_agents = [
    book_context_pipeline,    # Must run first (provides context)
    reader_profile,           # Can run anytime (reads preferences)
    parallel_discovery,       # Needs book_context output
    region_analyzer,          # Needs parallel_discovery output
]
```

- `book_context` must complete before discovery (discovery needs setting info)
- `region_analyzer` must wait for all discoveries (needs cities to group)
- `trip_composer` needs region selection (can't run until Phase 3)

**Rule:** Parallelize only when agents have zero data dependencies.

---

## 4. Exponential Backoff Retry Strategy

### Decision
Configure HTTP retries with aggressive exponential backoff (exp_base=7) for rate limits.

### Implementation
```python
retry_config = types.HttpRetryOptions(
    attempts=5,           # Retry up to 5 times
    exp_base=7,           # Aggressive backoff: 1s → 7s → 49s
    initial_delay=1,      # Start with 1 second
    http_status_codes=[429, 500, 503, 504]
)
model = Gemini(model=config.model_name, retry_options=retry_config)
```

### Rationale

**Problem:** Gemini API free tier has **15 requests per minute (RPM)** limit. With:
- 6+ agents in workflow
- Parallel discovery (3 concurrent agents)
- Each agent making multiple Google Search calls
- Total: 20-30 API calls in <30 seconds

We frequently hit 429 (Too Many Requests) errors.

**Backoff Schedule:**
- 1st retry: 1 second delay
- 2nd retry: 7 seconds delay
- 3rd retry: 49 seconds delay
- 4th retry: 343 seconds delay (5.7 minutes)
- 5th retry: 2,401 seconds delay (40 minutes)

**Why aggressive (exp_base=7)?**
- RPM limits reset after 60 seconds
- Small delays (1s, 2s) don't help if limit window hasn't reset
- 7-second delay gives time for window to partially reset
- 49-second delay almost guarantees new rate limit window
- Better to wait once than fail permanently

### Trade-offs

**Benefits:**
- **Reliability:** Workflows succeed despite rate limits
- **No user intervention:** Automatic recovery from transient errors
- **Covers server errors:** Also retries 500, 503, 504 (server overload)

**Costs:**
- **Potential long waits:** 49s delay feels slow to users
- **Delayed failures:** 5 retries can take minutes before final error
- **Masking problems:** Retry might hide persistent issues

### Alternatives Considered

1. **Linear backoff (exp_base=2)**
   - Rejected: 1s → 2s → 4s delays too short for RPM window reset
   - Would burn through all retries before rate limit expires

2. **No retries**
   - Rejected: Workflow fails immediately on rate limit
   - Poor UX for free tier users

3. **Fixed delay (e.g., 60s)**
   - Considered: Always wait 60s for rate limit reset
   - Rejected: Wastes time on non-rate-limit errors (500, 503)

4. **Jittered backoff**
   - Considered: Add randomness to delays (7s ± 2s)
   - Not implemented: ADK HttpRetryOptions doesn't support jitter
   - Would help if multiple users hit limits simultaneously

### Future Improvements

1. **Adaptive backoff** based on `Retry-After` header (if Gemini provides it)
2. **Circuit breaker** pattern to fail fast after sustained errors
3. **Rate limit tracking** to proactively slow down before hitting limits

---

## 5. Session State for Cross-Workflow Communication

### Decision
Use ADK session state as shared memory for passing data between workflow phases.

### Implementation
```python
# Phase 1: Write to state
formatter = LlmAgent(
    output_key="book_metadata",  # Writes to session.state["book_metadata"]
    output_schema=BookMetadata,
)

# Phase 2: Read from state
book_metadata = session.state.get("book_metadata", {})
exact_title = book_metadata.get("book_title")

# Phase 2→3: Manual state update
session.state["selected_regions"] = selected_regions  # User's choice
```

### Rationale

**Problem:** Three-phase workflow requires data to flow across Runner instances:
- Phase 1 extracts metadata → Phase 2 needs exact title/author
- Phase 2 discovers regions → Phase 3 needs user's selected regions
- All phases share user preferences → Read from `state["user:preferences"]`

**ADK Session Model:**
- Each `Runner` instance gets a `session_service`
- Session identified by `(app_name, user_id, session_id)`
- `session.state` is a persistent dict (JSON-serializable)
- Agents write to state via `output_key="key_name"`
- Python code reads/writes via `session.state["key"]`

### Trade-offs

**Benefits:**
- **Simple API:** Dict-like access to shared state
- **Persistence:** DatabaseSessionService saves to SQLite (survives restarts)
- **Multi-user:** Isolated state per user_id
- **Type flexibility:** Store any JSON-serializable data

**Costs:**
- **No schema enforcement:** `session.state` is untyped dict
- **Manual key management:** Risk of typos (`"book_metdata"` vs. `"book_metadata"`)
- **No transactions:** Race conditions possible with concurrent access
- **Debugging:** Must inspect database to see state between phases

### Alternatives Considered

1. **Function parameters**
   - Rejected: Can't pass data between separate Runner instances
   - Would require merging all phases into single workflow (loses HITL)

2. **Conversation history**
   - Considered: Agents read previous agents' outputs from history
   - Rejected: Fragile (depends on parsing text), not structured

3. **External database**
   - Considered: Write discoveries to Postgres, read in next phase
   - Rejected: Over-engineering for small data volumes
   - session.state with SQLite backend is sufficient

4. **In-memory cache**
   - Rejected: Doesn't persist across restarts
   - session.state already has in-memory backend option

### Eval Workflow Variant

For ADK evals (no HITL), we use a single workflow with region_analyzer:

```python
eval_workflow = SequentialAgent(
    sub_agents=[
        book_metadata_pipeline,
        book_context_pipeline,
        parallel_discovery,
        region_analyzer,  # Creates regions
        trip_composer,    # Auto-uses all regions (no selection)
    ]
)
```

Trip composer reads `state["region_analysis"]` directly, no manual selection step.

---

## 6. Dual Session Backend (In-Memory vs. SQLite)

### Decision
Support two session storage backends: `InMemorySessionService` and `DatabaseSessionService`.

### Implementation
```python
def create_session_service(connection_string, use_database):
    if use_database:
        return DatabaseSessionService(db_url=connection_string)
    else:
        return InMemorySessionService()
```

### Rationale

**Development needs:**
- Fast iteration (no database setup)
- Clean slate every run (no state pollution)
- Simple debugging (print session.state)

**Production needs:**
- Persistence across restarts
- Multi-user support with isolation
- Conversation history for debugging
- Future: User preference learning across sessions

### Trade-offs

**In-Memory Backend:**
- ✅ Fast (no I/O)
- ✅ Zero setup (no database required)
- ✅ Clean state every run
- ❌ Lost on restart
- ❌ No multi-user isolation (all share memory)

**SQLite Backend:**
- ✅ Persistent (survives restarts)
- ✅ Multi-user (isolated per user_id)
- ✅ Queryable (SQL for debugging)
- ❌ Slower (disk I/O)
- ❌ Requires database file/permissions

### Usage
```bash
# Development (in-memory)
python main.py "1984"

# Production (SQLite)
python main.py "1984" --database

# Or set in .env
USE_DATABASE=true
```

### Future: PostgreSQL
For production scale, add third backend:
```python
elif use_database and connection_string.startswith("postgresql://"):
    return PostgresSessionService(db_url=connection_string)
```

Requires implementing PostgresSessionService with same interface.

---

## 7. Workflow Timeout with Diagnostic Reporting

### Decision
Wrap all workflow phases in `asyncio.timeout()` context with diagnostic error reporting.

### Implementation
```python
event_count = 0
try:
    async with asyncio.timeout(workflow_timeout):
        # Phase 1, 2, 3...
        async for event in runner.run_async(...):
            event_count += 1
except asyncio.TimeoutError:
    raise WorkflowTimeoutError(
        f"Workflow exceeded {workflow_timeout}s timeout. "
        f"Processed {event_count} events before timeout."
    )
```

### Rationale

**Problem:** Workflows can hang on:
- Rate limit errors (429) with long backoff delays
- Slow Google Search queries (10+ seconds per query)
- Network issues (dropped connections, DNS failures)
- LLM generation stalls (rare but possible)

Without timeout: Workflow runs indefinitely, user doesn't know if it's stuck or slow.

**Diagnostic Value of event_count:**
- `event_count=5` → Stuck in Phase 1 (metadata extraction)
- `event_count=50` → Stuck in Phase 2 (discovery running)
- `event_count=100` → Stuck in Phase 3 (composition)

Helps debug WHERE the timeout occurred without detailed logs.

### Trade-offs

**Benefits:**
- **User confidence:** Workflow won't hang forever
- **Actionable errors:** "Processed 5 events" tells user it failed early
- **Configurable:** `--timeout 600` for complex books
- **Safety net:** Catches unexpected hangs

**Costs:**
- **False positives:** Complex books might need >300s legitimately
- **Abrupt failures:** Timeout interrupts mid-agent execution
- **Lost work:** Partial results not saved (could save Phase 1/2 results)

### Default Timeout: 300s (5 minutes)

**Typical workflow timing:**
- Phase 1 (metadata): ~5-10 seconds
- Phase 2 (discovery): ~30-60 seconds
- Phase 3 (composition): ~20-30 seconds
- **Total:** ~60-100 seconds

300s provides 3x safety margin.

### Future Improvements

1. **Per-phase timeouts**
   ```python
   async with asyncio.timeout(60):  # Phase 1: max 60s
       await phase_1()
   ```

2. **Partial result saving**
   - Save Phase 1 metadata even if Phase 2 times out
   - User can retry from last successful phase

3. **Streaming progress**
   - Show "Discovering cities... 45s elapsed" while waiting
   - User knows it's working, not stuck

---

## Summary: Key Architectural Patterns

| Pattern | Benefit | Trade-off |
|---------|---------|-----------|
| **Three-phase HITL** | User control, practical itineraries | Not fully autonomous |
| **Two-stage pipelines** | Anti-hallucination, type safety | 2x LLM calls |
| **Parallel discovery** | 3x speedup | Rate limit risk |
| **Exponential backoff** | Reliability on rate limits | Potential long waits |
| **Session state** | Cross-workflow data flow | Untyped dict |
| **Dual backends** | Dev speed + prod persistence | Config complexity |
| **Workflow timeout** | Safety net for hangs | Abrupt failures |

These patterns work together to create a **reliable, performant, and user-friendly** multi-agent system for generating literary travel itineraries.
