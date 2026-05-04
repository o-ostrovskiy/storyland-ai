# Architecture Decision Records

This document explains key design decisions in the StoryLand AI project and the rationale behind them.

---

## 1. Two-Phase Workflow with Human-in-the-Loop

### Decision
Split itinerary generation into two separate workflow phases with human region selection between phases 1 and 2.

### Architecture
```
Input: pre-confirmed book_title + author (DTO from Backend)
    ↓
Phase 1: Discovery & Region Analysis (LLM agents)
    ↓
[USER SELECTS REGION(S)] ← Human-in-the-Loop
    ↓
Phase 2: Itinerary Composition (LLM agents)
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

User selects regions of interest, and Phase 2 only generates itineraries for those regions.

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

### Note on ADR #12 (Google Books removal)
Book title and author are now **pre-confirmed inputs** — the Backend service owns the Google Books API lookup and passes the confirmed DTO to this service. storyland-ai no longer performs any book search.

### Alternatives Considered

1. **Single workflow, auto-select all regions**
   - Rejected: Creates impractical multi-continent itineraries

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
- `book_context_agent.py` → BookContext
- `discovery_agents.py` → CityDiscovery, LandmarkDiscovery, AuthorSites

**Note:** `book_metadata_agent.py` previously followed this pattern (researcher + formatter). It was later removed entirely when the ADK web UI was dropped (see ADR #12 and #13).

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
# Set in .env to use SQLite backend
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

## 8. FastAPI SSE Streaming API

### Decision
Add a FastAPI HTTP API layer with Server-Sent Events (SSE) streaming for the agent workflow, splitting the two-phase workflow into two streaming endpoints.

### Architecture
```
Client                          FastAPI Server
  |                                |
  |-- POST /discover ------------->|  Phase 1: Discovery agents
  |<-- SSE: metadata --------------|  (book DTO formatted into BookMetadata)
  |<-- SSE: progress (×N) ---------|
  |<-- SSE: regions ----------------|  ← regions for user selection
  |<-- SSE: done {job_id} ---------|
  |                                |
  |  [user selects regions]        |
  |                                |
  |-- POST /{job_id}/compose ----->|  Phase 2: Composition agent
  |<-- SSE: progress --------------|
  |<-- SSE: itinerary --------------|
  |<-- SSE: done ------------------|
```

### Rationale

**Problem:** The two-phase workflow takes ~60 seconds. Synchronous HTTP would mean clients wait with no feedback. The human-in-the-loop region selection between phases 1 and 2 doesn't fit a single request-response cycle.

**Solution:** Two SSE streaming endpoints that mirror the existing HITL pattern:
1. `/discover` streams Phase 1 (discovery + region analysis), returns regions with a `job_id`
2. `/compose` accepts the `job_id` + selected region IDs, streams Phase 2

**Key design decisions:**
- **Session ID = Job ID:** The ADK session serves as job storage. All intermediate state (metadata, discoveries, regions) persists in `session.state` between the two HTTP calls.
- **SSE over WebSocket:** SSE is unidirectional (server→client), which matches the workflow pattern. No need for bidirectional communication during streaming.
- **Errors as SSE events:** Once headers are sent (200 OK), HTTP status can't change. All errors during streaming are emitted as `error` SSE events.
- **Fresh LangfusePlugin per request:** Isolates token counters between concurrent requests.

### Trade-offs

**Benefits:**
- **Real-time progress:** Clients see step-by-step progress during the ~60s workflow
- **Preserves HITL:** Two-endpoint design maps naturally to the discover→select→compose flow
- **No existing code changes:** API layer is purely additive — reuses orchestrator, models, and session service
- **Standard protocol:** SSE works in all browsers via `EventSource` API

**Costs:**
- **Long-lived connections:** SSE streams stay open for ~30-60s per phase
- **No reconnection for partial results:** If client disconnects mid-discover, must restart
- **Session lookup requires user_id:** Compose endpoint needs the same `user_id` as discover

### Files
- `api/app.py` — FastAPI application factory with lifespan
- `api/routes.py` — HTTP endpoint definitions
- `api/streaming.py` — Async generators wrapping ADK Runner → SSE events
- `api/models.py` — Request/response/SSE event Pydantic models
- `api/dependencies.py` — Shared app state (config, model, session service)

---

## 9. Transport-Agnostic Core SDK with Thin HTTP Adapter

### Decision
Split the codebase into two distinct layers: a transport-agnostic **Core SDK** (`core/`) that yields domain events, and a thin **HTTP adapter** (`api/`) that maps those events to SSE. Keep all business logic in the Core SDK so it can be consumed directly as a Python library or indirectly via HTTP.

### Architecture

```
Two consumption patterns — same business logic:

Pattern A: Library (direct import)
  Tests / Evaluation tools
    └── from core.executor import WorkflowExecutor
         └── executor.discover("1984")  → yields DomainEvent (in-process)

Pattern B: HTTP API
  Frontend / Gateway
    └── POST /api/v1/itinerary/discover
         └── api/routes.py         (thin wiring)
              └── api/streaming.py  (DomainEvent → SSE dict)
                   └── core/executor.py → yields DomainEvent
```

**Core SDK layer** (`core/`):
- `WorkflowExecutor` — the only public interface; has no HTTP imports
- `executor.discover()` / `executor.compose()` — async generators yielding `DomainEvent`
- `DomainEvent` types: `ProgressEvent`, `MetadataReady`, `RegionsReady`, `ItineraryReady`, `WorkflowError`, `WorkflowComplete`
- `ExecutorConfig` — plain dataclass; config values come from the caller, not `os.environ`

**HTTP adapter layer** (`api/`):
- `api/streaming.py` — single function `domain_event_to_sse(event: DomainEvent) -> dict`; pattern-matches each domain event type → SSE format
- `api/routes.py` — wires HTTP requests to executor calls, returns `EventSourceResponse`
- `api/models.py` — Pydantic request/response models (HTTP-specific, not shared with core)
- Zero business logic — no branching on content, no state manipulation

**Interface between layers:**
```python
# core/events.py — language-native, no serialization
@dataclass(frozen=True)
class RegionsReady:
    job_id: str
    regions: list[dict]
    analysis_note: str

# api/streaming.py — converts at the HTTP boundary only
case RegionsReady(job_id=j, regions=r, analysis_note=n):
    yield {"event": "regions", "data": json.dumps({...})}
```

### Rationale

**Problem:** The three-phase workflow was originally built for CLI/Streamlit use (direct Python calls). Adding a web API shouldn't require rewriting the workflow logic — and the API shouldn't be the only way to use the system.

**Why transport-agnostic core matters:**
- Evaluation tools import `WorkflowExecutor` directly — no HTTP, no serialization, no service to run
- Unit tests instantiate the executor without starting a web server
- Future consumers (WebSocket, gRPC, batch runner) can reuse the same executor without touching `api/`
- The executor can be published as a standalone Python package if needed

**Why a thin adapter (not a fat controller):**
- If business logic lived in `api/routes.py`, it would be invisible to direct library users
- All bugs and features would need to be fixed in two places (routes + library)
- A thin adapter means: if the agent behavior changes, only `core/` changes — HTTP wiring stays the same

### Trade-offs

**Benefits:**
- **Two consumption modes:** Evaluation tools use library mode (zero latency, single process); web frontend uses API mode (language-agnostic, independently scalable)
- **Single source of truth:** `core/executor.py` is the authoritative implementation — no duplicate logic
- **Testability:** `WorkflowExecutor` is directly instantiable in tests without spinning up FastAPI
- **Future-proof:** Adding WebSocket or gRPC transport is a new thin adapter, not a rewrite

**Costs:**
- **Interface drift risk:** The backend gateway (`backend/`) calls the agent API via hand-written `httpx` calls. If `api/models.py` renames a field, the backend silently sends wrong data (no compile-time check). Mitigation: keep a typed `AgentClient` wrapper in the gateway that uses the Pydantic request models.
- **Domain event → SSE mapping is manual:** `domain_event_to_sse()` in `api/streaming.py` must be updated whenever a new `DomainEvent` type is added. Missing a case silently drops the event.

### Files
- `core/executor.py` — `WorkflowExecutor`, transport-agnostic async generator
- `core/events.py` — frozen dataclasses for all domain events
- `core/types.py` — `ExecutorConfig` (plain dataclass, no env coupling)
- `api/streaming.py` — `domain_event_to_sse()`: the only place domain events become HTTP
- `api/routes.py` — thin wiring, no business logic

---

## 11. Job Failure Status Tracking via Session State Flag

### Decision
Track terminal workflow failures with an explicit `job_failed` boolean flag in session state, rather than inferring failure from the absence of data.

### Architecture

```
WorkflowExecutor error path:
    await _mark_session_failed(job_id, user_id)
        → append_event(session, state_delta={"job_failed": True})
        ↑ In-place mutation of session.state does NOT persist across get_session() calls
          in ADK's InMemorySessionService (each call returns a new Session object).
          Persistence requires append_event() with state_delta.
    yield WorkflowError(...)
    yield WorkflowComplete(...)

WorkflowExecutor compose() retry (after validation passes):
    append_event(session, state_delta={
        "job_failed": False,          → clears stale failure flag
        "final_itinerary": None,      → removes stale itinerary from a prior successful compose
        "selected_regions": [...],    → persists selected region IDs for COMPOSING status
    })

_derive_job_status() precedence (routes.py):
    job_failed      → FAILED      (terminal failure — highest priority)
    final_itinerary → COMPLETED   (terminal success)
    selected_regions → COMPOSING
    regions         → REGIONS_READY
    book_metadata   → DISCOVERING
    (none)          → SEARCHING
```

**Error paths covered:**
- `discover()`: book search exception, book not found, discovery timeout, generic exception, `asyncio.CancelledError`
- `compose()`: no regions, invalid region IDs, extraction failure, composition timeout, generic exception, `asyncio.CancelledError`
- **Excluded**: session creation failure (no session exists yet — `/status` returns 404, which is correct)

### Rationale

**Problem:** Status was derived purely from which data keys were present in session state. This worked for the happy path but had a critical gap: after any error, the session still contained whatever partial data existed at failure time, so `/status` would report the last _in-progress_ phase indefinitely instead of `failed`.

For example, after "book not found":
- `book_metadata` is absent → status reads `searching` (correct during search, wrong after failure)

After a composition timeout:
- `book_metadata` ✓, `region_analysis` ✓, `selected_regions` ✓ → status reads `composing` forever

**Why an explicit flag?**

The fundamental issue is that "no data yet" and "failed before data was written" are indistinguishable from presence checks alone. An explicit `job_failed` flag makes the failure state unambiguous and directly writable from any error path.

**Why `FAILED` beats `COMPLETED` in precedence?**

If a compose retry succeeds, `final_itinerary` is written and `job_failed` is `False` (cleared at retry start) — so both checks agree on `COMPLETED`. The conflict only arises when a retry *fails*: `job_failed=True` is set, but a `final_itinerary` from a prior successful compose still lives in session state. Checking `job_failed` first ensures the current failure is visible rather than silently masked by the stale result.

`clear_final_itinerary()` (called at retry start) removes the stale itinerary from the previous run. This is the primary guard; the `FAILED > COMPLETED` precedence is a secondary defence for any edge cases where both flags end up set simultaneously (e.g., runner writes `final_itinerary` and then an extraction error occurs).

**Why clear both `job_failed` and `final_itinerary` in `compose()` rather than elsewhere?**

Clearing happens unconditionally after validation passes (valid regions, valid IDs) — at the point where we're committed to running a new composition attempt. Doing it earlier (before validation) would clear the state even if we're about to reject the request with a validation error, leaving a misleading "no data" state instead of `failed`. Doing it at the start of `discover()` would not help because discover always creates a brand-new session with a new `job_id`.

**`asyncio.CancelledError` handling:**

When a client disconnects, the framework cancels the SSE generator coroutine. Once caught in the `except asyncio.CancelledError` block, subsequent `await` calls are not automatically cancelled — we're past the cancellation point. `_mark_session_failed()` is therefore safe to `await` here before re-raising.

### Trade-offs

**Benefits:**
- **Correct terminal states:** Clients can distinguish "still running" from "failed" reliably
- **Retryable compose:** Stale failure flag is cleared automatically on the next valid retry
- **Disconnection safety:** Client disconnects mark the job failed instead of leaving it in-progress forever
- **Minimal state overhead:** One boolean per job in the existing session state dict

**Costs:**
- **Async cleanup in cancellation handler:** `await _mark_session_failed()` runs inside an `except CancelledError` block — unusual but safe (see rationale above)
- **Flag requires explicit clearing via `append_event`:** Compose retries must call `append_event` with `state_delta={"job_failed": False, ...}`. In-place mutation of `session.state` after `get_session()` does not persist — ADK's `InMemorySessionService` reconstructs the Session object on each retrieval. This was the root cause of a bug where `/status` continued to report `FAILED` after a successful retry.
- **`_mark_session_failed` is best-effort:** If the session service itself is unavailable, the flag is not set (failure is logged and swallowed). The `/status` endpoint may then report a stale in-progress state, which is preferable to masking the original error.

### Alternatives Considered

1. **Separate status key (`_job_status = "failed"`)**
   - Considered: Store the full status string in state
   - Rejected: Adds redundancy with the derived-status logic; `_derive_job_status` would need to both read the key AND fall back to data-presence derivation for forward/backward compatibility

2. **Dedicated status table in the database**
   - Considered: Track job lifecycle in a separate `job_status` table
   - Rejected: Over-engineering; session state is already the job store, adding a separate table creates two sources of truth

3. **Error event replay on `/status`**
   - Considered: Store the last `WorkflowError` in session state and surface it on `/status`
   - Partially implemented: The `job_failed` flag could be extended to include the error message in a future iteration

---

## 12. Frontend Owns Google Books Lookup; storyland-ai Accepts Pre-Confirmed DTO

### Decision
Remove the Google Books API integration from storyland-ai. The system architecture is `Frontend → Backend → storyland-ai`. The Backend service now owns book discovery (Google Books API). It passes a pre-confirmed `book_title` + `author` DTO to storyland-ai after the user selects a book. storyland-ai treats these as authoritative inputs.

### Architecture Before
```
storyland-ai: POST /discover { book_title: "1984" }
    → search Google Books API → select best match → extract exact title/author
    → Phase 2: Discovery agents
```

### Architecture After
```
Frontend: user searches for book
Backend: calls Google Books API, user confirms book
storyland-ai: POST /discover { book_title: "1984", author: "George Orwell" }
    → both fields required; no search step
    → Phase 1: Discovery agents
```

### Rationale

**Problem:** storyland-ai was performing its own Google Books API lookup as Phase 1. This created:
- A duplicate search (Backend already owns the book-selection UX and calls Google Books)
- A dependency on `GOOGLE_BOOKS_API_KEY` in storyland-ai's config
- A complex multi-step Phase 1 (search → select best match → extract metadata)
- Fragile behavior when Book API returned no results or ambiguous matches

**Solution:** Move book selection entirely to the Backend service. storyland-ai receives a pre-confirmed DTO and proceeds directly to discovery.

### Changes
- `tools/google_books.py` — deleted
- `core/executor.py` — Phase 1 replaced with direct `BookMetadata` construction
- `api/models.py` — `author` field made required (both `book_title` and `author` required)
- `agents/book_metadata_agent.py` — deleted (no longer needed; see ADR #13)

### Trade-offs

**Benefits:**
- **Separation of concerns:** storyland-ai focuses on itinerary generation; Backend handles book discovery
- **Simpler code:** Removes ~300 lines of Google Books integration, retry logic, and ambiguity handling
- **Faster workflows:** Phase 1 (book search) is eliminated from storyland-ai's processing time
- **No API key dependency:** `GOOGLE_BOOKS_API_KEY` no longer needed in storyland-ai config

**Costs:**
- **Caller must provide both fields:** Any client calling storyland-ai directly (e.g. CLI, tests) must supply both `book_title` and `author` — there is no fallback lookup

---

## 13. Local Atmosphere Mode (Single-Phase, No HITL)

### Decision
Add a **separate single-phase endpoint** (`POST /api/v1/itinerary/local-atmosphere`) for users who want to feel a book's atmosphere near their current location, instead of retrofitting the existing two-phase discover/compose flow.

### Rationale
- The two-phase HITL design (ADR #1) exists because discovery returns multiple candidate regions and the user must pick one. When the user is already pinned to a single geographic point, region selection is meaningless — there is one region (the area around them) by construction.
- Forking `discover()` to skip region selection conditionally would tangle two distinct workflows into one orchestrator. A separate endpoint keeps the original flow untouched and makes the "near me" semantics explicit at the API boundary.
- The pipeline reuses `book_context_pipeline` (so the composer has themes/era/mood) but **deliberately skips** `city_pipeline`, `landmark_pipeline`, and `author_pipeline` — those agents discover the book's *actual* geography, which is irrelevant when the goal is to find atmospheric matches *elsewhere*.

### Implementation
- New agent `local_atmosphere_pipeline` follows the **two-stage researcher → formatter pattern** (ADR #2): a `google_search`-enabled researcher locates candidate places near the user, then a structured-output formatter validates them into a `TripItinerary`. Reuses the same model class, same session state, and the existing `extract_itinerary_from_response` helper.
- The user's `{lat, lng, label}` is stored under a new `USER_LOCATION` session state key. `radius_km` is enforced soft-side via the prompt (no routing API).
- Frontend collects the location via Nominatim (browser geolocation, ZIP/city lookup, or a saved home location on the User row); the server only sees the resolved coords.

### Trade-offs
**Benefits:**
- Original two-phase flow untouched — no regressions in the travel mode.
- Reuses the validated researcher/formatter pattern; no new model architecture risk.
- Single endpoint = single quota class (counts toward `discover_count`); no schema change needed for accounting.

**Costs:**
- A second top-level endpoint to maintain alongside `discover` + `compose`.
- Radius enforcement is LLM-side only (no routing/distance API), so very sparse rural locations may yield fewer-than-ideal results. Acceptable for v1.

---

## 14. Book Recommendation Agent (On-Demand, Single-Agent, Server-Stamped Chip)

### Context

After generating a travel itinerary based on a book, readers often want to know what to read next — books set in the same destination, books with similar themes, or other works by the same author. This feature adds on-demand book recommendations triggered by clicking a "Find books like this" chip in the UI.

### Decision

**Single-agent pipeline (not two-stage researcher → formatter):** Books are well-known entities to the LLM. A single `LlmAgent` with `google_search` and `output_schema=BookRecommendationsResult` is enough to search, validate, and structure results in one pass. This contrasts with the expansion and local-atmosphere flows, which use two-stage pipelines because addresses and place details benefit from a dedicated research pass followed by strict formatting.

**Server-stamped chip (not LLM-generated):** The "Find books like this" chip is created deterministically by the executor after composition (`_build_book_recommendation_chip`) — not by the LLM. The chip dict is stored in session state under `BOOK_RECOMMENDATION_CHIP` (and its UUID under `BOOK_RECOMMENDATION_CHIP_ID`), and surfaced as a dedicated `book_recommendation_chip` field on the `itinerary` and `expansion` SSE events (separate from `suggestions[]`, which is reserved for expansion chips). The `/recommend-books` endpoint validates the incoming `action_id` against the stored id. This eliminates LLM flake risk and provides clean routing: expansion chips go to `/expand`; the books chip goes to `/recommend-books`.

**Separate lock and counter from expansion:** `BOOK_RECS_IN_PROGRESS` and `BOOK_RECOMMENDATION_COUNT` are distinct from expansion's equivalents, allowing concurrent expand + recommend-books without false conflicts.

**No follow-up chips:** Book recommendations are a terminal action. The hard cap is 5 requests per session.

### Files Affected

- `agents/book_recommendation_agent.py`, `agents/orchestrator.py`, `agents/prompts/v2.json`, `agents/prompts.py`
- `models/book.py` — `BookRecommendation`, `BookRecommendationsResult`
- `core/events.py`, `core/session_state.py`, `core/extraction.py`, `core/executor.py`
- `api/models.py`, `api/streaming.py`, `api/routes.py` — `POST /api/v1/itinerary/{job_id}/recommend-books`

### Trade-offs

**Benefits:** Zero LLM flake for chip routing; single-agent avoids 2x LLM cost; save infrastructure requires no changes; independent lock allows concurrency with expansion.

**Costs:** New endpoint to maintain; image_url may be null (covers resolved by frontend); single-agent is less hallucination-resistant than two-stage, acceptable for book titles.

---

## Summary: Key Architectural Patterns

| Pattern | Benefit | Trade-off |
|---------|---------|-----------|
| **Two-phase HITL** | User control, practical itineraries | Not fully autonomous |
| **Two-stage pipelines** | Anti-hallucination, type safety | 2x LLM calls |
| **Parallel discovery** | 3x speedup | Rate limit risk |
| **Exponential backoff** | Reliability on rate limits | Potential long waits |
| **Session state** | Cross-workflow data flow | Untyped dict |
| **Dual backends** | Dev speed + prod persistence | Config complexity |
| **Workflow timeout** | Safety net for hangs | Abrupt failures |
| **SSE streaming API** | Real-time progress, HITL preserved | Long-lived connections |
| **Transport-agnostic core SDK** | Library or HTTP consumption, same logic | Interface drift risk between gateway and API layer |
| **Failure status flag** | Terminal failures visible via /status, retryable | State must be persisted via `append_event`, not in-place mutation |
| **Gateway secret** | Blocks direct access when deployed behind a gateway | Health endpoint intentionally excluded; leave secret empty for standalone dev |
| **Local-atmosphere mode** | Lets readers feel a book without traveling; reuses pipeline pattern | Separate endpoint to maintain; LLM-side radius enforcement only |

These patterns work together to create a **reliable, performant, and user-friendly** multi-agent system for generating literary travel itineraries.
