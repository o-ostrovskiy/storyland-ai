# Architecture Decision Records

This document explains key design decisions in the StoryLand AI project and the rationale behind them.

> Reconciled against `main` on 2026-07-19 (MYS-554). Per engineering-standards §6, a PR that makes a new architecture decision appends a dated numbered entry here in the same PR.

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

**Wire-numbering note (2026-07-19):** the workflow is conceptually two-phase, but `Phase(IntEnum)` in `core/events.py` keeps its legacy values — `BOOK_SEARCH = 1` (retired, ADR #12), `DISCOVERY = 2`, `COMPOSITION = 3` — and SSE `progress` events serialize these ints, so clients receive **2 for discovery and 3 for composition** today. Renumbering the enum is a breaking wire change and must be its own code PR, never a docs edit. In this document, "Phase 1/Phase 2" means the conceptual workflow; anything about events or diagnostics uses the wire values explicitly.

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

**Note:** `book_metadata_agent.py` previously followed this pattern (researcher + formatter). It was later removed entirely when the ADK web UI was dropped and the Google Books lookup moved to the frontend (see ADR #12).

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
- `trip_composer` needs region selection (can't run until Phase 2)

**Rule:** Parallelize only when agents have zero data dependencies.

---

## 4. Bounded Exponential Backoff Retry Strategy (revised)

> **Revised 2026-07 (doc reconciled 2026-07-19).** The original decision here was aggressive backoff (`exp_base=7`, `attempts=5`, no delay cap, hardcoded inline in `core/executor.py`). It was replaced: its ~400s worst-case cumulative backoff exceeded the 300s `workflow_timeout`, so a 429/500/503 burst sat in multi-minute blind backoff until the timeout wall fired — surfacing a false "timed out" error to the user precisely when the API was throttling.

### Decision
Retry transient Gemini errors with **bounded** exponential backoff — `exp_base=2`, an explicit `max_delay` cap, and env-configurable knobs — built in one place, `core/retry.py`, shared by the executor and the eval runner.

### Implementation
```python
# core/retry.py — build_retry_options(), the single source of truth
# Defaults (env-overridable via common/config.py):
#   RETRY_ATTEMPTS=4 · RETRY_EXP_BASE=2.0 · RETRY_MAX_DELAY=12.0
retry_config = build_retry_options(
    attempts=4, exp_base=2, initial_delay=1, max_delay=12
)  # called from core/executor.py and evaluation/tools/run_scheduled_eval.py
```
Retry status codes unchanged: `[429, 500, 503, 504]` (`RETRY_STATUS_CODES` in `core/retry.py`).

### Rationale

**Constraint:** every workflow phase runs inside `workflow_timeout` (300s default, ADR #7). The retry schedule must resolve — success or clean error — well inside that window, or backoff converts throttling into false timeouts.

**Backoff schedule (worst case):** 1s → 2s → 4s ≈ **~7s cumulative** (3 retries; the 12s `max_delay` caps any further growth). Two orders of magnitude under the timeout wall.

`core/retry.py` also exposes `worst_case_backoff_seconds()` so the relationship between the retry schedule and `workflow_timeout` is checkable, not folklore.

### Trade-offs

**Benefits:**
- **No false timeouts:** throttling resolves to a fast success or a fast, honest error.
- **Single source of truth:** executor and eval runner can't drift apart.
- **Tunable per environment** via `RETRY_*` env vars without code changes.

**Costs:**
- **Less patience with long throttle windows:** a sustained RPM exhaustion fails fast instead of waiting out the window (acceptable: prod runs on a billed key, and the UI surfaces a clean retryable error).

### Alternatives Considered

1. **Aggressive backoff (`exp_base=7`, no cap)** — the original design. Rejected in practice: worst case (~400s) exceeded `workflow_timeout`, producing blind multi-minute waits and false "timed out" errors.
2. **No retries** — rejected: transient 429/5xx would fail workflows immediately.
3. **Fixed 60s delay** — rejected: wastes time on non-rate-limit errors.

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

**Problem:** Two-phase workflow requires data to flow across Runner instances:
- Phase 1 extracts metadata → Phase 2 needs exact title/author
- The discovery phase produces regions → the composition phase needs the user's selected regions
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
- `event_count=50` → Stuck in discovery (wire `phase: 2`)
- `event_count=100` → Stuck in composition (wire `phase: 3` — legacy enum value, see the wire-numbering note in ADR #1)

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
- Composition (wire `phase: 3`): ~20-30 seconds
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
- **Fresh LangfusePlugin per request:** Isolates token counters between concurrent requests. (Later found insufficient under parallel discovery: branches of one request shared scalar generation/span state — now scoped per ParallelAgent branch via `_branch_key()` in `plugins/langfuse_plugin.py`; see ADR #21.)

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

**Problem:** The two-phase workflow was originally built for CLI/Streamlit use (direct Python calls). Adding a web API shouldn't require rewriting the workflow logic — and the API shouldn't be the only way to use the system.

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

## 10. Gateway Service-to-Service Authentication (fail-closed)

*Decided 2026-06-27 → 2026-07-11 (MYS-399); entry backfilled 2026-07-19 — the Patterns table row existed but the section was never written.*

### Decision
When deployed behind the backend gateway, require a shared secret on every itinerary endpoint and take end-user identity **only** from the trusted `X-User-ID` header the gateway sets after JWT validation. Fail closed.

### Implementation
- `api/dependencies.py` — `verify_gateway_secret`: when `INTERNAL_API_SECRET` is set, all itinerary endpoints require a matching `X-Internal-Secret` header (`/health` stays open). The secret is compared as **bytes**, constant-time.
- Identity: the `X-User-ID` header only — the service trusts nothing else from the request. Missing header → 403 (except standalone dev, where identity falls back to the shared `dev_user`).
- `REQUIRE_GATEWAY_SECRET=true` (`common/config.py`) refuses to **start** with an empty secret — closing the foot-gun where an empty secret silently accepts everyone and `X-User-ID` becomes a forgeable identity.

### Rationale
The service holds per-user sessions; without the secret gate, anyone who can reach the container could read/mutate any user's sessions by forging `X-User-ID`. The gateway owns real authn (JWT); this service only needs to verify "this request came through the gateway."

### Trade-offs
**Benefits:** blocks direct access in prod; identity model stays trivially simple; standalone dev still works.
**Costs:** one more env var to coordinate between gateway and service; `/health` deliberately unauthenticated.

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
- `agents/book_metadata_agent.py` — deleted (no longer needed; see ADR #12)

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

**Two-stage researcher → formatter pipeline:** Mirrors the expansion and local-atmosphere flows. ADK's `LlmAgent` forbids combining `tools=[...]` with `output_schema=...` on the same agent (the model can either reply with structured output OR call tools, not both). The researcher uses `google_search` to find candidate books and capture their facts; the formatter has no tools and applies `output_schema=BookRecommendationsResult` to produce exactly 5 balanced entries. Even though books are well-known LLM entities, web search is essential for fresh recommendations and accurate metadata, so the second stage is necessary.

**Server-stamped chip (not LLM-generated):** The "Find books like this" chip is created deterministically by the executor after composition (`_build_book_recommendation_chip`) — not by the LLM. The chip dict is stored in session state under `BOOK_RECOMMENDATION_CHIP` (and its UUID under `BOOK_RECOMMENDATION_CHIP_ID`), and surfaced as a dedicated `book_recommendation_chip` field on the `itinerary` and `expansion` SSE events (separate from `suggestions[]`, which is reserved for expansion chips). The `/recommend-books` endpoint validates the incoming `action_id` against the stored id. This eliminates LLM flake risk and provides clean routing: expansion chips go to `/expand`; the books chip goes to `/recommend-books`.

**Separate lock and counter from expansion:** `BOOK_RECS_IN_PROGRESS` and `BOOK_RECOMMENDATION_COUNT` are distinct from expansion's equivalents, allowing concurrent expand + recommend-books without false conflicts.

**No follow-up chips:** Book recommendations are a terminal action. The hard cap is 5 requests per session.

### Files Affected

- `agents/book_recommendation_agent.py`, `agents/orchestrator.py`, `agents/prompts/v2.json`, `agents/prompts.py`
- `models/book.py` — `BookRecommendation`, `BookRecommendationsResult`
- `core/events.py`, `core/session_state.py`, `core/extraction.py`, `core/executor.py`
- `api/models.py`, `api/streaming.py`, `api/routes.py` — `POST /api/v1/itinerary/{job_id}/recommend-books`

### Trade-offs

**Benefits:** Zero LLM flake for chip routing; researcher → formatter split satisfies ADK's "tools XOR output_schema" constraint while preserving structured output; save infrastructure requires no changes; independent lock allows concurrency with expansion.

**Costs:** New endpoint to maintain; image_url may be null (covers resolved by frontend); two LLM hops per recommendation request (consistent with expansion/local-atmosphere flows).

---

## 15. Persistent, Content-Versioned Discovery Result Cache (2026-06 → 07-01)

### Decision
Cache Discovery results in a **persistent, SQLite-backed `diskcache`** store that survives restart/redeploy, keyed with a **content fingerprint** so code changes self-invalidate the cache.

### Implementation
- `core/cache.py` + `core/disk_cache.py`; config `CACHE_ENABLED` (default **true** — a missing env var degrades to caching on, never a correctness bug), `CACHE_BACKEND` (`disk` = persistent SQLite on a docker volume, or in-memory), `CACHE_DIR`, `CACHE_TTL_SECONDS`, `CACHE_MAX_ENTRIES`. Effective config logged at boot.
- `core/cache_version.py` — `compute_cache_version()` hashes the model name plus the **source of the prompt/schema modules** (`_PROMPT_MODULES`, which includes `models.discovery`, `models.place_key`, `core.regions`). A model/prompt/schema change mints a new namespace; stale entries can never be served after a deploy that changed what a cached result means. **Boundary:** the fingerprint covers module *source*, not data — `agents.prompts`' `CURRENT_PROMPT_VERSION` bump changes the hash, but an in-place edit to the `v2`/`v3` prompt JSON *content* does not, since that JSON is data rather than source read by `inspect.getsource()`. That content gap is tracked separately as MYS-462, not closed by this ADR.

### Rationale
Discovery is the expensive phase (parallel LLM fan-out + search). Same book → same regions, so recomputing per request wastes tokens and seconds. An earlier `ENABLE_RESULT_CACHE` feature flag was removed — caching is core behavior, not an experiment. Content-fingerprinting removed the post-deploy "cache warmup / manual flush" ritual.

### Trade-offs
**Benefits:** repeat lookups return in milliseconds at zero token cost; persistence survives redeploys; self-invalidation makes deploys safe by construction.
**Costs:** a real-API integration test (`tests/integration/test_cache_real_api.py`) is needed to prove the seam; disk growth bounded by `CACHE_MAX_ENTRIES`/TTL.

---

## 16. Canonical Cross-Job `place_key` Identity (2026-07-15, MYS-435/MYS-460)

### Decision
Mint a canonical `"<cc>:<locality-slug>"` key (ISO-3166-1 alpha-2 country code + canonicalized principal locality) for every grounded region, emitted on the `regions` SSE payload, so the **same real place can be recognized across different book jobs** (combined "readaway" journeys).

### Implementation
- `models/place_key.py` (~650 lines: ISO table, transliteration, slug hardening) — minting only through the ONE checked seam `mint_checked_place_key()`, which enforces locality-matches-cities and country self-consistency.
- `core/regions.py` — `enrich_region_analysis()` stamps `place_key` (and `admin_area`) onto each region; `models/discovery.py` — `RegionOption.place_key`.
- Deliberately **never** derived from `region_id` (a per-response ordinal) or `region_name` (prose). If the checked mint can't be confident it returns `None`: the asymmetry is "a missed combine, never a wrong one."
- The minting modules are part of the cache-version fingerprint (ADR #15), so the persistent cache could not serve keyless/legacy regions after this landed.

### Rationale
Combining two books' journeys needs a place identity to intersect on; nothing existing was stable across jobs. Frontend-side name matching would guess; a wrong merge is worse than no merge.

---

## 17. Harness-First CI: Hash-Locked Deps, Coverage Ratchet, Audit Gate, ADK Pin (2026-07-17; pin 2026-06-18)

### Decision
Make "done" mechanical (per team engineering-standards): reproducible dependencies, a coverage floor that only ratchets up, a strict vulnerability gate, and a guarded major-version pin — all wired to fail CI, not to report.

### Implementation
- **Hash-locked deps:** `requirements.lock` / `requirements-dev.lock` from `uv pip compile --universal --generate-hashes`; the unit workflow (`codex.yml`) installs `--require-hashes` and fails if `make lock` would change the committed lock. Qualifier: `integration-tests.yml` still installs with `pip install -e ".[dev]"` (live resolution) — the lock gates the unit workflow, not yet every CI run.
- **Coverage ratchet:** `fail_under = 74` in `pyproject.toml` (76% baseline; bare `--cov` honoring `[tool.coverage.run]`). Ratchet, not target — raise, never lower.
- **Audit gate:** `pip-audit --strict --require-hashes` on the prod lock; ignores are inline with justification + ticket.
- **ADK pin:** `google-adk[eval]>=1.33.0,<2` in `pyproject.toml` + a CI guard asserting the **locked** resolution stays 1.x (prevents a non-reproducible jump to ADK 2.x).

### Rationale
The agent (and everyone else) is only as autonomous as the definition of "done" is mechanical; a green build must mean tests passed, coverage didn't regress, deps are exactly what was reviewed, and no known-vulnerable package ships.

---

## 18. Sentry Error Tracking with Privacy Hardening (2026-07-17/18, MYS-541/543)

### Decision
Env-gated Sentry for the API — performance tracing **off** by default, logs and metrics **opt-out** (on unless disabled), with aggressive scrubbing, because agent runs are already traced in Langfuse and our payloads carry user reading taste and location data. Setting only `SENTRY_DSN` ships exceptions **plus INFO+ logs plus metrics**; set `SENTRY_ENABLE_LOGS=false` / `SENTRY_ENABLE_METRICS=false` for a genuinely errors-only posture.

### Implementation
- `api/sentry.py`, gated on `SENTRY_DSN`; `common/logging.py` bridges structlog → Sentry (INFO+ as Logs, auto-counted `log.events` metrics) — both default **on** (`SENTRY_ENABLE_LOGS`/`SENTRY_ENABLE_METRICS` default `true`). Release tagging via `SENTRY_RELEASE` (deployed SHA).
- Performance tracing off by default: `traces_sample_rate` resolves to `None` (not `0.0`) so inbound trace headers that would bypass scrubbers aren't honored.
- Privacy: `send_default_pii=False`, `max_request_body_size="never"`, `include_local_variables=False` (frame locals hold prompts, taste context, lat/lng), URL query-string scrubbing across events/logs/breadcrumbs/spans, health-probe noise dropped.

### Rationale
We need to see production exceptions without shipping user data to a third party; Langfuse owns the "what did the agents do" question, Sentry only the "what broke" question.

---

## 19. Single-Box Deployment via G5-Gated Manual Workflow (2026-06-24 → 07-02)

### Decision
Decommission ECR-build/ECS-deploy; production is a **single self-hosted Lightsail box** running docker compose, deployed only through a founder-gated manual workflow.

### Implementation
- `.github/workflows/deploy-ai-prod.yml` — `workflow_dispatch`-only, paused on the `production` GitHub Environment for required-reviewer approval (that pause **is** the G5 gate), then a reusable SSH-deploy workflow.
- Hardening: just-in-time SSH via OIDC — port 22 opens only during a deploy and closes after (fully closed at rest); `rsync --delete` so removals/rollbacks reach the box; on-box `.env.prod` excluded from rsync. `Dockerfile`: non-root, HEALTHCHECK, digest-pinned base.
- CI (`.github/workflows/ci-cd.yml`) compiles and tests; it never deploys.

### Rationale
One small product, one box: ECS added cost and indirection without capacity needs. The founder gate keeps deploys a deliberate act; JIT SSH removes the standing attack surface a always-open management port is.

---

## 20. Place→Book Reverse Routing + Server-Derived Grounding (2026-06-19 → 06-22)

### Decision
Add an isolated, gateway-internal capability that routes a **place back to books** (the reverse of the main flow), and derive grounding labels **server-side** rather than trusting the model's self-report.

### Implementation
- `core/place_to_book.py` + `models/place_to_book.py` + `agents/place_to_book_agent.py`, exposed as its own endpoint; follows the researcher→formatter pattern (ADR #2); dedicated eval runner `evaluation/tools/run_place_to_book_eval.py`.
- Grounding: `CityStop` gained `match_type` (`literal | historical | thematic | vibe`) and `grounding_source`; `match_type` is **derived server-side** so ungrounded literal/historical claims get downgraded regardless of what the model asserts about itself.

### Rationale
"What should I read for this place?" is the inverse product question and shares the anti-hallucination machinery. The model grading its own grounding is exactly the "uncomputed claim" class the team bans in UI copy — so the server computes it.

---

## 21. API Abuse + Operational Hardening (grouped; 2026-06-23 → 07-17)

Four smaller decisions, recorded together:

- **Rate limiting + concurrency cap** (2026-06-23): per-identity request rate limit and in-flight concurrency cap on discovery endpoints (`api/ratelimit.py`); input-size bounds return 422 **before** any Gemini tokens are spent.
- **Bounded session retention** (2026-06-24): a periodic retention sweep (`services/session_retention.py`) bounds the in-memory/SQLite session store — extends ADR #6, which predates it.
- **Recommendation-tone guardrail** (2026-06-28): rec explanations are fit-only and never grade the reader (`core/guardrails/tone_guardrail.py`).
- **Langfuse per-branch scoping** (2026-07-17, MYS-398): generation/agent-stack/span state is scoped **per ParallelAgent branch** (`_branch_key()` in `plugins/langfuse_plugin.py`) — per-request plugin instances (ADR #8) were not enough under the ADR #3 parallel fan-out.

---

## 22. Shared Run Harness (Single ADK-Facing Scaffold for All Flows) (2026-07-19)

### Context

By mid-2026 `core/executor.py` had grown to ~1,700 lines: five endpoint flows (`discover`, `compose`, `local_atmosphere`, `expand`, `recommend_books`) plus the place→book resolver each carried a near-identical copy of the same scaffolding — Runner construction, the event-drain loop with agent→progress mapping, researcher-text capture, final-response tracking, the workflow timeout, and a four-branch `TimeoutError` / `CancelledError` / `Exception` boundary ending in `WorkflowError` + `WorkflowComplete`. Every new flow copy-pasted ~150 lines, and any change to how we drive ADK (e.g. the planned ADK 2.x migration) had to be applied and verified six times.

### Decision

Extract the ADK-facing scaffolding into `core/run_harness.py` as four small primitives, leaving all business logic (guards, caching, merging, grounding filters) in the flows:

- **`pump_events(runner, ...)`** — drains `runner.run_async`, yields at most one `ProgressEvent` per agent (from a per-flow `agent_steps` map), and optionally fills a **`RunCapture`** with per-author text (grounding post-validation input) and the final response.
- **`run_guarded(body, GuardSpec)`** — wraps a flow body (an async generator) in the workflow timeout and the shared exception boundary. `GuardSpec` parameterizes the per-flow differences that MUST stay different: cleanup policy (`expand`'s exception path clears the lock but does *not* mark the session failed, while its timeout path does both), timeout message, and an optional `map_exception` hook (used by `discover` to classify TaskGroup failures into client-safe typed errors).
- **`collect_token_usage(plugin)`** / **`error_events(...)`** — the token-usage close-out and the standard `WorkflowError` + `WorkflowComplete` terminal pair.

Runner **construction** deliberately stays in the calling module (`WorkflowExecutor._build_runner` resolves the module-level `Runner` name at call time), preserving the historical test seams — unit tests keep monkeypatching `core.executor.Runner` / `core.place_to_book.Runner` unchanged.

The timeout context wraps the *iteration* of the body, so consumer time (a slow SSE client between events) still counts against the workflow timeout — identical to the historical inline placement.

### Files Affected

- `core/run_harness.py` (new), `core/executor.py` (flows rewritten on the harness), `core/place_to_book.py` (pump + shared `_normalize_text` from `core/extraction.py`)
- `tests/unit/test_run_harness.py` (harness contract), `tests/unit/test_golden_stream.py` (SSE wire-contract snapshot for discover→compose — the migration safety net)

Note: the grounding *filters* in `core/extraction.py` and `core/place_to_book.py` remain intentionally separate — they have different fail-open contracts (all-dropped is a valid honest not-found for place→book, but a fail-open trigger for book recommendations).

### Trade-offs

**Benefits:** One place to change how we drive ADK (the prerequisite for the ADK 2.x migration); a new flow needs ~30 lines of scaffold instead of ~150; the error-cleanup policy of each flow is now explicit data (`GuardSpec`) instead of five hand-maintained try/except blocks; behavior-frozen — the full pre-existing unit suite passed unmodified.

**Costs:** Flow bodies are nested async generators (one more level of indirection when reading a single flow top-to-bottom); per-flow error policy lives in a spec object rather than inline try/except, so a reader must know the harness contract.

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
| **Gateway secret (ADR #10)** | Blocks direct access when deployed behind a gateway; fail-closed X-User-ID identity | Health endpoint intentionally excluded; leave secret empty for standalone dev |
| **Local-atmosphere mode** | Lets readers feel a book without traveling; reuses pipeline pattern | Separate endpoint to maintain; LLM-side radius enforcement only |
| **Persistent versioned cache (ADR #15)** | Repeat lookups free + instant; deploy-safe self-invalidation | Real-API seam test; bounded disk growth |
| **Cross-job place_key (ADR #16)** | Same real place recognized across book jobs | Missed combine possible (never a wrong one) |
| **Harness-first CI (ADR #17)** | "Done" is mechanical: locks, ratchet, audit, pin | Lock/relock discipline on every dep change |
| **Sentry, no-tracing + scrubbed (ADR #18)** | Prod exceptions/logs visible with PII scrubbed | Logs/metrics are opt-out; no perf tracing (Langfuse owns tracing) |
| **G5-gated single-box deploy (ADR #19)** | Cheap, deliberate, port-22-closed-at-rest deploys | Single box = no horizontal redundancy |
| **Place→book + server-derived grounding (ADR #20)** | Reverse product flow; grounding labels computed, not self-reported | Two LLM hops; separate endpoint |
| **Abuse/ops hardening (ADR #21)** | Token spend protected; bounded stores; honest tone; per-branch traces | More knobs to configure |
| **Shared run harness (ADR #22)** | One ADK-facing scaffold for all flows; explicit per-flow error policy | Flow bodies are nested generators; policy lives in `GuardSpec`, not inline |

These patterns work together to create a **reliable, performant, and user-friendly** multi-agent system for generating literary travel itineraries.
