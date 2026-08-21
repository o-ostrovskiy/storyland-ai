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
# ^ historical (ADK 1.x). Since the graph rewrite (ADR #24) the pair is wired
#   as Workflow edges: (START, researcher, formatter) — same two-stage contract.
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

> **Implementation superseded (2026-07, ADR #24):** the fan-out is now explicit graph edges — `book_context_formatter → (city, landmark, author) → JoinNode → region_analyzer`. The decision (parallelize the three independent branches) is unchanged; only the mechanism moved from the template agent to the graph runtime.

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

*Decided 2026-06-27 → 2026-07-11 (MYS-399); entry backfilled 2026-07-19 — the Patterns table row existed but the section was never written. Default flipped to fail-closed 2026-07-19 (July 2026 architecture review).*

### Decision
When deployed behind the backend gateway, require a shared secret on every itinerary endpoint and take end-user identity **only** from the trusted `X-User-ID` header the gateway sets after JWT validation. Fail closed.

### Implementation
- `api/dependencies.py` — `verify_gateway_secret`: when `INTERNAL_API_SECRET` is set, all itinerary endpoints require a matching `X-Internal-Secret` header (`/health` stays open). The secret is compared as **bytes**, constant-time.
- Identity: the `X-User-ID` header only — the service trusts nothing else from the request. Missing header → 403 (except standalone dev, where identity falls back to the shared `dev_user`).
- `REQUIRE_GATEWAY_SECRET` (`common/config.py`) refuses to **start** with an empty secret — closing the foot-gun where an empty secret silently accepts everyone and `X-User-ID` becomes a forgeable identity.
- **Default flip (2026-07-19):** `REQUIRE_GATEWAY_SECRET` originally defaulted to `false` — enforcement was opt-in, and a fresh deploy that never set the secret came up open with only the `gateway_auth_disabled` boot warning. The July 2026 architecture review called this the cheapest real risk on the board, so the default is now **`true`**: running open requires an explicit `REQUIRE_GATEWAY_SECRET=false` (standalone/local dev; `.env.example` ships that opt-out). Prod is unaffected — `.env.prod` on the deploy box already sets `INTERNAL_API_SECRET`, and the switch only bites when the secret is empty.

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
- **Hash-locked deps:** `requirements.lock` / `requirements-dev.lock` from `uv pip compile --universal --generate-hashes`; the unit workflow (`codex.yml`) installs `--require-hashes` and fails if `make lock` would change the committed lock. `integration-tests.yml` installs from the same dev lock as of the ADK 2 lift (the previously documented live-resolution gap bit exactly as predicted: a fresh resolve drifted httpx/vcrpy and broke VCR replay with a method-case mismatch the locked env couldn't reproduce).
- **Coverage ratchet:** `fail_under = 74` in `pyproject.toml` (76% baseline; bare `--cov` honoring `[tool.coverage.run]`). Ratchet, not target — raise, never lower.
- **Audit gate:** `pip-audit --strict --require-hashes` on the prod lock; ignores are inline with justification + ticket.
- **ADK pin:** `google-adk[db]>=2.4.0,<3` in `pyproject.toml` + a CI guard asserting the **locked** resolution stays 2.x (prevents a non-reproducible jump to the next major). Historical: this was `google-adk[eval]>=1.33.0,<2` guarded at 1.x until the ADK 2 lift (ADR #23), which moved the pin one major and swapped the extra — `[db]` restores the `DatabaseSessionService` deps that are no longer core on 2.x, and nothing imports `google.adk.evaluation`. The mechanism is unchanged: the guard tracks the tested major, it does not forbid a particular one.

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

Five smaller decisions, recorded together:

- **Rate limiting + concurrency cap** (2026-06-23): per-identity request rate limit and in-flight concurrency cap on discovery endpoints (`api/ratelimit.py`); input-size bounds return 422 **before** any Gemini tokens are spent.
- **Bounded session retention** (2026-06-24): a periodic retention sweep (`services/session_retention.py`) bounds the in-memory/SQLite session store — extends ADR #6, which predates it.
- **Recommendation-tone guardrail** (2026-06-28): rec explanations are fit-only and never grade the reader (`core/guardrails/tone_guardrail.py`).
- **Langfuse per-branch scoping** (2026-07-17, MYS-398): generation/agent-stack/span state is scoped **per ParallelAgent branch** (`_branch_key()` in `plugins/langfuse_plugin.py`) — per-request plugin instances (ADR #8) were not enough under the ADR #3 parallel fan-out.
- **Langfuse trace-level I/O written explicitly** (2026-08-05, MYS-788): the plugin sets input/output on the trace as well as on its root span (`set_trace_io()` in `plugins/langfuse_plugin.py`), because Langfuse's session view and trace list render **trace**-level I/O — setting only the root span's made every production trace display as "This trace has no input or output" despite a complete span tree. Python SDK v4 dropped `update_current_trace()` and deprecates trace I/O in favour of the observations-first model, but `propagate_attributes()` carries correlating attributes only (user/session/tags/metadata), so `set_trace_io()` is the sole remaining writer for those two fields; revisit when those views read observations instead. Trace output is the run's final-response text, picked with the same rule as `core/run_harness.py` (last event where `is_final_response()` is true).

---

## 22. Shared Run Harness (Single ADK-Facing Scaffold for All Flows) (2026-07-19)

### Context

By mid-2026 `core/executor.py` had grown to ~1,700 lines: five endpoint flows (`discover`, `compose`, `local_atmosphere`, `expand`, `recommend_books`) plus the place→book resolver each carried a near-identical copy of the same scaffolding — Runner construction, the event-drain loop with agent→progress mapping, researcher-text capture, final-response tracking, the workflow timeout, and a four-branch `TimeoutError` / `CancelledError` / `Exception` boundary ending in `WorkflowError` + `WorkflowComplete`. Every new flow copy-pasted ~150 lines, and any change to how we drive ADK (e.g. the planned ADK 2.x migration) had to be applied and verified six times.

### Decision

Extract the ADK-facing scaffolding into `core/run_harness.py` as four small primitives, leaving all business logic (guards, caching, merging, grounding filters) in the flows:

- **`pump_events(runner, ...)`** — drains `runner.run_async`, yields at most one `ProgressEvent` per agent (from a per-flow `agent_steps` map), and optionally fills a **`RunCapture`** with per-author text (grounding post-validation input) and the final response.
- **`run_guarded(body, GuardSpec)`** — wraps a flow body (an async generator) in the workflow timeout and the shared exception boundary. `GuardSpec` parameterizes the per-flow differences that MUST stay different: cleanup policy (`expand`'s exception path clears the lock but does *not* mark the session failed, while its timeout path does both), timeout message, and an optional `map_exception` hook that classifies TaskGroup/generic failures into client-safe typed errors via `classify_discovery_failure` (MYS-400, 2026-07-26: wired into all five flows through a shared `_map_phase_exception` helper — originally only `discover` set it, so the other four leaked a raw `str(e)`/`ExceptionGroup` string to the client).
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

## 23. ADK 2.x Migration (Template-First, Graph Runtime Deferred to a Follow-Up PR) (2026-07-19)

### Context

The service ran on `google-adk 1.x` (pinned `<2`) for eight months. ADK 2.0 (GA 2026-05) replaced the hierarchical agent executor with a graph-based workflow runtime and shipped breaking changes (incompatible session schema, removed `Gemini.api_key`, `[db]` extra required for `DatabaseSessionService`). The `<2` cap had two growing costs: five starlette CVE pip-audit ignores (ADK 1.x capped `starlette<1`) and a private-attribute coupling in the Langfuse plugin.

### Decision

Migrate in stages, eval-gated against a two-run pre-migration baseline (storyland_eval pooled avg ≈4.33/5, books_v1 pooled ≈3.14/5 on `gemini-2.5-flash-lite`; gates calibrated to measured judge noise — ±0.17 and ±0.40 respectively — not a flat percentage):

1. **PR 1 — run harness** (ADR #22): all ADK-facing scaffolding behind one seam, behavior-frozen.
2. **PR 2 — this lift**: `google-adk[db]>=2.4.0,<3` on the still-supported `SequentialAgent`/`ParallelAgent` template workflows (deprecated in 2.x but functional), so the dependency jump and the orchestration rewrite are separately bisectable. Model stays pinned to `gemini-2.5-flash-lite` so eval deltas are attributable to ADK alone. Gate result: PASS (storyland_eval 4.33 — dead-on baseline; books_v1 3.20 — above baseline; per-generation usage audit: 216/216 generations on both sides with non-zero recorded usage).
3. **PR 3 — graph rewrite** (follow-up): re-express the workflows on `google.adk.workflow` (`Workflow`/`Node`/`Edge`), keeping the two-endpoint HTTP contract. Native HITL pause/resume is deliberately deferred — the discover→select→compose split stays as-is.
4. **PR 4 — model upgrade** (follow-up): single `gemini-3-flash` + cassette re-record.
   **Outcome (2026-07, PR #228):** shipped as **`gemini-3.1-flash-lite`** ($0.25/$1.50 per 1M tok), not `gemini-3-flash` — the like-for-like modern replacement for the deprecated 2.x lite tier, chosen over the pricier 3-flash/3.5-flash tiers; eval-gated with the both-shapes gating (see `DEFAULT_MODEL_NAME` in `common/config.py`). The eval **judge** deliberately stays on `gemini-2.5-flash-lite` (`evaluation/tools/llm_scorer.py`) so scores remain comparable across system-model lifts. *Prod drift note (observed 2026-07-20 via Langfuse, `production` env):* the deployed box's `.env.prod` pins `MODEL_NAME=gemini-3.1-flash-lite-preview` — the preview alias, presumably set before the stable alias existed. Deploys never rewrite `.env.prod`, so aligning prod to the stable `gemini-3.1-flash-lite` requires editing that file on the box. (Cost tracking is unaffected: the pricing lookup matches by substring, so the `-preview` alias bills at 3.1-flash-lite rates.)

**Key 2.x facts this migration verified against the real package** (spike, `google-adk==2.4.0`/`2.5.0`):
- `tools` + `output_schema` may now coexist on one `LlmAgent` — ADR #2's researcher→formatter split is no longer *forced* by the framework. Collapsing pairs is a separate, eval-gated experiment; the anti-hallucination rationale stands until disproven per-pipeline.
- `Gemini(api_key=...)` is **silently dropped** on 2.x (pydantic ignores it; auth falls back to env) — the key must go through `client_kwargs={"api_key": ...}`. Both construction sites fixed.
- Session storage: **fresh DB at cutover** (default file renamed to `storyland_sessions_v2.db` so 2.x never opens a 1.x file). Jobs are minutes-long and the retention sweeper prunes anyway, so no data migration. The v-current schema keeps `sessions.update_time` and exposes the engine as `db_engine`, so the retention sweeper's raw-SQL prune still works — now proven by `tests/integration/test_session_retention_db.py` (previously the prune was fail-open and untested against a real store).
- The ADR #11 `append_event(state_delta)` persistence contract holds unchanged on both backends.
- All eight `LangfusePlugin` hooks survive; `CallbackContext.branch` is now **public**, removing the private `_invocation_context` coupling (kept only as a fallback). Token-usage extraction treats absent/None `usage_metadata` as missing (never a zero-token success) and logs a WARNING so cost tracking can't silently zero.
- ADK 2.x auto-retries a tool only when its exception propagates; `tools/preferences.py` deliberately stays fail-open (documented in place).
- starlette resolves to >=1.3 → the five PYSEC ignores were removed from `codex.yml` and `Makefile`; the CI pin guard (ADR #17) now asserts a 2.x resolution.
- `[eval]` extra dropped (nothing imports `google.adk.evaluation`); `[db]` + explicit `greenlet` added.

### Trade-offs

**Benefits:** supported dependency line; five CVE ignores retired; public-API-only plugin; per-stage bisectability with the golden-stream SSE snapshot + eval gates proving "no functionality broken."

**Costs:** template workflows emit deprecation warnings until PR 3; a fresh session DB drops in-flight jobs at the cutover deploy (accepted — deploy in a quiet window); cassettes may need one extra re-record if genai 2.x changed the wire surface before PR 4's planned re-record.

---

## 24. Graph-Runtime Orchestration + the Primary Flow Named `book_to_place` (2026-07-19)

### Context

PR 2 of the ADK migration (ADR #23) ran on the deprecated-but-functional `SequentialAgent`/`ParallelAgent` templates. Separately, the codebase had a naming asymmetry: the reverse direction was a named capability (`place_to_book`) while the primary direction — a book in, real places out — was the unnamed default ("the workflow"), which is why the harness had grown two dialects for driving the same runner.

### Decision

**All six workflows are explicit `google.adk.workflow.Workflow` graphs** built in `agents/orchestrator.py`; the `agents/create_*_agents` factories return plain `(researcher, formatter)` `LlmAgent` pairs and own NO composition. Zero template agents remain (asserted: constructing every workflow emits no Sequential/ParallelAgent deprecation warning). `Runner` is driven via `node=` (one seam: `_build_runner` in the executor and the resolver).

**The primary flow is named `book_to_place`**, the symmetric counterpart of `place_to_book`: phase workflows `book_to_place_discovery` and `book_to_place_composition` (factories `create_book_to_place_discovery_workflow` / `create_book_to_place_composition_workflow`). Executor methods keep their phase verbs (`discover`/`compose`) and the HTTP contract is byte-identical (golden-stream snapshot unchanged).

Discovery graph (the only non-linear one):

```
START → book_context_researcher → book_context_formatter
      → fan-out: city_r→city_f, landmark_r→landmark_f, author_r→author_f
      → JoinNode(discovery_join) → region_analyzer
```

The JoinNode is load-bearing: a plain node fires on ANY predecessor; region_analyzer must wait for ALL three branches.

**Graph semantics are pinned by tests, not assumed** (`tests/unit/test_graph_workflows.py` drives real Workflows through a real Runner with a scripted model): (1) downstream nodes see upstream responses in their request contents — the ADR #2 researcher→formatter contract; (2) `output_key` still writes session state — ADR #5; (3) events keep per-agent authorship — the progress-mapping contract; (4) JoinNode gates the fan-in to exactly one run after all branches. `tests/unit/test_agents.py` pins the graph shapes (edges, join, terminals).

Kept deliberately: the researcher/formatter split (ADK 2.x would allow tools + output_schema on one agent; collapsing is a separate eval-gated experiment), the ADR #11 `append_event` persistence pattern (verified compatible), the two-endpoint HITL facade (native pause/resume still deferred).

### Trade-offs

**Benefits:** off the deprecated path before removal; composition is explicit data (edges) instead of nested agent trees; dead progress-map entries for container agents removed; the primary capability is greppable by name; symmetric naming ends the "named reverse, unnamed default" asymmetry.

**Costs:** factory renames touch the executor's imports and monkeypatch targets (mechanical); Langfuse span hierarchy shows graph node names (dashboards keyed on `discovery_workflow_invocation` see `book_to_place_discovery_invocation` after this change).

---

## 25. Per-Session Lock for `expand()`/`recommend_books()` Concurrency Guards + Merge Re-validation (2026-07-26)

### Context

Tech Radar's AI review (2026-07-09, MYS-401) flagged that `expand()`'s and `recommend_books()`'s "already in progress" guards were check-then-set with an `await` boundary in between: each read `session.state` (a snapshot copy — `InMemorySessionService.get_session` returns `_copy_session(...)`, never a live reference) captured once near the top of the call, then later wrote the flag via a separate `append_event`. Two overlapping requests for the same `job_id` (a double-clicked suggestion chip, or an FE retry) could both read the flag `False` from their own independent snapshot before either's write landed — both proceed, double-billing Gemini, and whichever `append_event` lands last silently overwrites the other's merge (`expansion_count` ends up a lost update instead of the correct total). A second, independent finding on the same ticket: `expand()`'s itinerary merge mutates raw dicts and persists the result to `FINAL_ITINERARY` with no schema check, so a formatter quirk anywhere upstream (a stop missing a required field, an empty `cities` list) would silently persist a structurally-degraded itinerary that `/status` re-serves forever after. A third, low-severity finding: target-city selection for a suggestion chip used a plain substring test, so a trip city literally named "York" would match an `action_prompt` mentioning "New York".

### Decision

- **Atomic guard, single-snapshot admission (r4).** A per-`job_id` lock registry (`WorkflowExecutor._session_locks`) and `_get_session_lock(job_id)` serialize both guards. Every admission decision — job lookup, failed/not-ready state, the hard cap, `action_id` validity, and the in-progress flag itself — is made from **one** session snapshot fetched **while holding the lock**; there is no separate pre-lock read left to go stale. (r1–r3 iterated toward this: r1 added the lock but only re-fetched to check the flag; r2 wrapped that re-fetch in the SSE error envelope; r3 adopted the re-fetch for the merge base but still ran admission checks against the stale outer capture, which Codex correctly flagged as reopening the double-charge finding one line later.) In-process only: sufficient because this service runs a single uvicorn worker (`Dockerfile` has no `--workers`); a multi-process deployment would need a durable compare-and-set instead (the same coupling MYS-176/MYS-168/MYS-169 already flagged for this code path).
- **Bounded lock registry, eviction strictly before insertion (r4, re-owned r6).** `_session_locks` is an `OrderedDict`, not an unbounded `defaultdict`. `_get_session_lock` moves an entry to MRU on access and evicts least-recently-used, currently-unheld entries once the registry reaches `_SESSION_LOCK_REGISTRY_CAP` — never evicting a lock that's held or has a pending waiter, so a genuinely-overlapping second caller for the same `job_id` can never be handed a different `Lock` instance. r4's version ran eviction *after* inserting the new entry: usually harmless (a brand-new entry sorts last), but not structural — if every pre-existing entry happened to be held or waited-on at cap, the eviction loop would walk past all of them and delete the entry the same call had just created, handing the caller a `Lock` no longer in the registry. r6 reorders this: eviction now runs *before* the new entry is created, so it is never a candidate — the entry simply doesn't exist yet when eviction runs, independent of how many older entries are evictable. This is the fix Eng Lead review asked for after r5, framed as re-owning the guard rather than a sixth incremental patch (see PR discussion); no further hardening was layered onto this diff beyond it.
- **Merge re-validation.** Before `expand()` persists the merged `FINAL_ITINERARY`, it now runs the merge through `validate_trip_itinerary` (the same `TripItinerary.model_validate` helper `core/extraction.py` already uses elsewhere). On failure, the merge is **not** persisted — `FINAL_ITINERARY` stays whatever it already was, so `/status` keeps re-serving the last-known-good result instead of a newly-degraded one — and the client gets a typed `MergeValidationError` instead of a silently-corrupted success.
- **Word-boundary city match, hyphen-aware (r4).** `expand()`'s target-city scan uses `_matches_city_as_standalone_word(city_name, action_prompt, other_city_names)` — a regex word-boundary match that rejects a match only when the preceding word forms another trip city (not any capitalized word — r2). The boundary itself is plain `\w` adjacency, not `[\w-]` (r4): a hyphen already ends a word on its own, so a city immediately followed by a hyphenated compound ("Bath-based") is no longer misrouted to `cities[0]`. Moved to `core/extraction.py` (r4) so `_drop_suggestions_naming_removed_cities` (MYS-660) can call the exact same predicate `expand()` uses, instead of the plain substring test that used to let the two silently drift apart. Not a gazetteer; documented as a heuristic, tradeoffs included, in its own docstring.

### Files Affected

- `core/extraction.py` (`_matches_city_as_standalone_word` — moved here from `core/executor.py`, r4; `_drop_suggestions_naming_removed_cities` now calls it instead of a plain substring test)
- `core/executor.py` (`_session_locks` — now a bounded `OrderedDict` — + `_get_session_lock`/`_evict_stale_session_locks` on `WorkflowExecutor`; `expand()` and `recommend_books()` guards restructured to single-snapshot admission; `expand()`'s merge persist path)
- `tests/unit/test_executor_expansion_concurrency.py` (interleaving/merge-revalidation/word-boundary/hyphen/registry-bound cases)
- `tests/unit/test_core.py` (`_drop_suggestions_naming_removed_cities` cases proving it now agrees with `expand()`'s real resolution)

### Trade-offs

**Benefits:** no more double-billed Gemini calls or lost-update `expansion_count` on a double-clicked chip, including the delayed-replay-of-a-consumed-chip case; a degraded merge can never reach `/status`; a same-name-substring city mismatch is much rarer, including hyphenated-compound prompts; the lock registry no longer grows unbounded; the removed-city suggestion filter can no longer disagree with `expand()`'s own resolution.

**Costs:** the word-boundary heuristic is capitalization-dependent and can still mismatch on inconsistently-cased `action_prompt` text; a city that's the tail of a genuinely hyphenated proper noun ("Winston-Salem") can now match a bare search for the tail word alone (r4, documented in the helper's own docstring); the lock registry's eviction is best-effort LRU, not a hard guarantee, and relies on `asyncio.Lock`'s private `_waiters` attribute; if every existing entry is held/waited at cap, a newly-inserted entry can leave the registry one above `_SESSION_LOCK_REGISTRY_CAP` until a later call finds room to evict (r6) — an accepted, bounded, and rare relaxation, never a correctness issue.

---

## 26. Search-Grounding Receipts + Fail-Closed Discovery Enforcement (2026-08-12)

### Context

Researcher agents are configured with Gemini's built-in `google_search` and their prompts instruct them to use it — `city_researcher` carries an explicit "You MUST call google_search at least once" clause. Nothing verified that it happened. A researcher answering from model memory produced output structurally identical to a grounded one, and every downstream stage treated it as researched fact; one observed run returned an author site named `Personal Office`. Measured across the full evalset (18 cases, 72 researcher calls), **46% of researcher calls skipped search** — `book_context_researcher`, the first node in the chain, at 83%. The skipping is stochastic and near-universal, not an edge case tied to fictional settings (an earlier framing this ticket's own measurement refuted), and its root cause is the task framing in `build_discovery_prompt` (MYS-846): reframing the request as research drops the skip rate to 0/12, while appending an explicit search instruction to the real prompt moves nothing (6/12 either way).

### Decision

- **Receipts, not inference.** `LangfusePlugin` keeps a per-agent ledger written in `after_model_callback` *before* the Langfuse `enabled` gate, so grounding remains answerable on a deploy with no credentials. `searched_agents()` reports only agents **positively observed** calling `google_search`; `unsearched_agents()` is deliberately three-valued (an agent that never ran is not a skip); `observed_any()` reports whether the ledger saw anything at all. Receipts are read from **both** channels a response can carry them on — reading only `grounding_metadata` made every server-side-invocation agent read as "never searched".
- **Fail closed on unverified evidence, not on missing evidence.** Payload keys whose researcher produced no receipt are written to session state as `unverified_discovery` and excluded from `grounding_research_text`, so the existing `downgrade_ungrounded_match_types` guard stops finding evidence for them. The load-bearing asymmetry: an empty haystack means *"cannot prove anything ungrounded — change nothing"* on the local-atmosphere path, and *"nothing here can possibly be grounded — demote everything"* when discovery ran and every present payload is unverified. Both arrive at the guard identically, so the caller passes `all_discovery_unverified` to tell them apart. Without it the guard is weakest precisely when the run is least trustworthy.
- **An unobserved receipt is not a verification, and enforcement is scoped to the payloads that EXIST (r4).** The enforcement path originally asked `unsearched_agents()`, whose three-valued rule omits an agent the ledger never saw — so a researcher whose receipts were missed was left out of `unverified_discovery` and went on vouching for `literal`/`historical` claims, while the metric path (`searched_agents`, positive-only) already refused to call the same researcher grounded. Two paths, two answers, and the permissive one reached the user. The reconciliation is *not* to disqualify every **expected** payload — that would re-break the case the three-valued rule exists for — but every **present** one: a payload's existence is proof its researcher ran, and it is the only such proof available where the ledger being empty is the state under test. An agent that never ran contributes no payload and is untouched; an agent that ran and left no receipt is unverified, whether it skipped or the seam broke. Those two are indistinguishable from here and, for "may this vouch for a real place", identical.
- **"Everything was verified" is always a measurement, never an inference from an absence.** This shape had to be closed at three separate sites in one feature, and the third only surfaced in review round 3: the ledger (`searched_agents` is positive-only, because `total - len(unsearched)` reported a clean 4/4 on an empty ledger); the enforcement path (`observed_any()` + a `discovery_search_ledger_empty` warning, because a broken observation seam and a clean run both returned `[]`); and session state itself (`discovery_verification_ran`, because writing the verdict only `if unverified:` left a clean run and a run where the pass never happened byte-identical). **The verdict is therefore written unconditionally, empty list included, and the cache bundle replays it by type rather than truthiness** — an empty list is a verdict; a missing key is silence.
- **The trigger is documented rather than tightened.** `all_discovery_unverified` is true when *at least one* discovery payload is non-empty and every non-empty one is unverified — not, as earlier descriptions claimed, only when all four researchers skipped. Empty payloads filter out as absent and `all()` over one element is `True`, so a lone unsearched payload blanket-demotes the itinerary. That is correct: the lone payload is excluded from the haystack, so requiring the full researcher set would restore the fail-open at a different maximum.
- **Report, don't gate, in the eval.** Skips are near-universal today, so a hard eval gate would be red from day one and get switched off. `search_grounding` is reported beside the judge scores; the eval applies production's downgrade before scoring so it stops measuring claims the product would never ship.

### Files Affected

- `common/search_grounding.py` (new), `common/logging.py` (Sentry allowlist; query strings deliberately never forwarded — they embed the user-supplied book title)
- `plugins/langfuse_plugin.py` (the ledger), `core/place_to_book.py` (registers a credential-less `LangfusePlugin` so its researcher is not invisible to the ledger)
- `core/session_state.py` (`unverified_discovery`, `all_discovery_unverified`, `discovery_verification_ran`), `core/extraction.py` (`evidence_disqualified`), `core/executor.py` (verdict write + cache replay; cache key `v2` → `v3`)
- `evaluation/tools/run_scheduled_eval.py`

### Trade-offs

**Benefits:** a claim the product cannot support is demoted rather than shipped at full confidence; the skip rate is measurable per researcher instead of invisible; the guard's blind spot at its own maximum is closed; a broken observation seam is now loud rather than indistinguishable from a clean run.

**Costs:** a user-visible downgrade — stops that would previously have read `literal`/`historical` now read `vibe` with no source whenever their supporting researcher skipped, which at today's 46% rate is a substantial fraction of stops; **a wholesale instrumentation fault (an ADK callback rename, or an agent renamed without `RESEARCHER_PAYLOAD_KEYS` following) now demotes every strong claim on every itinerary rather than silently passing unverified ones** — a deliberate choice of direction, since the alternative is asserting `literal` with a `grounding_source` for places nobody checked, with `discovery_search_ledger_empty` and `test_payload_map_covers_every_researcher` as the compensating controls; one bounded cold-cache window for the `v3` key bump; `search_grounding_absent` fires for every tool-less formatter (roughly half of all model calls) and is INFO-level noise a Sentry health pass must learn to ignore; the Sentry allowlist widening is per-key across the whole codebase, so generic keys (`kind`, `total`, `agent`) now forward from anywhere — namespacing them is deferred to its own card. This does **not** fix the skipping; MYS-846 owns the cause.

---

## 27. Judge-Calibration Reads a UNION of the Legacy Runs API and the Experiments API (2026-08-20)

### Context

Langfuse retires the legacy dataset-run endpoints on **2026-11-16**. The obvious sequencing — move the writes and the reads together — was argued for on real evidence: `GET /experiments` returns **zero** runs today, so a reads-only cutover finds nothing and calibration goes dark.

That argument rules out a *switch*, not a *union*. Every run is either legacy-written or experiment-written and never both, so reading both is total at every moment of the migration, and the read side can therefore land first and alone.

Building it surfaced the constraint that decides the shape: **`ExperimentItem` exposes no dataset-item id.** Its `experiment_item_id` is run-scoped by Langfuse's data model — one per dataset item *per experiment* — while `select_candidates` caps generations per case on `(dataset, item_id)` so one evalset case cannot crowd the pack. Fed run-scoped ids, that cap silently stops capping: same key name, same type, no error.

### Decision

The dataset-item id travels on the **root span's metadata as `eval_id`**, written by all four eval tools (three already did; `run_scheduled_eval.py` was the gap), and the experiment leg reads it back and **raises when it is absent**. It never falls back to `experiment_item_id`.

🔴 **Correction (2026-08-21).** This entry originally justified the raise with *"the population is empty: zero experiment runs exist, so a leg that raises on absence is raising on a bug, never on history."* A live probe on 2026-08-20 found **ten** pre-existing experiment runs. The premise was false, and it was the entire argument.

The raise stands, on a different and weaker footing: the runs the probe read do carry `eval_id`, but no one has asserted that over all ten. So it is safe by **loudness**, not by an empty population — a build that stops is recoverable; a green pack quietly missing a leg is not. The distinction matters because the two sentences have different failure modes, and only the second one is true.

🟢 **Probe returned 2026-08-21** (live Langfuse, read-only, zero spend): **54** experiment runs, **248** items, **0** missing `eval_id` — every one carried in `experiment_item_metadata`, and no run hit the page bound. The raise is now safe on evidence over the whole population rather than on loudness alone. ⚠️ The probe also established a fact the code depends on and could not have derived: the **request** spelling of the metadata groups is the camelCase enum (`itemMetadata`, `experimentMetadata`) while the **response** exposes snake_case attributes — sending the snake_case spelling is a 400. `_EVAL_ID_GROUP_REQUEST_FIELDS` names that mapping so a narrowing of `_EXPERIMENT_ITEM_FIELDS` reds a row instead of silently reading a group nobody asked for.

Two supporting choices:

- **`from_start_time` is required by the endpoint and is deliberately wide.** The legacy path applies no time window at all — it selects the newest *N* runs by count — so there is no existing window to derive a floor from, and a floor derived from the oldest kept legacy run is only safe while that leg is saturated. Both endpoints return time-descending and the same count is applied after the union, so a floor that is never later than anything cannot filter a candidate.
- **De-duplication keys on `trace_id`, not on run name.** The two APIs' id spaces are disjoint — the same partition that makes the union total — so a name is the only cross-API handle a *run* has, and it is the wrong one: two genuinely different runs sharing a name would lose one silently. A generation is one trace whichever endpoint described it.

Two decisions from review, both about what the leg does when the data is not what it expects:

- **The data-contract signal has its own class, `CalibrationDataError(LookupError)`.** The caller must re-raise our fail-closed signal while still degrading on a transport fault, and stating that guard over bare `LookupError` over-reaches badly: it is the base class of `KeyError` and `IndexError`, so any incidental container miss inside the langfuse SDK became indistinguishable from a malformed write and aborted the whole build. *A guard stated over a base class inherits every meaning that class already had.*
- **A paged read that hits its page bound with a cursor outstanding raises (`CalibrationTruncatedError`) rather than returning what it has.** The bound guarantees termination; it is not permission to answer a different question. The first version returned the first 2000 items silently — and a test asserted that truncation as correct, so the suite pinned it.
- **A transport fault is scoped to the ONE run it happened on; a data-contract fault is not.** `_experiment_items` is split out so `collect_experiment_candidates` can skip a single unreadable run — naming it on stdout — and keep the runs already fetched, the way the legacy per-run loop always has. Previously one flaky read propagated out of the function and the caller's broad arm replaced the entire experiments leg with `[]`; after the legacy endpoint retires that dataset would contribute nothing. `CalibrationDataError` (and therefore `CalibrationTruncatedError`) is re-raised unchanged: *the scope of a degrade is a claim about the fault's subject, and a run-scoped catch must not acquire a dataset-scoped one.*

### Consequences

The read is additive, flagless and revertible, and it is correct before, during and after the write migration (PR2) and after the legacy leg is removed (PR3). The cost is that calibration now depends on a metadata key rather than a first-class API field, and that dependence is enforced by a raise rather than a degradation — a missing `eval_id` stops the pack instead of quietly skewing it.

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
| **Gateway secret (ADR #10)** | Blocks direct access when deployed behind a gateway; fail-closed X-User-ID identity; empty secret is a fatal boot error by default | Health endpoint intentionally excluded; standalone dev needs explicit `REQUIRE_GATEWAY_SECRET=false` |
| **Local-atmosphere mode** | Lets readers feel a book without traveling; reuses pipeline pattern | Separate endpoint to maintain; LLM-side radius enforcement only |
| **Persistent versioned cache (ADR #15)** | Repeat lookups free + instant; deploy-safe self-invalidation | Real-API seam test; bounded disk growth |
| **Cross-job place_key (ADR #16)** | Same real place recognized across book jobs | Missed combine possible (never a wrong one) |
| **Harness-first CI (ADR #17)** | "Done" is mechanical: locks, ratchet, audit, pin | Lock/relock discipline on every dep change |
| **Sentry, no-tracing + scrubbed (ADR #18)** | Prod exceptions/logs visible with PII scrubbed | Logs/metrics are opt-out; no perf tracing (Langfuse owns tracing) |
| **G5-gated single-box deploy (ADR #19)** | Cheap, deliberate, port-22-closed-at-rest deploys | Single box = no horizontal redundancy |
| **Place→book + server-derived grounding (ADR #20)** | Reverse product flow; grounding labels computed, not self-reported | Two LLM hops; separate endpoint |
| **Abuse/ops hardening (ADR #21)** | Token spend protected; bounded stores; honest tone; per-branch traces | More knobs to configure |
| **Shared run harness (ADR #22)** | One ADK-facing scaffold for all flows; explicit per-flow error policy | Flow bodies are nested generators; policy lives in `GuardSpec`, not inline |
| **ADK 2.x, template-first (ADR #23)** | Supported line; CVE ignores retired; public-API plugin; bisectable stages | Deprecation warnings until the graph rewrite; fresh session DB at cutover |
| **Graph workflows + `book_to_place` naming (ADR #24)** | Explicit edges, no deprecated templates; primary flow named; semantics test-pinned | Factory renames; Langfuse trace names change |
| **Per-session lock + merge re-validation (ADR #25)** | No double-billed Gemini calls on overlapping requests, incl. delayed chip replay; a degraded merge never reaches `/status`; lock registry now bounded | Word-boundary city match is a capitalization heuristic, not a gazetteer |
| **Search-grounding receipts + fail-closed discovery (ADR #26)** | A claim the product cannot support is demoted rather than shipped at full confidence; skip rate measurable per researcher; enforcement and the metric give one answer on an unobserved receipt | User-visible downgrade at today's 46% skip rate; a wholesale instrumentation fault demotes everything (loud, by design); one cold-cache window at the `v3` bump |
| **Union eval reads + `eval_id` case key (ADR #27)** | Read side migrates alone and stays total at every step of the Nov 16 cutoff; a run-scoped id can never silently disable the per-case cap; an incomplete read refuses rather than answering short | Calibration depends on a metadata key, not an API field; a missing `eval_id` or a truncated run stops the build rather than degrading it |

These patterns work together to create a **reliable, performant, and user-friendly** multi-agent system for generating literary travel itineraries.
