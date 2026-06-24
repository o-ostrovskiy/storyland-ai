"""
FastAPI application factory.

Creates the FastAPI app with lifespan management, CORS, and router inclusion.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import initialize, shutdown
from api.routes import router, system_router
from services.session_retention import SessionSweeper


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown.

    Also starts the bounded session-retention sweep so the in-memory /
    SQLite session store does not grow unbounded on the single box (evicts
    discover->compose job state older than SESSION_TTL_SECONDS; disabled when
    SESSION_TTL_SECONDS=0).
    """
    state = await initialize()

    def _live_session_services():
        # The executor's store is the discover->compose leak source. The
        # place->book resolver keeps its own isolated in-memory store, created
        # lazily on first request - include it once it exists.
        services = [state.executor.session_service]
        import api.dependencies as deps

        resolver = getattr(deps, "_place_to_book_resolver", None)
        if resolver is not None:
            svc = getattr(resolver, "_session_service", None)
            if svc is not None:
                services.append(svc)
        return services

    sweeper = SessionSweeper(
        services=_live_session_services,
        ttl_seconds=state.config.session_ttl_seconds,
        max_entries=state.config.session_max_entries,
        interval_seconds=state.config.session_sweep_interval_seconds,
    )
    sweeper.start()
    app.state.session_sweeper = sweeper
    try:
        yield
    finally:
        await sweeper.stop()
        await shutdown()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="StoryLand AI API",
        description=(
            "Transform books into literary travel itineraries using AI agents.\n\n"
            "## Two-Step Flow\n\n"
            "1. **POST /api/v1/itinerary/discover** — Runs book search + location discovery "
            "(phases 1-2), streams SSE progress events, returns travel regions\n"
            "2. **POST /api/v1/itinerary/{job_id}/compose** — Accepts selected region IDs, "
            "runs itinerary composition (phase 3), streams final itinerary\n\n"
            "## SSE Event Types\n\n"
            "| Event | When | Payload |\n"
            "|-------|------|---------|\n"
            "| `progress` | During any phase | `{phase, step, detail}` |\n"
            "| `metadata` | After book search | `{book_title, author, ...}` |\n"
            "| `regions` | After discovery | `{job_id, regions, analysis_note}` |\n"
            "| `itinerary` | After composition | `{itinerary: {...}}` |\n"
            "| `error` | On failure | `{message, error_type, phase}` |\n"
            "| `done` | Stream complete | `{job_id, token_usage}` |\n"
        ),
        version="0.1.0",
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "itinerary",
                "description": "Itinerary generation endpoints (SSE streaming + status)",
            },
            {
                "name": "system",
                "description": "Health check and system status",
            },
        ],
    )

    # CORS configuration
    cors_origins = os.getenv("CORS_ORIGINS", "*")
    origins = [o.strip() for o in cors_origins.split(",")]

    # Browsers reject Access-Control-Allow-Origin: * with credentials.
    # Only enable credentials when specific origins are configured.
    allow_credentials = origins != ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check — no gateway secret required (probes, standalone testing)
    app.include_router(system_router, prefix="/api/v1")
    # Itinerary endpoints — gateway secret enforced
    app.include_router(router, prefix="/api/v1")

    return app
