"""
FastAPI application factory.

Creates the FastAPI app with lifespan management, CORS, and router inclusion.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import initialize, shutdown
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown."""
    await initialize()
    yield
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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(router, prefix="/api/v1")

    return app
