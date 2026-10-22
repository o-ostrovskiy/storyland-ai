"""
Session service factory.

Provides factory functions for creating session services with SQLite or
in-memory backends.
"""

import asyncio
import os
from typing import Any, Awaitable, Optional, TypeVar

from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from common.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CancellationSafeDatabaseSessionService(DatabaseSessionService):
    """``DatabaseSessionService`` whose DB work survives a cancelled caller.

    A client dropping the SSE stream makes sse_starlette cancel its anyio task
    group. anyio re-delivers that cancellation at EVERY later ``await`` in the
    scope, so ADK's own unwind (``rollback()``, then the session's ``close()``)
    is cancelled too and the pooled connection is never checked back in:
    the GC "non-checked-in connection" and ``no active connection`` errors
    (storyland-ai-private#11).

    Each DB call therefore runs as its own task under ``asyncio.shield``. The
    caller still gets ``CancelledError`` at once (nothing waits on the DB), and
    the task, outside the cancelled scope, finishes and returns its connection.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Strong refs: the loop only holds tasks weakly, and a shielded task
        # whose caller has gone could otherwise be collected mid-query.
        self._inflight: set[asyncio.Task] = set()

    async def _shielded(self, op: str, coro: Awaitable[T]) -> T:
        task = asyncio.ensure_future(coro)
        self._inflight.add(task)
        task.add_done_callback(lambda t: self._finish(op, t))
        return await asyncio.shield(task)

    def _finish(self, op: str, task: asyncio.Task) -> None:
        self._inflight.discard(task)
        # A shield whose caller was cancelled drops the task's exception
        # silently; log it so a failure after a disconnect stays visible.
        if not task.cancelled() and task.exception() is not None:
            logger.warning(
                "session_db_op_failed_after_caller_left",
                op=op,
                error=repr(task.exception()),
            )

    async def create_session(self, *args: Any, **kwargs: Any):
        return await self._shielded("create_session", super().create_session(*args, **kwargs))

    async def get_session(self, *args: Any, **kwargs: Any):
        return await self._shielded("get_session", super().get_session(*args, **kwargs))

    async def list_sessions(self, *args: Any, **kwargs: Any):
        return await self._shielded("list_sessions", super().list_sessions(*args, **kwargs))

    async def delete_session(self, *args: Any, **kwargs: Any):
        return await self._shielded("delete_session", super().delete_session(*args, **kwargs))

    async def append_event(self, *args: Any, **kwargs: Any):
        return await self._shielded("append_event", super().append_event(*args, **kwargs))

    async def close(self) -> None:
        # Let in-flight DB work return its connections before the engine goes.
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
        await super().close()


def create_session_service(
    connection_string: Optional[str] = None, use_database: bool = False
):
    """
    Factory for creating the appropriate session service.

    Args:
        connection_string: Optional database URL (e.g., "sqlite+aiosqlite:///sessions.db")
                          If not provided, defaults to "sqlite+aiosqlite:///storyland_sessions_v2.db"
                          Note: This is passed as 'db_url' to DatabaseSessionService
        use_database: If True, use DatabaseSessionService; otherwise InMemorySessionService

    Returns:
        Session service instance (InMemorySessionService or DatabaseSessionService)

    Examples:
        # Development: In-memory (default)
        >>> session_service = create_session_service()

        # Production: SQLite database
        >>> session_service = create_session_service(use_database=True)

        # Custom SQLite database
        >>> session_service = create_session_service(
        ...     connection_string="sqlite+aiosqlite:///custom.db",
        ...     use_database=True
        ... )

        # PostgreSQL (requires psycopg2)
        >>> session_service = create_session_service(
        ...     connection_string="postgresql://user:pass@localhost/db",
        ...     use_database=True
        ... )
    """
    if use_database:
        if not connection_string:
            # Default to SQLite with async driver in the current directory.
            # "_v2" suffix: ADK 2.x writes a session schema incompatible with
            # the 1.x one, and the migration decision was a FRESH DB at
            # cutover (jobs are minutes-long; the retention sweeper prunes
            # anyway). A new filename guarantees 2.x never opens a 1.x file;
            # the old storyland_sessions.db is deleted manually post-deploy.
            connection_string = "sqlite+aiosqlite:///storyland_sessions_v2.db"

        logger.info("session_service_created", type="database", connection_string=connection_string)
        return CancellationSafeDatabaseSessionService(db_url=connection_string)
    else:
        logger.info("session_service_created", type="in_memory", persistent=False)
        return InMemorySessionService()


def create_session_service_from_env():
    """
    Create session service based on environment variables.

    Reads from:
        - DATABASE_URL: Connection string for database
        - USE_DATABASE: "true" or "false" to enable database sessions

    Returns:
        Session service instance

    Examples:
        # In .env file:
        # DATABASE_URL=sqlite+aiosqlite:///storyland_sessions_v2.db
        # USE_DATABASE=true

        >>> session_service = create_session_service_from_env()
    """
    use_database = os.getenv("USE_DATABASE", "false").lower() == "true"
    connection_string = os.getenv("DATABASE_URL")

    return create_session_service(
        connection_string=connection_string, use_database=use_database
    )
