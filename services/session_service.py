"""
Session service factory.

Provides factory functions for creating session services with SQLite or
in-memory backends.
"""

import os
from typing import Optional

from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from common.logging import get_logger

logger = get_logger(__name__)


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
        return DatabaseSessionService(db_url=connection_string)
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
