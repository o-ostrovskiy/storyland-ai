"""
Session service factory.

Provides factory functions for creating session services with SQLite or
in-memory backends.
"""

import os
from typing import Optional

from google.adk.sessions import InMemorySessionService, DatabaseSessionService


def create_session_service(
    connection_string: Optional[str] = None, use_database: bool = False
):
    """
    Factory for creating the appropriate session service.

    Args:
        connection_string: Optional database URL (e.g., "sqlite:///sessions.db")
                          If not provided, defaults to "sqlite:///storyland_sessions.db"
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
        ...     connection_string="sqlite:///custom.db",
        ...     use_database=True
        ... )

        # PostgreSQL (requires psycopg2)
        >>> session_service = create_session_service(
        ...     connection_string="postgresql://user:pass@localhost/db",
        ...     use_database=True
        ... )
    """
    if use_database:
        # Production: Database-backed session service
        if not connection_string:
            # Default to SQLite in the current directory
            connection_string = "sqlite:///storyland_sessions.db"

        print(f"Using DatabaseSessionService with: {connection_string}")
        return DatabaseSessionService(db_url=connection_string)
    else:
        # Development: In-memory session service
        print("Using InMemorySessionService (not persistent)")
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
        # DATABASE_URL=sqlite:///storyland_sessions.db
        # USE_DATABASE=true

        >>> session_service = create_session_service_from_env()
    """
    use_database = os.getenv("USE_DATABASE", "false").lower() == "true"
    connection_string = os.getenv("DATABASE_URL")

    return create_session_service(
        connection_string=connection_string, use_database=use_database
    )
