"""
Memory service factory.

Provides factory functions for creating memory services with in-memory
keyword search backend.
"""

import os
from typing import Optional

from google.adk.memory import InMemoryMemoryService


def create_memory_service(use_in_memory: bool = True):
    """
    Factory for creating the appropriate memory service.

    Currently supports InMemoryMemoryService (keyword-based search).
    For production with semantic search, consider implementing a custom
    SQLite-backed memory service with embeddings.

    Args:
        use_in_memory: If True, use InMemoryMemoryService (default)

    Returns:
        Memory service instance (InMemoryMemoryService)

    Examples:
        # Development: In-memory keyword search
        >>> memory_service = create_memory_service()

        # After each session
        >>> await memory_service.add_session_to_memory(session)

        # Query memories
        >>> results = await memory_service.search_memory(
        ...     app_name="storyland",
        ...     user_id="user1",
        ...     query="travel preferences for France"
        ... )
    """
    if use_in_memory:
        print("Using InMemoryMemoryService (keyword search)")
        return InMemoryMemoryService()
    else:
        # Future: Add custom SQLite-backed memory service
        # For now, default to InMemoryMemoryService
        print("Using InMemoryMemoryService (keyword search)")
        return InMemoryMemoryService()


def create_memory_service_from_env():
    """
    Create memory service based on environment variables.

    Currently always returns InMemoryMemoryService.
    Future: Could read from MEMORY_SERVICE environment variable.

    Returns:
        Memory service instance

    Examples:
        >>> memory_service = create_memory_service_from_env()
    """
    # For now, always use in-memory
    # Future: Add environment variable support
    return create_memory_service(use_in_memory=True)


# Note: Custom SQLite-backed memory service can be added here in the future
# Example structure:
#
# class SQLiteMemoryService(BaseMemoryService):
#     """Custom SQLite-backed memory service for persistence."""
#
#     def __init__(self, db_path: str = "storyland_memory.db"):
#         self.db_path = db_path
#         self._init_database()
#
#     def _init_database(self):
#         # Create tables for storing memories
#         pass
#
#     async def add_session_to_memory(self, session):
#         # Store session events in SQLite
#         pass
#
#     async def search_memory(self, app_name, user_id, query):
#         # Search memories using keyword matching
#         pass
