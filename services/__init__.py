"""
Service layer for StoryLand AI.

This package contains service factories for:
- Session management (InMemory or SQLite-backed)
"""

from .session_service import create_session_service

__all__ = [
    "create_session_service",
]
