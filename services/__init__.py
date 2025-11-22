"""
Service layer for StoryLand AI.

This package contains service factories for:
- Session management (InMemory or SQLite-backed)
- Context management (token optimization)
"""

from .session_service import create_session_service
from .context_manager import ContextManager

__all__ = [
    "create_session_service",
    "ContextManager",
]
