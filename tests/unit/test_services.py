"""
Unit tests for services.

Tests session service factory and context manager functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from services.session_service import (
    create_session_service,
    create_session_service_from_env
)


# =============================================================================
# Session Service Factory Tests
# =============================================================================

class TestCreateSessionService:
    """Tests for create_session_service factory."""

    def test_create_in_memory_service_default(self):
        """Test default creates InMemorySessionService."""
        service = create_session_service()

        # Should be InMemorySessionService (check class name)
        assert "InMemory" in type(service).__name__

    def test_create_in_memory_service_explicit(self):
        """Test explicit use_database=False creates InMemorySessionService."""
        service = create_session_service(use_database=False)

        assert "InMemory" in type(service).__name__

    def test_create_database_service(self, tmp_path):
        """Test use_database=True creates DatabaseSessionService."""
        db_path = tmp_path / "test.db"
        connection_string = f"sqlite:///{db_path}"

        service = create_session_service(
            connection_string=connection_string,
            use_database=True
        )

        assert "Database" in type(service).__name__

    def test_create_database_service_default_path(self):
        """Test DatabaseSessionService uses default path when not provided."""
        service = create_session_service(use_database=True)

        assert "Database" in type(service).__name__

    def test_create_session_service_ignores_connection_string_when_not_database(self):
        """Test connection_string is ignored when use_database=False."""
        service = create_session_service(
            connection_string="sqlite:///should_be_ignored.db",
            use_database=False
        )

        assert "InMemory" in type(service).__name__


class TestCreateSessionServiceFromEnv:
    """Tests for create_session_service_from_env function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults_to_in_memory(self):
        """Test defaults to in-memory when env vars not set."""
        # Clear relevant env vars
        os.environ.pop('USE_DATABASE', None)
        os.environ.pop('DATABASE_URL', None)

        service = create_session_service_from_env()

        assert "InMemory" in type(service).__name__

    @patch.dict(os.environ, {'USE_DATABASE': 'true'})
    def test_from_env_with_use_database_true(self):
        """Test creates database service when USE_DATABASE=true."""
        service = create_session_service_from_env()

        assert "Database" in type(service).__name__

    @patch.dict(os.environ, {'USE_DATABASE': 'false'})
    def test_from_env_with_use_database_false(self):
        """Test creates in-memory service when USE_DATABASE=false."""
        service = create_session_service_from_env()

        assert "InMemory" in type(service).__name__

    @patch.dict(os.environ, {
        'USE_DATABASE': 'true',
        'DATABASE_URL': 'sqlite:///custom_test.db'
    })
    def test_from_env_with_custom_database_url(self):
        """Test uses DATABASE_URL from environment."""
        service = create_session_service_from_env()

        assert "Database" in type(service).__name__

    @patch.dict(os.environ, {'USE_DATABASE': 'TRUE'})
    def test_from_env_case_insensitive(self):
        """Test USE_DATABASE is case insensitive."""
        service = create_session_service_from_env()

        assert "Database" in type(service).__name__


# =============================================================================
# Context Manager Tests (if context_manager.py exists)
# =============================================================================

class TestContextManager:
    """Tests for ContextManager service."""

    @pytest.fixture
    def context_manager(self):
        """Create a ContextManager instance with max_events only."""
        try:
            from services.context_manager import ContextManager
            return ContextManager(max_events=20)
        except ImportError:
            pytest.skip("ContextManager not available")

    @pytest.fixture
    def context_manager_with_tokens(self):
        """Create a ContextManager instance with token limit."""
        try:
            from services.context_manager import ContextManager
            return ContextManager(max_events=20, max_tokens=1000)
        except ImportError:
            pytest.skip("ContextManager not available")

    def test_context_manager_initialization(self, context_manager):
        """Test ContextManager initializes with max_events."""
        assert context_manager.max_events == 20

    def test_context_manager_initialization_with_tokens(self, context_manager_with_tokens):
        """Test ContextManager initializes with max_tokens."""
        assert context_manager_with_tokens.max_tokens == 1000

    def test_limit_events_within_limit(self, context_manager):
        """Test limit_events when events are within limit."""
        events = [{"content": f"Event {i}"} for i in range(5)]

        limited = context_manager.limit_events(events, num_recent=10)

        assert len(limited) == 5

    def test_limit_events_exceeds_limit(self, context_manager):
        """Test limit_events when events exceed limit."""
        events = [{"content": f"Event {i}"} for i in range(20)]

        limited = context_manager.limit_events(events, num_recent=5)

        assert len(limited) == 5
        # Should keep the most recent events
        assert limited[-1]["content"] == "Event 19"

    def test_should_compact_returns_bool(self, context_manager):
        """Test should_compact returns boolean based on event count."""
        # Create events below the limit
        events = [{"content": "Test"}]
        result = context_manager.should_compact(events)
        assert result is False

    def test_should_compact_exceeds_max_events(self, context_manager):
        """Test should_compact returns True when exceeding max_events."""
        # Create events exceeding the limit (max_events=20)
        events = [{"content": f"Event {i}"} for i in range(25)]
        result = context_manager.should_compact(events)
        assert result is True
