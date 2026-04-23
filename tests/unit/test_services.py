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
        connection_string = f"sqlite+aiosqlite:///{db_path}"

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
            connection_string="sqlite+aiosqlite:///should_be_ignored.db",
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
        'DATABASE_URL': 'sqlite+aiosqlite:///custom_test.db'
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
