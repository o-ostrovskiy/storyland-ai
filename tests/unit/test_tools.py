"""
Unit tests for tools.

Tests preferences tool functionality.
"""

import json
import pytest

from tools.preferences import get_user_preferences, get_preferences_tool


# =============================================================================
# Preferences Tool Tests
# =============================================================================

class TestGetUserPreferences:
    """Tests for get_user_preferences function."""

    def test_get_preferences_with_preferences(self, mock_tool_context, sample_preferences_dict):
        """Test getting preferences when they exist."""
        result = get_user_preferences(mock_tool_context)

        data = json.loads(result)
        assert data['found'] is True
        assert data['preferences']['budget'] == "moderate"
        assert "Jane Austen" in data['preferences']['favorite_authors']

    def test_get_preferences_without_preferences(self, mock_tool_context_no_preferences):
        """Test getting preferences when none exist."""
        result = get_user_preferences(mock_tool_context_no_preferences)

        data = json.loads(result)
        assert data['found'] is False
        assert data['preferences'] == {}
        assert "No user preferences found" in data['message']

    def test_get_preferences_returns_valid_json(self, mock_tool_context):
        """Test that get_user_preferences always returns valid JSON."""
        result = get_user_preferences(mock_tool_context)

        # Should not raise
        data = json.loads(result)
        assert isinstance(data, dict)


class TestGetPreferencesTool:
    """Tests for get_preferences_tool FunctionTool."""

    def test_get_preferences_tool_exists(self):
        """Test get_preferences_tool is properly created."""
        assert get_preferences_tool is not None
        assert get_preferences_tool.func == get_user_preferences
