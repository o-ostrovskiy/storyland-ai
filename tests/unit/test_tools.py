"""
Unit tests for tools.

Tests Google Books API tool and preferences tool functionality.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from tools.google_books import search_books, search_book, google_books_tool
from tools.preferences import get_user_preferences, get_preferences_tool
from models.book import BookInfo, BookMetadata


# =============================================================================
# Google Books API Tests
# =============================================================================

class TestSearchBooks:
    """Tests for search_books function."""

    @patch('tools.google_books.requests.get')
    def test_search_books_success(self, mock_get, mock_google_books_response):
        """Test successful book search."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_google_books_response
        mock_get.return_value.raise_for_status = MagicMock()

        results = search_books("Pride and Prejudice", "Jane Austen")

        assert len(results) == 1
        assert results[0].title == "Pride and Prejudice"
        assert "Jane Austen" in results[0].authors

    @patch('tools.google_books.requests.get')
    def test_search_books_empty_results(self, mock_get, mock_google_books_empty_response):
        """Test search with no results."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_google_books_empty_response
        mock_get.return_value.raise_for_status = MagicMock()

        results = search_books("Nonexistent Book Title XYZ123")

        assert len(results) == 0

    @patch('tools.google_books.requests.get')
    def test_search_books_with_author(self, mock_get, mock_google_books_response):
        """Test search with author filter."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_google_books_response
        mock_get.return_value.raise_for_status = MagicMock()

        search_books("Pride", "Austen")

        # Verify the query includes both title and author
        call_args = mock_get.call_args
        assert "intitle:Pride" in call_args[1]['params']['q']
        assert "inauthor:Austen" in call_args[1]['params']['q']

    @patch('tools.google_books.requests.get')
    def test_search_books_api_error(self, mock_get):
        """Test handling of API errors."""
        mock_get.return_value.raise_for_status.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            search_books("Test Book")

    @patch('tools.google_books.requests.get')
    def test_search_books_handles_missing_fields(self, mock_get):
        """Test search handles missing optional fields in API response."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "items": [{
                "volumeInfo": {
                    "title": "Minimal Book"
                    # Missing authors, description, etc.
                }
            }]
        }
        mock_get.return_value.raise_for_status = MagicMock()

        results = search_books("Minimal")

        assert len(results) == 1
        assert results[0].title == "Minimal Book"
        assert results[0].authors == []
        assert results[0].description is None

    @patch('tools.google_books.requests.get')
    def test_search_books_max_results(self, mock_get, mock_google_books_response):
        """Test max_results parameter is passed correctly."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_google_books_response
        mock_get.return_value.raise_for_status = MagicMock()

        search_books("Test", max_results=3)

        call_args = mock_get.call_args
        assert call_args[1]['params']['maxResults'] == 3


class TestSearchBook:
    """Tests for search_book function (returns JSON string)."""

    @patch('tools.google_books.search_books')
    def test_search_book_success(self, mock_search, sample_book_info_list):
        """Test successful book search returns valid JSON."""
        mock_search.return_value = sample_book_info_list

        result = search_book("Pride and Prejudice", "Jane Austen")

        # Should be valid JSON
        data = json.loads(result)
        assert data['book_title'] == "Pride and Prejudice"
        assert data['author'] == "Jane Austen"

    @patch('tools.google_books.search_books')
    def test_search_book_returns_pydantic_validated_json(self, mock_search, sample_book_info_list):
        """Test search_book returns Pydantic-validated JSON."""
        mock_search.return_value = sample_book_info_list

        result = search_book("Pride and Prejudice")

        # Should be parseable as BookMetadata
        data = json.loads(result)
        metadata = BookMetadata(**data)
        assert metadata.book_title == "Pride and Prejudice"

    @patch('tools.google_books.search_books')
    def test_search_book_no_results(self, mock_search):
        """Test search_book with no results returns error JSON."""
        mock_search.return_value = []

        result = search_book("Nonexistent Book")

        data = json.loads(result)
        assert "error" in data
        assert data['error'] == "No books found"

    @patch('tools.google_books.search_books')
    def test_search_book_exception(self, mock_search):
        """Test search_book handles exceptions gracefully."""
        mock_search.side_effect = Exception("Network error")

        result = search_book("Test Book")

        data = json.loads(result)
        assert "error" in data
        assert "Network error" in data['error']

    @patch('tools.google_books.search_books')
    def test_search_book_selects_first_result(self, mock_search, sample_book_info_list):
        """Test search_book selects the first/best result."""
        mock_search.return_value = sample_book_info_list

        result = search_book("Pride")

        data = json.loads(result)
        # Should select first result, not "Pride and Prejudice and Zombies"
        assert data['book_title'] == "Pride and Prejudice"


class TestGoogleBooksTool:
    """Tests for google_books_tool FunctionTool."""

    def test_google_books_tool_exists(self):
        """Test google_books_tool is properly created."""
        assert google_books_tool is not None
        # FunctionTool wraps the search_book function
        assert google_books_tool.func == search_book


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
