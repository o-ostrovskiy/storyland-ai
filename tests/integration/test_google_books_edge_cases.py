"""
Integration tests for Google Books API edge cases and error handling.

These tests use the 'responses' library to simulate error conditions
that are difficult to trigger with real API calls.
"""

import json
import pytest
import responses
from requests.exceptions import Timeout, HTTPError, ConnectionError
from tenacity import RetryError
from tools.google_books import search_books, search_book


class TestNetworkErrors:
    """Test Suite D: Network & Timeout Errors"""

    @pytest.mark.integration
    def test_network_timeout_error(self):
        """Test handling of network timeout using responses library."""

        @responses.activate
        def run_test():
            responses.add(
                responses.GET,
                "https://www.googleapis.com/books/v1/volumes",
                body=Timeout("Connection timeout after 10 seconds"),
            )

            with pytest.raises((Timeout, Exception)) as exc_info:
                search_books("Test Book")

            # Verify exception was raised
            assert exc_info.value is not None

        run_test()

    @pytest.mark.integration
    def test_api_500_error(self):
        """Test handling of API 500 internal server error."""

        @responses.activate
        def run_test():
            responses.add(
                responses.GET,
                "https://www.googleapis.com/books/v1/volumes",
                status=500,
                json={"error": {"message": "Internal server error"}},
            )

            with pytest.raises((RetryError, HTTPError)) as exc_info:
                search_books("Test Book")

            # Extract the actual error (either HTTPError or the cause of RetryError)
            error = exc_info.value
            if isinstance(error, RetryError):
                error = error.last_attempt.exception()
            assert error.response.status_code == 500

        run_test()

    @pytest.mark.integration
    def test_rate_limit_429(self):
        """Test handling of API rate limit error (429)."""

        @responses.activate
        def run_test():
            responses.add(
                responses.GET,
                "https://www.googleapis.com/books/v1/volumes",
                status=429,
                json={
                    "error": {
                        "code": 429,
                        "message": "Rate limit exceeded",
                        "status": "RESOURCE_EXHAUSTED",
                    }
                },
            )

            with pytest.raises((RetryError, HTTPError)) as exc_info:
                search_books("Test Book")

            # Extract the actual error (either HTTPError or the cause of RetryError)
            error = exc_info.value
            if isinstance(error, RetryError):
                error = error.last_attempt.exception()
            assert error.response.status_code == 429

        run_test()

    @pytest.mark.integration
    def test_connection_error(self):
        """Test handling of connection errors."""

        @responses.activate
        def run_test():
            responses.add(
                responses.GET,
                "https://www.googleapis.com/books/v1/volumes",
                body=ConnectionError("Failed to establish connection"),
            )

            with pytest.raises((ConnectionError, Exception)) as exc_info:
                search_books("Test Book")

            assert exc_info.value is not None

        run_test()


class TestQueryValidation:
    """Test Suite E: Query Parameter Validation"""

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_special_characters_in_query(self):
        """Test handling of special characters in title/author."""
        # Should not crash, may return 0 or more results
        results = search_books("C++ Programming", "Bjarne Stroustrup")
        assert isinstance(results, list)

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_unicode_characters(self):
        """Test handling of Unicode characters in queries."""
        results = search_books("Café", "Gabriel García Márquez")
        assert isinstance(results, list)

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_very_long_title(self):
        """Test handling of very long title strings."""
        long_title = "A" * 500  # Very long title
        results = search_books(long_title)
        assert isinstance(results, list)

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_empty_string_title(self):
        """Test behavior with empty string (should still make API call)."""
        # Empty title still constructs a query, may return many results
        results = search_books("", "Shakespeare")
        assert isinstance(results, list)

    @pytest.mark.integration
    def test_search_book_handles_exception_gracefully(self):
        """Test that search_book returns error JSON instead of raising."""

        @responses.activate
        def run_test():
            responses.add(
                responses.GET,
                "https://www.googleapis.com/books/v1/volumes",
                status=500,
                json={"error": "Server error"},
            )

            # search_book should NOT raise, it returns error JSON
            result = search_book("Test Book")
            data = json.loads(result)  # Parse JSON string

            assert "error" in data or "type" in data

        run_test()


class TestSearchBookEdgeCases:
    """Additional edge cases for search_book function"""

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_search_book_no_results_returns_error_json(self):
        """Test that search_book returns error JSON when no books found."""
        result = search_book("XYZABC123NONEXISTENT456IMPOSSIBLE")
        data = json.loads(result)

        assert "error" in data
        assert data["error"] == "No books found matching your search"

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_search_book_selects_first_result(self):
        """Test that search_book automatically selects the first/best match."""
        result = search_book("Harry Potter")
        data = json.loads(result)

        # Should have valid book metadata
        assert "book_title" in data
        assert "author" in data
        assert "Harry Potter" in data["book_title"]

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_search_book_with_both_params(self):
        """Test search_book with both title and author."""
        result = search_book("The Hobbit", "J.R.R. Tolkien")
        data = json.loads(result)

        assert "book_title" in data
        assert "Hobbit" in data["book_title"]
