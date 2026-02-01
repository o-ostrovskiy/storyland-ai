"""
Integration tests for Google Books API.

These tests use VCR.py to record/replay real API responses:
- First run: Records responses from real Google Books API
- Subsequent runs: Replays from cassettes (fast, no quota usage)
"""

import json
import pytest
from tools.google_books import search_books, search_book
from models.book import BookMetadata


class TestSearchBooksIntegration:
    """Test Suite A: Basic API Functionality"""

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_search_books_success(self):
        """Test successful book search with real API (VCR recorded)."""
        results = search_books("Pride and Prejudice", "Jane Austen")

        assert len(results) > 0, "Should return at least one result"
        assert any("Pride and Prejudice" in book.title for book in results)
        assert any(
            any("Austen" in author for author in book.authors) for book in results
        )

        # Verify BookInfo structure
        first_result = results[0]
        assert isinstance(first_result.title, str)
        assert isinstance(first_result.authors, list)

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_search_books_no_results(self):
        """Test search with nonsensical query returns empty list."""
        results = search_books("XYZABC123NONEXISTENT456IMPOSSIBLE")

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_search_book_returns_valid_json(self):
        """Test search_book returns Pydantic-validated JSON."""
        result = search_book("1984", "George Orwell")
        data = json.loads(result)

        assert "book_title" in data
        assert "author" in data

        # Validate against BookMetadata schema
        metadata = BookMetadata(**data)
        assert "1984" in metadata.book_title

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_max_results_parameter(self):
        """Test max_results parameter is respected."""
        results = search_books("Python Programming", max_results=3)

        assert len(results) <= 3

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_search_books_with_title_only(self):
        """Test search with only title (no author)."""
        results = search_books("The Great Gatsby", max_results=5)

        assert len(results) > 0
        assert any("Gatsby" in book.title for book in results)


class TestAPIKeyScenarios:
    """Test Suite B: API Key Scenarios"""

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_search_with_api_key(self):
        """Test API call with GOOGLE_API_KEY present (uses real key from .env)."""
        # This test uses the real API key loaded from .env
        results = search_books("The Nightingale", "Kristin Hannah")

        assert len(results) > 0
        # Verify we get expected results
        assert any("Nightingale" in book.title for book in results)

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_search_without_api_key(self, monkeypatch):
        """Test API call behavior without API key (may hit quota limits)."""
        from requests.exceptions import HTTPError

        # Remove API key to test public quota fallback
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        try:
            # Use a simple query
            results = search_books("The Hobbit")

            # If successful, verify results
            assert isinstance(results, list)
        except HTTPError as e:
            # Public quota may be exhausted (429), which is expected behavior
            if e.response.status_code == 429:
                pytest.skip("Public quota exhausted (expected without API key)")
            else:
                # Other HTTP errors should still fail the test
                raise


class TestResponseParsing:
    """Test Suite C: Response Parsing"""

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_image_url_https_upgrade(self):
        """Test that HTTP image URLs are upgraded to HTTPS."""
        results = search_books("Pride and Prejudice", max_results=5)

        for result in results:
            if result.image_url:
                assert result.image_url.startswith(
                    "https://"
                ), f"Image URL should be HTTPS: {result.image_url}"

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_multiple_authors(self):
        """Test handling of books with multiple authors."""
        results = search_books("Good Omens", "Terry Pratchett")

        assert len(results) > 0
        # Find a result with multiple authors
        multi_author_books = [r for r in results if len(r.authors) > 1]
        if multi_author_books:
            assert isinstance(multi_author_books[0].authors, list)
            assert len(multi_author_books[0].authors) >= 2

    @pytest.mark.integration
    @pytest.mark.vcr()
    def test_missing_optional_fields(self):
        """Test handling of books with missing descriptions, dates, etc."""
        results = search_books("The Odyssey", "Homer")

        # Verify graceful handling even if some fields are sparse
        for result in results:
            assert result.title is not None
            # authors, description, published_date, image_url may be None/empty
            assert isinstance(result.authors, list)
            assert isinstance(result.categories, list)
