"""Integration test fixtures and configuration."""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Load environment variables from .env file
load_dotenv()


def normalize_query_string(url):
    """
    Normalize a URL by sorting its query parameters.
    This ensures consistent query string order regardless of dict ordering.
    """
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query, keep_blank_values=True)
    # Remove API key parameter
    query_params.pop('key', None)
    # Sort and rebuild query string
    normalized_query = urlencode(sorted(query_params.items()), doseq=True)
    # Rebuild URL
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        normalized_query,
        parsed.fragment
    ))


@pytest.fixture(scope="module")
def vcr_config():
    """
    VCR configuration for integration tests.

    This configuration:
    - Records API responses (use --vcr-record=all to regenerate cassettes)
    - Filters sensitive data (API keys) from cassettes
    - Normalizes query strings for consistent matching
    """
    # Ensure cassette directory exists
    cassette_dir = Path(__file__).parent / "cassettes"
    cassette_dir.mkdir(exist_ok=True)

    def before_record_request(request):
        """Normalize URL query parameters before recording."""
        request.uri = normalize_query_string(request.uri)
        return request

    def before_record_response(response):
        """Placeholder for consistency."""
        return response

    return {
        "cassette_library_dir": str(cassette_dir),
        "record_mode": "once",  # Record once, replay thereafter
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "filter_headers": ["authorization"],  # Don't record auth headers
        "filter_query_parameters": ["key"],  # Redact API key from cassettes
        "decode_compressed_response": True,
        "before_record_request": before_record_request,
        "before_record_response": before_record_response,
    }


@pytest.fixture
def real_api_key():
    """
    Provide real API key for testing (if available).

    Skips test if GOOGLE_API_KEY is not set.
    """
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set - skipping real API test")
    return key


@pytest.fixture
def sample_integration_queries():
    """Common queries for integration testing."""
    return [
        {"title": "Pride and Prejudice", "author": "Jane Austen"},
        {"title": "1984", "author": "George Orwell"},
        {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald"},
        {"title": "To Kill a Mockingbird", "author": "Harper Lee"},
        {"title": "The Nightingale", "author": "Kristin Hannah"},
    ]
