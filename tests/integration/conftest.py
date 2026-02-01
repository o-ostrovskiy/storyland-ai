"""Integration test fixtures and configuration."""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@pytest.fixture(scope="module")
def vcr_config():
    """
    VCR configuration for integration tests.

    This configuration:
    - Records API responses once, then replays them
    - Filters sensitive data (API keys) from cassettes
    - Matches requests on method, host, path, and query
    """
    # Ensure cassette directory exists
    cassette_dir = Path(__file__).parent / "cassettes"
    cassette_dir.mkdir(exist_ok=True)

    return {
        "cassette_library_dir": str(cassette_dir),
        "record_mode": "once",  # Record once, replay thereafter
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "filter_headers": ["authorization"],  # Don't record auth headers
        "filter_query_parameters": ["key"],  # Redact API key from cassettes
        "decode_compressed_response": True,
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
