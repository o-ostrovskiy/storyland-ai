"""Integration test fixtures and configuration."""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Load environment variables from .env file
load_dotenv()

# Provide a dummy GOOGLE_API_KEY for offline VCR cassette replay when none is set
# (e.g. proxied/no-key sandbox or autonomous QA runs). setdefault never overrides a
# real key, so CI — which injects secrets.GOOGLE_API_KEY — and live runs are unaffected.
os.environ.setdefault("GOOGLE_API_KEY", "test-key-for-vcr-replay")


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


def _match_method_case_insensitive(r1, r2):
    """Case-insensitive HTTP-method matcher.

    The ADK 2.x / google-genai 2.x client stack reports request methods
    lowercased ("post") where the 1.x-era cassettes recorded "POST"; VCR's
    built-in "method" matcher is a case-sensitive string compare, so every
    replay failed with `POST != post` on an otherwise-identical request.
    Method casing is semantically meaningless in HTTP, so match it
    case-insensitively instead of re-recording every cassette (PR 4 of the
    ADK 2 migration re-records them anyway, for the model bump).
    """
    assert r1.method.upper() == r2.method.upper(), f"{r1.method} != {r2.method}"


@pytest.fixture(scope="module")
def vcr(vcr):
    """Shadow pytest-vcr's instance fixture to register the custom matcher."""
    vcr.register_matcher("method_ci", _match_method_case_insensitive)
    return vcr


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
        "match_on": ["method_ci", "scheme", "host", "port", "path", "query"],
        "filter_headers": ["authorization", "x-goog-api-key"],  # Don't record auth headers
        "filter_query_parameters": ["key"],  # Redact API key from cassettes
        "decode_compressed_response": True,
        "before_record_request": before_record_request,
        "before_record_response": before_record_response,
        "ignore_hosts": [
            "cloud.langfuse.com",
            "us.cloud.langfuse.com",
            "eu.cloud.langfuse.com",
        ],  # Don't intercept Langfuse OTLP telemetry calls
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
