"""
Shared fixtures and mocks for StoryLand AI tests.

This module provides:
- Mock responses for Google Books API
- Sample book data fixtures
- Session state fixtures
- Preference fixtures
"""

import os

# Langfuse export guard. This has to stay above every other import: common.config
# runs load_dotenv() at import time, and that walks up from common/ to the real
# .env, so a test run in the main checkout or in any worktree under it would get
# production keys and send its fixture spans to the production project.
#
# load_dotenv() never overwrites a variable that is already set, even to "", so
# blanking here wins over the file. Blank keys alone are not enough: the SDK only
# disables itself when a key is None, so LANGFUSE_TRACING_ENABLED is what stops a
# client built with explicit keys.
#
# tests/integration opts in with STORYLAND_TEST_LANGFUSE_EXPORT=1, which the
# Makefile's live and re-record targets set. tests/unit/conftest.py refuses it.
LANGFUSE_EXPORT_OPT_IN = "STORYLAND_TEST_LANGFUSE_EXPORT"
LANGFUSE_CONNECTION_VARS = (
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
)

if os.environ.get(LANGFUSE_EXPORT_OPT_IN) != "1":
    for _var in LANGFUSE_CONNECTION_VARS:
        os.environ[_var] = ""
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any

from models.book import BookMetadata, BookContext, BookInfo
from models.discovery import CityDiscovery, LandmarkDiscovery, AuthorSites, CityInfo, LandmarkInfo, AuthorSiteInfo
from models.itinerary import TripItinerary, CityPlan, CityStop
from models.preferences import TravelPreferences


# Sample Book Data Fixtures

@pytest.fixture
def sample_book_metadata() -> BookMetadata:
    """Sample BookMetadata for Pride and Prejudice."""
    return BookMetadata(
        book_title="Pride and Prejudice",
        author="Jane Austen",
        description="A classic novel about love and social standing in early 19th century England.",
        published_date="1813",
        categories=["Fiction", "Romance", "Classic Literature"],
        image_url="https://books.google.com/books/content?id=s1gVAAAAYAAJ"
    )


@pytest.fixture
def sample_book_context() -> BookContext:
    """Sample BookContext for Pride and Prejudice."""
    return BookContext(
        primary_locations=["Hertfordshire, England", "Derbyshire, England", "London"],
        time_period="Early 19th century Regency Era",
        themes=["Marriage", "Social class", "Pride", "Prejudice", "Love"]
    )


@pytest.fixture
def sample_book_info() -> BookInfo:
    """Sample BookInfo from Google Books API."""
    return BookInfo(
        title="Pride and Prejudice",
        authors=["Jane Austen"],
        description="A classic novel about love and social standing.",
        published_date="1813",
        categories=["Fiction", "Romance"],
        image_url="https://books.google.com/books/content?id=s1gVAAAAYAAJ"
    )


@pytest.fixture
def sample_book_info_list(sample_book_info) -> list:
    return [
        sample_book_info,
        BookInfo(
            title="Pride and Prejudice and Zombies",
            authors=["Seth Grahame-Smith"],
            description="A parody mashup.",
            published_date="2009",
            categories=["Fiction", "Horror"],
            image_url=None
        )
    ]


# Discovery Fixtures

@pytest.fixture
def sample_city_discovery() -> CityDiscovery:
    return CityDiscovery(
        cities=[
            CityInfo(
                name="Bath",
                country="England",
                relevance="Jane Austen lived here and set scenes from her novels"
            ),
            CityInfo(
                name="Chatsworth",
                country="England",
                relevance="Inspiration for Pemberley estate"
            )
        ]
    )


@pytest.fixture
def sample_landmark_discovery() -> LandmarkDiscovery:
    return LandmarkDiscovery(
        landmarks=[
            LandmarkInfo(
                name="Chatsworth House",
                city="Bakewell",
                connection="Believed to be the inspiration for Mr. Darcy's Pemberley estate"
            ),
            LandmarkInfo(
                name="Jane Austen Centre",
                city="Bath",
                connection="Museum dedicated to Jane Austen's life and works"
            )
        ]
    )


@pytest.fixture
def sample_author_sites() -> AuthorSites:
    return AuthorSites(
        author_sites=[
            AuthorSiteInfo(
                name="Jane Austen's House Museum",
                type="museum",
                city="Chawton"
            ),
            AuthorSiteInfo(
                name="Winchester Cathedral",
                type="burial site",
                city="Winchester"
            )
        ]
    )


# Itinerary Fixtures

@pytest.fixture
def sample_city_stop() -> CityStop:
    return CityStop(
        name="Chatsworth House",
        type="landmark",
        reason="The grand estate that inspired Pemberley",
        time_of_day="morning",
        notes="Book tickets in advance; allow 3-4 hours"
    )


@pytest.fixture
def sample_city_plan(sample_city_stop) -> CityPlan:
    return CityPlan(
        name="Bakewell",
        country="England",
        days_suggested=2,
        overview="A charming market town in the Peak District, home to Chatsworth House",
        stops=[sample_city_stop]
    )


@pytest.fixture
def sample_trip_itinerary(sample_city_plan) -> TripItinerary:
    return TripItinerary(
        cities=[sample_city_plan],
        summary_text="Follow in the footsteps of Elizabeth Bennet through the English countryside."
    )


# Preferences Fixtures

@pytest.fixture
def sample_preferences() -> TravelPreferences:
    return TravelPreferences(
        prefers_museums=True,
        travels_with_kids=False,
        budget="moderate",
        favorite_genres=["Classic Literature", "Romance"],
        favorite_authors=["Jane Austen", "Charlotte Bronte"],
        dietary_restrictions=[],
        accessibility_needs=False,
        preferred_pace="relaxed"
    )


@pytest.fixture
def sample_preferences_dict() -> Dict[str, Any]:
    """Sample preferences as dictionary (for session state)."""
    return {
        "prefers_museums": True,
        "travels_with_kids": False,
        "budget": "moderate",
        "favorite_genres": ["Classic Literature", "Romance"],
        "favorite_authors": ["Jane Austen"],
        "dietary_restrictions": [],
        "accessibility_needs": False,
        "preferred_pace": "relaxed"
    }


@pytest.fixture
def luxury_preferences_dict() -> Dict[str, Any]:
    return {
        "prefers_museums": True,
        "travels_with_kids": False,
        "budget": "luxury",
        "favorite_genres": ["Literary Fiction"],
        "favorite_authors": [],
        "dietary_restrictions": ["vegetarian"],
        "accessibility_needs": False,
        "preferred_pace": "relaxed"
    }


@pytest.fixture
def family_preferences_dict() -> Dict[str, Any]:
    return {
        "prefers_museums": True,
        "travels_with_kids": True,
        "budget": "moderate",
        "favorite_genres": ["Adventure", "Fantasy"],
        "favorite_authors": [],
        "dietary_restrictions": [],
        "accessibility_needs": False,
        "preferred_pace": "relaxed"
    }


# Mock Responses

@pytest.fixture
def mock_tool_context(sample_preferences_dict):
    context = MagicMock()
    context.state = {"user:preferences": sample_preferences_dict}
    return context


@pytest.fixture
def mock_tool_context_no_preferences():
    context = MagicMock()
    context.state = {}
    return context


# Session State Fixtures

@pytest.fixture
def sample_session_state(
    sample_book_metadata,
    sample_book_context,
    sample_city_discovery,
    sample_landmark_discovery,
    sample_author_sites,
    sample_preferences_dict
) -> Dict[str, Any]:
    """Complete session state after running all agents."""
    return {
        "book_title": "Pride and Prejudice",
        "author": "Jane Austen",
        "user:preferences": sample_preferences_dict,
        "book_metadata": sample_book_metadata.model_dump(),
        "book_context": sample_book_context.model_dump(),
        "city_discovery": sample_city_discovery.model_dump(),
        "landmark_discovery": sample_landmark_discovery.model_dump(),
        "author_sites": sample_author_sites.model_dump(),
        "reader_profile": "A relaxed traveler who enjoys museums and classic literature."
    }


# Async Test Helpers

@pytest.fixture
def event_loop_policy():
    """Use default event loop policy for async tests."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
