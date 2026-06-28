"""Unit tests for request input-size bounds on DiscoverRequest / LocalAtmosphereRequest.

Imports ONLY from api.models (pydantic-only) so the suite stays light: oversized
book_title / author / preferences must raise a ValidationError (-> HTTP 422) BEFORE
any prompt is built or Gemini is called.
"""

import pytest
from pydantic import ValidationError

from api.models import (
    DiscoverRequest,
    LocalAtmosphereRequest,
    MAX_TITLE_LENGTH,
    MAX_AUTHOR_LENGTH,
    MAX_PREFERENCES_KEYS,
    MAX_PREFERENCE_VALUE_LENGTH,
)

VALID_LOCATION = {"lat": 40.7128, "lng": -74.0060, "label": "New York, NY"}


def test_discover_rejects_oversized_book_title():
    with pytest.raises(ValidationError):
        DiscoverRequest(book_title="x" * (MAX_TITLE_LENGTH + 1), author="A")


def test_discover_rejects_oversized_author():
    with pytest.raises(ValidationError):
        DiscoverRequest(book_title="1984", author="A" * (MAX_AUTHOR_LENGTH + 1))


def test_discover_rejects_too_many_preference_keys():
    prefs = {f"k{i}": "v" for i in range(MAX_PREFERENCES_KEYS + 1)}
    with pytest.raises(ValidationError):
        DiscoverRequest(book_title="1984", author="George Orwell", preferences=prefs)


def test_discover_rejects_oversized_preference_value():
    prefs = {"budget": "x" * (MAX_PREFERENCE_VALUE_LENGTH + 1)}
    with pytest.raises(ValidationError):
        DiscoverRequest(book_title="1984", author="George Orwell", preferences=prefs)


def test_discover_accepts_max_length_title_and_normal_prefs():
    req = DiscoverRequest(
        book_title="t" * MAX_TITLE_LENGTH,
        author="George Orwell",
        preferences={"budget": "luxury", "preferred_pace": "relaxed"},
    )
    assert len(req.book_title) == MAX_TITLE_LENGTH
    assert req.preferences == {"budget": "luxury", "preferred_pace": "relaxed"}


def test_local_atmosphere_rejects_oversized_book_title():
    with pytest.raises(ValidationError):
        LocalAtmosphereRequest(
            book_title="x" * (MAX_TITLE_LENGTH + 1),
            author="A",
            user_location=VALID_LOCATION,
        )


def test_local_atmosphere_accepts_valid_input():
    req = LocalAtmosphereRequest(
        book_title="Wuthering Heights",
        author="Emily Bronte",
        user_location=VALID_LOCATION,
        preferences={"pace": "slow"},
    )
    assert req.book_title == "Wuthering Heights"


# --- optional mood/vibe field on DiscoverRequest ---

def test_discover_accepts_known_vibe():
    req = DiscoverRequest(book_title="1984", author="George Orwell", vibe="Cozy")
    assert req.vibe == "cozy"  # normalized to canonical lower-case


def test_discover_vibe_absent_is_none():
    req = DiscoverRequest(book_title="1984", author="George Orwell")
    assert req.vibe is None


def test_discover_blank_vibe_is_none():
    req = DiscoverRequest(book_title="1984", author="George Orwell", vibe="   ")
    assert req.vibe is None


def test_discover_rejects_unknown_vibe():
    with pytest.raises(ValidationError):
        DiscoverRequest(book_title="1984", author="George Orwell", vibe="grumpy")


def test_discover_accepts_hyphenated_vibe():
    req = DiscoverRequest(
        book_title="1984", author="George Orwell", vibe="Slow-Burn"
    )
    assert req.vibe == "slow-burn"


# --- optional taste_context (imported reading history) on DiscoverRequest ---

def test_discover_taste_context_absent_is_none():
    req = DiscoverRequest(book_title="1984", author="George Orwell")
    assert req.taste_context is None


def test_discover_taste_context_normalizes_and_dedups():
    req = DiscoverRequest(
        book_title="1984",
        author="George Orwell",
        taste_context={
            "titles": ["  Dune ", "dune", "Beloved", ""],
            "moods": ["Bleak", "bleak", "  "],
        },
    )
    assert req.taste_context["titles"] == ["Dune", "Beloved"]
    assert req.taste_context["moods"] == ["Bleak"]


def test_discover_taste_context_empty_lists_become_none():
    req = DiscoverRequest(
        book_title="1984",
        author="George Orwell",
        taste_context={"titles": [], "moods": []},
    )
    assert req.taste_context is None


def test_discover_taste_context_truncates_to_caps():
    from api.models import MAX_TASTE_TITLES, MAX_TASTE_MOODS

    req = DiscoverRequest(
        book_title="1984",
        author="George Orwell",
        taste_context={
            "titles": [f"Book {i}" for i in range(MAX_TASTE_TITLES + 25)],
            "moods": [f"mood{i}" for i in range(MAX_TASTE_MOODS + 25)],
        },
    )
    assert len(req.taste_context["titles"]) == MAX_TASTE_TITLES
    assert len(req.taste_context["moods"]) == MAX_TASTE_MOODS


def test_discover_taste_context_bounds_string_length():
    from api.models import MAX_TASTE_STRING_LENGTH

    req = DiscoverRequest(
        book_title="1984",
        author="George Orwell",
        taste_context={"titles": ["x" * (MAX_TASTE_STRING_LENGTH + 500)]},
    )
    assert len(req.taste_context["titles"][0]) <= MAX_TASTE_STRING_LENGTH


def test_discover_taste_context_rejects_wrong_type():
    with pytest.raises(ValidationError):
        DiscoverRequest(
            book_title="1984", author="George Orwell", taste_context="not-an-object"
        )
