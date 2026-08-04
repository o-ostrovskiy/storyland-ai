"""
Integration tests for LLM-as-judge scoring.

Tests the full scoring flow with real API calls (recorded via VCR).
"""

import pytest
import os
from evaluation.tools.llm_scorer import score_itinerary, ItineraryScores


@pytest.mark.integration
@pytest.mark.vcr()
async def test_score_itinerary_success():
    """Test scoring an itinerary with LLM-as-judge (real API call recorded)."""
    itinerary = {
        "summary": "A literary journey through Paris following Hemingway's footsteps",
        "cities": [
            {
                "city": "Paris",
                "country": "France",
                "days": 3,
                "description": "Explore the cafes and haunts of 1920s literary Paris",
                "stops": [
                    {
                        "name": "Shakespeare and Company",
                        "description": "Iconic English-language bookstore frequented by Hemingway",
                        "time_of_day": "morning",
                    },
                    {
                        "name": "Café de Flore",
                        "description": "Famous Left Bank café where writers gathered",
                        "time_of_day": "afternoon",
                    },
                ],
            }
        ],
    }

    preferences = {
        "budget": "moderate",
        "pace": "relaxed",
        "accessibility": None,
        "museums": True,
    }

    api_key = os.getenv("GOOGLE_API_KEY", "test-api-key-for-vcr")

    scores = await score_itinerary(
        api_key=api_key,
        book_title="A Moveable Feast",
        author="Ernest Hemingway",
        input_text="Plan a literary trip to Paris based on A Moveable Feast",
        itinerary=itinerary,
        preferences=preferences,
        model_name="gemini-2.0-flash-lite",
    )

    assert isinstance(scores, ItineraryScores)

    # All scores in valid range (1-5)
    assert 1 <= scores.book_relevance <= 5
    assert 1 <= scores.preference_adherence <= 5
    assert 1 <= scores.completeness <= 5
    assert 1 <= scores.actionability <= 5
    assert 1 <= scores.geographical_accuracy <= 5
    assert 1 <= scores.engagement <= 5

    avg = scores.average_score()
    assert 1.0 <= avg <= 5.0

    # This specific itinerary should score well on book_relevance and engagement
    # (Paris + Hemingway locations = highly relevant)
    assert scores.book_relevance >= 3, "Paris Hemingway itinerary should be relevant to book"
    assert scores.geographical_accuracy >= 4, "Real Paris locations should be geographically accurate"


@pytest.mark.integration
@pytest.mark.vcr()
async def test_score_itinerary_without_preferences():
    itinerary = {
        "summary": "Visit locations from Pride and Prejudice",
        "cities": [
            {
                "city": "Bath",
                "country": "England",
                "days": 2,
                "description": "Explore Regency-era Bath",
                "stops": [
                    {
                        "name": "The Roman Baths",
                        "description": "Historic Roman baths mentioned in the novel",
                        "time_of_day": "morning",
                    }
                ],
            }
        ],
    }

    api_key = os.getenv("GOOGLE_API_KEY", "test-api-key-for-vcr")

    scores = await score_itinerary(
        api_key=api_key,
        book_title="Pride and Prejudice",
        author="Jane Austen",
        input_text="Plan a trip based on Pride and Prejudice",
        itinerary=itinerary,
        preferences=None,  # No preferences
        model_name="gemini-2.0-flash-lite",
    )

    # Should still work without preferences
    assert isinstance(scores, ItineraryScores)
    assert 1 <= scores.book_relevance <= 5

    # PR-4 step zero: with no preferences there is no adherence target — the
    # dimension is NOT scored (None) and the average covers the 5 scored
    # dimensions. A judge grading adherence to nothing is noise, and blending
    # it into the average polluted the prod-shape signal.
    assert scores.preference_adherence is None
    scored_dims = [
        scores.book_relevance,
        scores.completeness,
        scores.actionability,
        scores.geographical_accuracy,
        scores.engagement,
    ]
    assert scores.average_score() == pytest.approx(sum(scored_dims) / 5)


@pytest.mark.integration
@pytest.mark.vcr()
async def test_score_poor_quality_itinerary():
    """Test scoring a low-quality itinerary (should get lower scores)."""
    # Generic itinerary with weak book connection
    itinerary = {
        "summary": "Generic travel itinerary",
        "cities": [
            {
                "city": "London",
                "country": "England",
                "days": 1,
                "description": "Visit London",
                "stops": [
                    {
                        "name": "Big Ben",
                        "description": "Famous clock tower",
                        "time_of_day": "afternoon",
                    }
                ],
            }
        ],
    }

    api_key = os.getenv("GOOGLE_API_KEY", "test-api-key-for-vcr")

    scores = await score_itinerary(
        api_key=api_key,
        book_title="The Sun Also Rises",  # Set in Spain/France, not England
        author="Ernest Hemingway",
        input_text="Plan a trip based on The Sun Also Rises",
        itinerary=itinerary,
        preferences=None,
        model_name="gemini-2.0-flash-lite",
    )

    # Should still return valid scores
    assert isinstance(scores, ItineraryScores)

    # Book relevance should be low (London is not in The Sun Also Rises)
    assert scores.book_relevance <= 3, "London itinerary should score low for The Sun Also Rises"

    # Completeness should be low (only 1 city, 1 stop)
    assert scores.completeness <= 3, "Minimal itinerary should score low on completeness"


@pytest.mark.integration
async def test_score_itinerary_error_handling():
    """Test error handling when API call fails (invalid API key)."""
    itinerary = {
        "summary": "Test itinerary",
        "cities": [{"city": "Paris", "country": "France", "days": 1, "stops": []}],
    }

    # Use invalid API key to trigger error
    with pytest.raises(Exception):
        await score_itinerary(
            api_key="invalid-api-key-that-will-fail",
            book_title="Test Book",
            author="Test Author",
            input_text="Test input",
            itinerary=itinerary,
            preferences=None,
            model_name="gemini-2.0-flash-lite",
        )
