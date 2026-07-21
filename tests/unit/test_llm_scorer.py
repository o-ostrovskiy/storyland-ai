"""
Unit tests for LLM-as-judge scorer.

Tests the scoring model, prompt building, and validation logic.
"""

import pytest
from pydantic import ValidationError

from evaluation.tools.llm_scorer import (
    ItineraryScores,
    _build_scoring_prompt,
    SCORING_CRITERIA,
)


class TestItineraryScoresModel:
    """Test the ItineraryScores Pydantic model."""

    def test_valid_scores(self):
        """Test creating model with valid scores."""
        scores = ItineraryScores(
            book_relevance=5,
            preference_adherence=4,
            completeness=3,
            actionability=4,
            geographical_accuracy=5,
            engagement=4,
        )

        assert scores.book_relevance == 5
        assert scores.preference_adherence == 4
        assert scores.completeness == 3
        assert scores.actionability == 4
        assert scores.geographical_accuracy == 5
        assert scores.engagement == 4

    def test_average_score_calculation(self):
        """Test average score calculation."""
        scores = ItineraryScores(
            book_relevance=5,
            preference_adherence=4,
            completeness=3,
            actionability=4,
            geographical_accuracy=5,
            engagement=3,
        )

        # (5 + 4 + 3 + 4 + 5 + 3) / 6 = 24 / 6 = 4.0
        assert scores.average_score() == 4.0

    def test_average_score_with_decimals(self):
        """Test average score with decimal result."""
        scores = ItineraryScores(
            book_relevance=5,
            preference_adherence=5,
            completeness=4,
            actionability=4,
            geographical_accuracy=5,
            engagement=3,
        )

        # (5 + 5 + 4 + 4 + 5 + 3) / 6 = 26 / 6 = 4.333...
        assert abs(scores.average_score() - 4.333) < 0.01

    def test_score_too_low_raises_validation_error(self):
        """Test that scores below 1 raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ItineraryScores(
                book_relevance=0,  # Invalid: too low
                preference_adherence=4,
                completeness=3,
                actionability=4,
                geographical_accuracy=5,
                engagement=4,
            )

        # Check that the error mentions the constraint
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_score_too_high_raises_validation_error(self):
        """Test that scores above 5 raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ItineraryScores(
                book_relevance=5,
                preference_adherence=6,  # Invalid: too high
                completeness=3,
                actionability=4,
                geographical_accuracy=5,
                engagement=4,
            )

        # Check that the error mentions the constraint
        assert "less than or equal to 5" in str(exc_info.value)

    def test_missing_field_raises_validation_error(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            # Missing engagement field
            ItineraryScores(
                book_relevance=5,
                preference_adherence=4,
                completeness=3,
                actionability=4,
                geographical_accuracy=5,
            )

    def test_model_json_serialization(self):
        """Test that model can be serialized to JSON."""
        scores = ItineraryScores(
            book_relevance=5,
            preference_adherence=4,
            completeness=3,
            actionability=4,
            geographical_accuracy=5,
            engagement=4,
        )

        json_str = scores.model_dump_json()
        assert "book_relevance" in json_str
        assert "5" in json_str

    def test_model_json_deserialization(self):
        """Test that model can be deserialized from JSON."""
        json_str = """{
            "book_relevance": 5,
            "preference_adherence": 4,
            "completeness": 3,
            "actionability": 4,
            "geographical_accuracy": 5,
            "engagement": 4
        }"""

        scores = ItineraryScores.model_validate_json(json_str)
        assert scores.book_relevance == 5
        assert scores.engagement == 4


class TestScoringCriteria:
    """Test scoring criteria definitions."""

    def test_all_criteria_defined(self):
        """Test that all 6 scoring criteria are defined."""
        expected_criteria = [
            "book_relevance",
            "preference_adherence",
            "completeness",
            "actionability",
            "geographical_accuracy",
            "engagement",
        ]

        for criterion in expected_criteria:
            assert criterion in SCORING_CRITERIA
            assert len(SCORING_CRITERIA[criterion]) > 0

    def test_criteria_contain_score_ranges(self):
        """Test that all criteria define 1-5 score ranges."""
        for criterion_name, criterion_text in SCORING_CRITERIA.items():
            # Each criterion should mention the 1-5 scale
            assert "1-5" in criterion_text or "Score 1-5" in criterion_text, \
                f"{criterion_name} doesn't mention 1-5 scale"

            # Each should define what each score means
            for score in [1, 2, 3, 4, 5]:
                assert str(score) in criterion_text, \
                    f"{criterion_name} doesn't define score {score}"


class TestBuildScoringPrompt:
    """Test the prompt building function."""

    def test_prompt_includes_book_info(self):
        """Test that prompt includes book title and author."""
        prompt = _build_scoring_prompt(
            book_title="The Great Gatsby",
            author="F. Scott Fitzgerald",
            input_text="Plan a trip based on The Great Gatsby",
            itinerary={"cities": ["New York"]},
            preferences=None,
        )

        assert "The Great Gatsby" in prompt
        assert "F. Scott Fitzgerald" in prompt

    def test_prompt_includes_input_text(self):
        """Test that prompt includes the original input."""
        input_text = "Plan a literary trip to Paris with museums"
        prompt = _build_scoring_prompt(
            book_title="A Moveable Feast",
            author="Ernest Hemingway",
            input_text=input_text,
            itinerary={"cities": ["Paris"]},
            preferences=None,
        )

        assert input_text in prompt

    def test_prompt_includes_itinerary(self):
        """Test that prompt includes the itinerary data."""
        itinerary = {
            "cities": ["Paris", "Lyon"],
            "summary": "A literary journey through France",
        }

        prompt = _build_scoring_prompt(
            book_title="A Moveable Feast",
            author="Ernest Hemingway",
            input_text="Plan a trip",
            itinerary=itinerary,
            preferences=None,
        )

        # Itinerary is JSON-formatted in the prompt
        assert "Paris" in prompt
        assert "Lyon" in prompt
        assert "literary journey" in prompt

    def test_prompt_includes_preferences_when_provided(self):
        """Test that prompt includes user preferences."""
        preferences = {
            "budget": "moderate",
            "pace": "relaxed",
            "accessibility": "wheelchair",
        }

        prompt = _build_scoring_prompt(
            book_title="A Moveable Feast",
            author="Ernest Hemingway",
            input_text="Plan a trip",
            itinerary={"cities": ["Paris"]},
            preferences=preferences,
        )

        assert "moderate" in prompt
        assert "relaxed" in prompt
        assert "wheelchair" in prompt

    def test_prompt_handles_no_preferences(self):
        """Test that prompt handles missing preferences gracefully."""
        prompt = _build_scoring_prompt(
            book_title="A Moveable Feast",
            author="Ernest Hemingway",
            input_text="Plan a trip",
            itinerary={"cities": ["Paris"]},
            preferences=None,
        )

        assert "No preferences specified" in prompt

    def test_prompt_includes_all_scoring_criteria(self):
        """Test that prompt includes all 6 scoring criteria."""
        prompt = _build_scoring_prompt(
            book_title="Test Book",
            author="Test Author",
            input_text="Test input",
            itinerary={"test": "data"},
            preferences=None,
        )

        # Check for criteria names
        assert "BOOK_RELEVANCE" in prompt
        assert "PREFERENCE_ADHERENCE" in prompt
        assert "COMPLETENESS" in prompt
        assert "ACTIONABILITY" in prompt
        assert "GEOGRAPHICAL_ACCURACY" in prompt
        assert "ENGAGEMENT" in prompt

        # Check that criteria text is included
        for criterion_name, criterion_text in SCORING_CRITERIA.items():
            # Check for at least part of each criterion
            # (prompts contain the full text, just verify it's there)
            assert "1-5" in prompt  # All should mention scale

    def test_prompt_format_is_valid(self):
        """Test that prompt is properly formatted (no syntax errors)."""
        prompt = _build_scoring_prompt(
            book_title="Test Book",
            author="Test Author",
            input_text="Test input",
            itinerary={"cities": ["TestCity"]},
            preferences={"budget": "low"},
        )

        # Basic checks for well-formed prompt
        assert len(prompt) > 100  # Should be substantial
        assert "**INPUT PROMPT**:" in prompt
        assert "**USER PREFERENCES**:" in prompt
        assert "**GENERATED ITINERARY**:" in prompt
        assert "Evaluate this travel itinerary" in prompt

    def test_prompt_json_formatting(self):
        """Test that JSON data in prompt is properly formatted."""
        itinerary = {
            "cities": ["Paris"],
            "nested": {"key": "value"},
        }
        preferences = {
            "budget": "moderate",
        }

        prompt = _build_scoring_prompt(
            book_title="Test",
            author="Author",
            input_text="Input",
            itinerary=itinerary,
            preferences=preferences,
        )

        # Check JSON is indented (formatted)
        assert '  "cities"' in prompt or '"cities"' in prompt
        assert '  "budget"' in prompt or '"budget"' in prompt


class TestBuildScoringPromptWithQualityCriteria:
    """Test quality_criteria injection in the scoring prompt."""

    def test_quality_criteria_injected_into_prompt(self):
        """Book-specific requirement lines appear for each provided key."""
        quality_criteria = {
            "book_relevance": "Must include Hobbiton and Rivendell filming locations in NZ.",
            "geographical_accuracy": "Sites must be real NZ/UK locations tied to Tolkien.",
        }
        prompt = _build_scoring_prompt(
            book_title="The Lord of the Rings",
            author="J.R.R. Tolkien",
            input_text="Plan a trip",
            itinerary={"cities": ["Matamata"]},
            quality_criteria=quality_criteria,
        )
        assert "Book-specific requirement:" in prompt
        assert "Hobbiton and Rivendell" in prompt
        assert "real NZ/UK locations" in prompt

    def test_quality_criteria_only_injects_provided_keys(self):
        """Only the dimension keys present in quality_criteria get injected."""
        quality_criteria = {"book_relevance": "Must reference the One Ring."}
        prompt = _build_scoring_prompt(
            book_title="The Lord of the Rings",
            author="J.R.R. Tolkien",
            input_text="Plan a trip",
            itinerary={"cities": ["Matamata"]},
            quality_criteria=quality_criteria,
        )
        assert prompt.count("Book-specific requirement:") == 1

    def test_no_quality_criteria_no_injection(self):
        """No injection when quality_criteria is None."""
        prompt = _build_scoring_prompt(
            book_title="The Lord of the Rings",
            author="J.R.R. Tolkien",
            input_text="Plan a trip",
            itinerary={"cities": ["Matamata"]},
            quality_criteria=None,
        )
        assert "Book-specific requirement:" not in prompt


class TestBuildScoringPromptWithExpectedOutput:
    """Test expected_output reference section injection in the scoring prompt."""

    def test_expected_output_section_appears_when_provided(self):
        """REFERENCE OUTPUT section appears in prompt when expected_output is given."""
        expected_output = {
            "locations": [{"city": "Matamata", "country": "New Zealand"}],
            "themes": ["Epic quest"],
            "duration": "10-14 days",
        }
        prompt = _build_scoring_prompt(
            book_title="The Lord of the Rings",
            author="J.R.R. Tolkien",
            input_text="Plan a trip",
            itinerary={"cities": ["Matamata"]},
            expected_output=expected_output,
        )
        assert "REFERENCE EXAMPLE" in prompt
        # MYS-586: the reference must be framed as context, never a benchmark
        # to resemble — that framing made books_v1 a similarity metric.
        assert "benchmark" not in prompt
        assert "do NOT penalize valid choices" in prompt
        assert "Matamata" in prompt
        assert "10-14 days" in prompt

    def test_no_expected_output_no_reference_section(self):
        """No reference section when expected_output is None."""
        prompt = _build_scoring_prompt(
            book_title="The Lord of the Rings",
            author="J.R.R. Tolkien",
            input_text="Plan a trip",
            itinerary={"cities": ["Matamata"]},
            expected_output=None,
        )
        assert "REFERENCE EXAMPLE" not in prompt
