"""
LLM-as-judge scorer for itinerary evaluation.

This module provides automated scoring of travel itineraries using Gemini 2.0 Flash Lite
with structured output. Scores are based on 6 quality dimensions aligned with the
scoring functions defined in langfuse_eval.py.

Issue: #96 - Add LLM-as-judge scoring to evaluation pipeline
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional

from langfuse import get_client, observe
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from common.logging import get_logger

logger = get_logger("storyland.llm_scorer")


# Scoring criteria extracted from evaluation/tools/langfuse_eval.py (lines 172-262)
SCORING_CRITERIA = {
    "book_relevance": (
        "Evaluate if the travel itinerary locations are directly connected to the book's "
        "settings, themes, characters, or author. Each suggested place should have a clear "
        "and meaningful connection to the literary work.\n\n"
        "Score 1-5 where:\n"
        "5 = All locations strongly connected to book\n"
        "4 = Most locations clearly relevant\n"
        "3 = Some locations relevant, some generic\n"
        "2 = Few relevant locations\n"
        "1 = No clear connection to book"
    ),
    "preference_adherence": (
        "Evaluate if the travel recommendations respect the user's stated preferences "
        "including budget level, travel pace, accessibility needs, and interest in museums. "
        "The response should acknowledge and adapt to these preferences.\n\n"
        "Score 1-5 where:\n"
        "5 = Perfectly aligned with all preferences\n"
        "4 = Respects most preferences well\n"
        "3 = Acknowledges some preferences\n"
        "2 = Ignores most preferences\n"
        "1 = No consideration of preferences"
    ),
    "completeness": (
        "Evaluate if the response includes a comprehensive itinerary with cities, landmarks, "
        "and author-related sites. The itinerary should provide enough detail for the traveler "
        "to plan their trip.\n\n"
        "Score 1-5 where:\n"
        "5 = Comprehensive with all components\n"
        "4 = Good coverage of main elements\n"
        "3 = Basic itinerary, missing some details\n"
        "2 = Minimal information\n"
        "1 = Incomplete or vague response"
    ),
    "actionability": (
        "Evaluate if the itinerary is practical and actionable with specific places, "
        "suggested times of day to visit, and logistical notes. The traveler should be "
        "able to use this information to actually plan their trip.\n\n"
        "Score 1-5 where:\n"
        "5 = Highly detailed and actionable\n"
        "4 = Good practical details\n"
        "3 = Some actionable information\n"
        "2 = Vague suggestions\n"
        "1 = No practical details"
    ),
    "geographical_accuracy": (
        "Evaluate if the locations mentioned are real places that can be visited. "
        "Cities and landmarks should be correctly associated with their countries. "
        "The geographical information should be accurate.\n\n"
        "Score 1-5 where:\n"
        "5 = All locations accurate and real\n"
        "4 = Minor geographical details off\n"
        "3 = Some questionable locations\n"
        "2 = Multiple inaccuracies\n"
        "1 = Fictional or incorrect locations"
    ),
    "engagement": (
        "Evaluate if the summary and descriptions are engaging and evoke the spirit of "
        "the book. The language should capture the literary connection and make the trip "
        "feel like a meaningful journey.\n\n"
        "Score 1-5 where:\n"
        "5 = Highly engaging, captures book's spirit\n"
        "4 = Good descriptions, engaging tone\n"
        "3 = Adequate but generic descriptions\n"
        "2 = Dry or uninspiring\n"
        "1 = No literary connection in writing"
    ),
}


class ItineraryScores(BaseModel):
    """Structured output model for LLM-as-judge scoring."""

    book_relevance: int = Field(..., ge=1, le=5, description="Connection to book's settings, themes, or author")
    # Optional: not scored when the case supplies no preferences (prod-shape
    # cases post-MYS-392) — a judge grading adherence to nothing is noise.
    preference_adherence: Optional[int] = Field(None, ge=1, le=5, description="Respect for user preferences")
    completeness: int = Field(..., ge=1, le=5, description="Comprehensive details included")
    actionability: int = Field(..., ge=1, le=5, description="Practical and actionable information")
    geographical_accuracy: int = Field(..., ge=1, le=5, description="Accuracy of locations")
    engagement: int = Field(..., ge=1, le=5, description="Engaging descriptions that evoke book's spirit")

    def average_score(self) -> float:
        """Average across the dimensions that were actually scored.

        preference_adherence is None for no-preference (prod-shape) cases and
        is excluded — a 5-dimension average — so the two shapes are each
        internally consistent and are compared per-shape, never blended
        silently (see run_scheduled_eval's by_shape summary).
        """
        dims = [
            self.book_relevance,
            self.preference_adherence,
            self.completeness,
            self.actionability,
            self.geographical_accuracy,
            self.engagement,
        ]
        present = [d for d in dims if d is not None]
        return sum(present) / len(present)


def _build_scoring_prompt(
    book_title: str,
    author: str,
    input_text: str,
    itinerary: Dict[str, Any],
    preferences: Optional[Dict[str, Any]] = None,
    quality_criteria: Optional[Dict[str, str]] = None,
    expected_output: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the scoring prompt for LLM-as-judge evaluation.

    Args:
        book_title: Title of the book
        author: Author of the book
        input_text: Original user input/prompt
        itinerary: Generated itinerary data
        preferences: User preferences (optional)

    Returns:
        Formatted prompt string with all scoring criteria
    """
    # Format preferences for display
    if preferences:
        preferences_json = json.dumps(preferences, indent=2)
    else:
        preferences_json = "No preferences specified"

    # Format itinerary for display
    itinerary_json = json.dumps(itinerary, indent=2)

    # Format expected output section (only when provided)
    if expected_output:
        reference_section = f"**REFERENCE OUTPUT** (use this as benchmark when scoring):\n{json.dumps(expected_output, indent=2)}\n"
    else:
        reference_section = ""

    def _criterion_block(key: str, number: int, label: str) -> str:
        block = f"**{number}. {label} (1-5)**\n{SCORING_CRITERIA[key]}"
        if quality_criteria and quality_criteria.get(key):
            block += f"\n\nBook-specific requirement: {quality_criteria[key]}"
        return block

    prompt = f"""Evaluate this travel itinerary for "{book_title}" by {author}.

Rate each dimension on a 1-5 scale based on the criteria below:

{_criterion_block('book_relevance', 1, 'BOOK_RELEVANCE')}

{_criterion_block('preference_adherence', 2, 'PREFERENCE_ADHERENCE')}

{_criterion_block('completeness', 3, 'COMPLETENESS')}

{_criterion_block('actionability', 4, 'ACTIONABILITY')}

{_criterion_block('geographical_accuracy', 5, 'GEOGRAPHICAL_ACCURACY')}

{_criterion_block('engagement', 6, 'ENGAGEMENT')}

---

**INPUT PROMPT**:
{input_text}

**USER PREFERENCES**:
{preferences_json}
{reference_section}**GENERATED ITINERARY**:
{itinerary_json}

---

Provide scores only (no explanations). Use the structured output format.
"""

    return prompt


@observe(name="llm_score_itinerary", as_type="generation")
async def score_itinerary(
    api_key: str,
    book_title: str,
    author: str,
    input_text: str,
    itinerary: Dict[str, Any],
    preferences: Optional[Dict[str, Any]] = None,
    quality_criteria: Optional[Dict[str, str]] = None,
    expected_output: Optional[Dict[str, Any]] = None,
    model_name: str = "gemini-2.5-flash-lite",
) -> ItineraryScores:
    """Score an itinerary using LLM-as-judge with structured output (6 dimensions, 1-5 scale)."""
    logger.info(
        "llm_scoring_start",
        book_title=book_title,
        author=author,
        model=model_name,
    )

    # Build scoring prompt
    prompt = _build_scoring_prompt(
        book_title=book_title,
        author=author,
        input_text=input_text,
        itinerary=itinerary,
        preferences=preferences,
        quality_criteria=quality_criteria,
        expected_output=expected_output,
    )

    logger.debug(
        "llm_scoring_prompt_built",
        prompt_length=len(prompt),
        has_preferences=preferences is not None,
        has_quality_criteria=quality_criteria is not None,
        quality_criteria_keys=list(quality_criteria.keys()) if quality_criteria else [],
        has_expected_output=expected_output is not None,
    )

    if quality_criteria:
        missing_keys = sorted(set(SCORING_CRITERIA.keys()) - set(quality_criteria.keys()))
        if missing_keys:
            logger.warning(
                "quality_criteria_partial",
                book_title=book_title,
                missing_keys=missing_keys,
                message="These dimensions will use generic scoring criteria",
            )

    get_client().update_current_generation(
        model=model_name,
        input={"book_title": book_title, "author": author, "has_preferences": preferences is not None},
        metadata={
            "scoring_method": "llm_judge",
            "dimensions": list(SCORING_CRITERIA.keys()),
        },
    )

    try:
        # Create GenAI client
        client = genai.Client(api_key=api_key)

        # Call model requesting JSON output in prompt
        # Note: Using simple prompt-based JSON request instead of response_schema
        # for broader model compatibility
        score_preferences = bool(preferences)
        if score_preferences:
            json_instruction = """

Respond with ONLY a valid JSON object (no markdown, no explanation) with these exact fields:
{
  "book_relevance": <integer 1-5>,
  "preference_adherence": <integer 1-5>,
  "completeness": <integer 1-5>,
  "actionability": <integer 1-5>,
  "geographical_accuracy": <integer 1-5>,
  "engagement": <integer 1-5>
}
"""
        else:
            json_instruction = """

No reader preferences were provided for this case, so do NOT score preference adherence.
Respond with ONLY a valid JSON object (no markdown, no explanation) with these exact fields:
{
  "book_relevance": <integer 1-5>,
  "completeness": <integer 1-5>,
  "actionability": <integer 1-5>,
  "geographical_accuracy": <integer 1-5>,
  "engagement": <integer 1-5>
}
"""

        response = client.models.generate_content(
            model=model_name,
            contents=prompt + json_instruction,
            config=types.GenerateContentConfig(
                temperature=0.0,  # Deterministic scoring
            ),
        )

        # Parse and validate response
        # Extract JSON from response (handle potential markdown code blocks)
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith("```"):
            response_text = response_text[3:]  # Remove ```
        if response_text.endswith("```"):
            response_text = response_text[:-3]  # Remove trailing ```
        response_text = response_text.strip()

        scores = ItineraryScores.model_validate_json(response_text)
        if not score_preferences and scores.preference_adherence is not None:
            # Judge scored a dimension it was told to skip — force honesty.
            scores = scores.model_copy(update={"preference_adherence": None})
        if score_preferences and scores.preference_adherence is None:
            # The inverse gap (Codex P2 on PR #228): the field is Optional for
            # the no-preference shape, so a judge that OMITS it on a
            # preference-carrying case would silently produce a 5-dimension
            # average — the API-contract shape passing without adherence ever
            # being measured. A missing demanded dimension is a scoring
            # FAILURE, not a thinner success: raise into the existing
            # scoring-failed path so the case shows up unscored and visible.
            raise ValueError(
                "judge omitted preference_adherence on a preference-carrying case"
            )

        usage = response.usage_metadata
        get_client().update_current_generation(
            output=scores.model_dump(),
            usage_details={
                "input": usage.prompt_token_count or 0,
                "output": usage.candidates_token_count or 0,
                "total": usage.total_token_count or 0,
            } if usage else None,
            level="DEFAULT",
            status_message="Scoring completed successfully",
        )

        logger.info(
            "llm_scoring_complete",
            book_relevance=scores.book_relevance,
            preference_adherence=scores.preference_adherence,
            completeness=scores.completeness,
            actionability=scores.actionability,
            geographical_accuracy=scores.geographical_accuracy,
            engagement=scores.engagement,
            average_score=round(scores.average_score(), 2),
        )

        return scores

    except Exception as e:
        get_client().update_current_generation(
            level="ERROR",
            status_message=f"Scoring failed: {str(e)}",
        )
        logger.error(
            "llm_scoring_failed",
            error=str(e),
            error_type=type(e).__name__,
            book_title=book_title,
        )
        raise
