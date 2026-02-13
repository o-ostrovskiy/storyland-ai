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
    preference_adherence: int = Field(..., ge=1, le=5, description="Respect for user preferences")
    completeness: int = Field(..., ge=1, le=5, description="Comprehensive details included")
    actionability: int = Field(..., ge=1, le=5, description="Practical and actionable information")
    geographical_accuracy: int = Field(..., ge=1, le=5, description="Accuracy of locations")
    engagement: int = Field(..., ge=1, le=5, description="Engaging descriptions that evoke book's spirit")

    def average_score(self) -> float:
        """Calculate average score across all dimensions."""
        return (
            self.book_relevance
            + self.preference_adherence
            + self.completeness
            + self.actionability
            + self.geographical_accuracy
            + self.engagement
        ) / 6


def _build_scoring_prompt(
    book_title: str,
    author: str,
    input_text: str,
    itinerary: Dict[str, Any],
    preferences: Optional[Dict[str, Any]] = None,
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

    prompt = f"""Evaluate this travel itinerary for "{book_title}" by {author}.

Rate each dimension on a 1-5 scale based on the criteria below:

**1. BOOK_RELEVANCE (1-5)**
{SCORING_CRITERIA['book_relevance']}

**2. PREFERENCE_ADHERENCE (1-5)**
{SCORING_CRITERIA['preference_adherence']}

**3. COMPLETENESS (1-5)**
{SCORING_CRITERIA['completeness']}

**4. ACTIONABILITY (1-5)**
{SCORING_CRITERIA['actionability']}

**5. GEOGRAPHICAL_ACCURACY (1-5)**
{SCORING_CRITERIA['geographical_accuracy']}

**6. ENGAGEMENT (1-5)**
{SCORING_CRITERIA['engagement']}

---

**INPUT PROMPT**:
{input_text}

**USER PREFERENCES**:
{preferences_json}

**GENERATED ITINERARY**:
{itinerary_json}

---

Provide scores only (no explanations). Use the structured output format.
"""

    return prompt


async def score_itinerary(
    api_key: str,
    book_title: str,
    author: str,
    input_text: str,
    itinerary: Dict[str, Any],
    preferences: Optional[Dict[str, Any]] = None,
    model_name: str = "gemini-2.0-flash-lite",
    langfuse_trace_id: Optional[str] = None,
    langfuse_client: Optional[Any] = None,
) -> ItineraryScores:
    """
    Score an itinerary using LLM-as-judge with structured output.

    This function uses Gemini 2.0 Flash Lite to evaluate a travel itinerary across
    6 quality dimensions: book relevance, preference adherence, completeness,
    actionability, geographical accuracy, and engagement.

    Args:
        api_key: Google API key for authentication
        book_title: Title of the book the itinerary is based on
        author: Author of the book
        input_text: Original user input/prompt that generated the itinerary
        itinerary: Generated itinerary data (dict)
        preferences: User preferences used for itinerary generation (optional)
        model_name: Gemini model to use for scoring (default: gemini-2.0-flash-lite)
        langfuse_trace_id: Optional Langfuse trace ID for execution tracing
        langfuse_client: Optional Langfuse client for creating observations

    Returns:
        ItineraryScores with scores 1-5 for each dimension

    Raises:
        Exception: If scoring fails (API error, invalid response, etc.)
    """
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
    )

    logger.debug(
        "llm_scoring_prompt_built",
        prompt_length=len(prompt),
        has_preferences=preferences is not None,
    )

    # Create Langfuse generation for execution tracing
    generation_id = None
    if langfuse_client and langfuse_trace_id:
        try:
            generation = langfuse_client.generation(
                trace_id=langfuse_trace_id,
                name="llm_judge_scoring",
                model=model_name,
                input={
                    "book_title": book_title,
                    "author": author,
                    "has_preferences": preferences is not None,
                },
                metadata={
                    "scoring_method": "llm_judge",
                    "dimensions": [
                        "book_relevance",
                        "preference_adherence",
                        "completeness",
                        "actionability",
                        "geographical_accuracy",
                        "engagement",
                    ],
                },
            )
            generation_id = generation.id
            logger.debug("langfuse_generation_created", generation_id=generation_id)
        except Exception as e:
            logger.warning("failed_to_create_langfuse_generation", error=str(e))

    try:
        # Create GenAI client
        client = genai.Client(api_key=api_key)

        # Call model requesting JSON output in prompt
        # Note: Using simple prompt-based JSON request instead of response_schema
        # due to API compatibility issues with gemini-2.0-flash-lite
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

        # Update Langfuse generation with output and usage
        if langfuse_client and generation_id:
            try:
                langfuse_client.generation(
                    id=generation_id,
                    output=scores.model_dump(),
                    usage={
                        "input": len(prompt + json_instruction),
                        "output": len(response_text),
                        "unit": "CHARACTERS",
                    },
                    level="DEFAULT",
                    status_message="Scoring completed successfully",
                )
                logger.debug("langfuse_generation_updated", generation_id=generation_id)
            except Exception as e:
                logger.warning("failed_to_update_langfuse_generation", error=str(e))

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
        # Update Langfuse generation with error
        if langfuse_client and generation_id:
            try:
                langfuse_client.generation(
                    id=generation_id,
                    level="ERROR",
                    status_message=f"Scoring failed: {str(e)}",
                )
            except Exception as update_error:
                logger.warning("failed_to_update_langfuse_error", error=str(update_error))

        logger.error(
            "llm_scoring_failed",
            error=str(e),
            error_type=type(e).__name__,
            book_title=book_title,
        )
        raise
