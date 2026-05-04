"""
Result extraction and validation.

Handles extracting structured data from agent responses, with fallback
to JSON text parsing when output_key doesn't work.

Extracted from api/streaming.py lines 89-104 and 564-598.
"""

import json
from typing import Optional, Tuple

from pydantic import ValidationError

from common.logging import get_logger
from models.book import BookRecommendationsResult
from models.itinerary import TripItinerary, ComposerEnvelope, ExpansionResult

logger = get_logger("storyland.core.extraction")


def validate_trip_itinerary(value: object) -> Optional[dict]:
    """Validate an itinerary payload against TripItinerary schema.

    Returns validated dict or None if invalid.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if not isinstance(value, dict):
        return None

    try:
        validated = TripItinerary.model_validate(value)
        return validated.model_dump()
    except ValidationError:
        return None


def validate_composer_envelope(value: object) -> Optional[Tuple[dict, list]]:
    """Validate a ComposerEnvelope payload.

    Returns (itinerary_dict, suggestions_list) or None if invalid.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if not isinstance(value, dict):
        return None

    try:
        envelope = ComposerEnvelope.model_validate(value)
        return envelope.itinerary.model_dump(), [s.model_dump() for s in envelope.suggestions]
    except ValidationError:
        return None


def validate_expansion_result(value: object) -> Optional[dict]:
    """Validate an ExpansionResult payload.

    Returns validated dict or None if invalid.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if not isinstance(value, dict):
        return None

    try:
        validated = ExpansionResult.model_validate(value)
        return validated.model_dump()
    except ValidationError:
        return None


def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract first JSON object from text (finds outermost braces)."""
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        try:
            return json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return None
    return None


def extract_itinerary_from_response(
    final_response, state_accessor
) -> Optional[Tuple[dict, list]]:
    """Two-phase extraction: session state first, then text fallback.

    Returns (itinerary_dict, suggestions_list). Suggestions may be empty for
    legacy responses that predate ComposerEnvelope.

    Args:
        final_response: The final ADK event from runner.run_async()
        state_accessor: SessionStateAccessor wrapping session.state

    Returns:
        (itinerary_dict, suggestions_list) tuple or None
    """
    # Primary: composer_envelope from session state (set by output_key="composer_envelope")
    envelope_data = state_accessor.composer_envelope
    if envelope_data is not None:
        result = validate_composer_envelope(envelope_data)
        if result is not None:
            logger.info("itinerary_from_envelope")
            return result

    # Legacy fallback: bare TripItinerary from state (set by output_key="final_itinerary")
    state_itinerary = state_accessor.final_itinerary
    itinerary_result = validate_trip_itinerary(state_itinerary)
    if itinerary_result is not None:
        logger.info("itinerary_from_state")
        return itinerary_result, []

    # Text fallback: parse from final response text
    if (
        final_response
        and final_response.content
        and final_response.content.parts
    ):
        for part in final_response.content.parts:
            if hasattr(part, "text") and part.text:
                candidate = extract_json_from_text(part.text)
                if candidate is not None:
                    # Try envelope first
                    env_result = validate_composer_envelope(candidate)
                    if env_result is not None:
                        logger.info("itinerary_from_text_envelope_fallback")
                        return env_result
                    # Then bare itinerary
                    itinerary_result = validate_trip_itinerary(candidate)
                    if itinerary_result is not None:
                        logger.info("itinerary_from_text_fallback")
                        return itinerary_result, []

    return None


def extract_expansion_from_state(state_accessor) -> Optional[dict]:
    """Extract and validate the last expansion result from session state."""
    return validate_expansion_result(state_accessor.last_expansion)


def validate_book_recommendations_result(value: object) -> Optional[dict]:
    """Validate a BookRecommendationsResult payload.

    Returns validated dict or None if invalid.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if not isinstance(value, dict):
        return None

    try:
        validated = BookRecommendationsResult.model_validate(value)
        return validated.model_dump()
    except ValidationError:
        return None


def extract_book_recommendations_from_state(state_accessor) -> Optional[dict]:
    """Extract and validate the last book recommendations result from session state."""
    return validate_book_recommendations_result(state_accessor.last_book_recommendations)
