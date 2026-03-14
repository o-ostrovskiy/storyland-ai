"""
Result extraction and validation.

Handles extracting structured data from agent responses, with fallback
to JSON text parsing when output_key doesn't work.

Extracted from api/streaming.py lines 89-104 and 564-598.
"""

import json
from typing import Optional

from pydantic import ValidationError

from common.logging import get_logger
from models.itinerary import TripItinerary

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
) -> Optional[dict]:
    """Two-phase extraction: session state first, then text fallback.

    Args:
        final_response: The final ADK event from runner.run_async()
        state_accessor: SessionStateAccessor wrapping session.state

    Returns:
        Validated TripItinerary dict or None
    """
    # Primary: from session state (set by output_key="final_itinerary")
    state_itinerary = state_accessor.final_itinerary
    result = validate_trip_itinerary(state_itinerary)
    if result is not None:
        logger.info("itinerary_from_state")
        return result

    # Fallback: parse from final response text
    if (
        final_response
        and final_response.content
        and final_response.content.parts
    ):
        for part in final_response.content.parts:
            if hasattr(part, "text") and part.text:
                candidate = extract_json_from_text(part.text)
                if candidate is not None:
                    result = validate_trip_itinerary(candidate)
                    if result is not None:
                        logger.info("itinerary_from_text_fallback")
                        return result

    return None
