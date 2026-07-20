"""
Result extraction and validation.

Handles extracting structured data from agent responses, with fallback
to JSON text parsing when output_key doesn't work.

Extracted from api/streaming.py lines 89-104 and 564-598.
"""

import json
import re
from typing import Optional, Tuple

from pydantic import ValidationError

from common.logging import get_logger
from core.guardrails import (
    sanitize_itinerary_explanations,
    sanitize_expansion_explanations,
    sanitize_book_recommendations,
)
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


# Match types that assert a grounded, verifiable book<->place connection and
# therefore must trace to the grounded discovery research. The other two
# ("thematic", "vibe") are explicitly weaker/atmospheric claims that need no
# source, so they are never touched here.
_GROUNDED_MATCH_TYPES = ("literal", "historical")

# The weakest, safest claim. An ungroundable strong claim is downgraded to this
# (never dropped, never upgraded), mirroring the schema default and the org
# "choose the weaker claim when unsure" guardrail.
_DOWNGRADE_TARGET = "vibe"


def downgrade_ungrounded_match_types(
    itinerary_dict: Optional[dict], grounding_text: str
) -> Optional[dict]:
    """Downgrade literal/historical stops that don't trace to grounded research.

    The trust core of the hallucination guardrail: the formatter self-labels each
    stop's ``match_type``, but a self-label is not evidence. Here we *derive* the
    label server-side from a real check — any stop the formatter marked
    ``literal``/``historical`` whose name is not grounded in the discovery
    research (city/landmark/author/context; see ``is_title_grounded`` for the
    token-overlap matching rule) is downgraded to the
    weakest claim (``vibe``) and has its ``grounding_source`` cleared, since the
    cited evidence didn't hold up. Stops are never dropped and never upgraded.

    Conservative and fail-open by design (it must never make results worse):
      * No grounding text captured -> return unchanged (we cannot prove anything
        is ungrounded, e.g. the local-atmosphere path has no discovery research).
      * ``thematic``/``vibe`` stops are left untouched (no source required).
    """
    if not itinerary_dict:
        return itinerary_dict

    haystack = grounding_token_set(grounding_text)
    if not haystack:
        return itinerary_dict

    downgraded = 0
    for city in itinerary_dict.get("cities") or []:
        for stop in city.get("stops") or []:
            if stop.get("match_type") not in _GROUNDED_MATCH_TYPES:
                continue
            if is_title_grounded(stop.get("name"), haystack):
                continue
            stop["match_type"] = _DOWNGRADE_TARGET
            stop["grounding_source"] = None
            downgraded += 1

    if downgraded:
        logger.info("itinerary_match_type_downgraded", downgraded=downgraded)
    return itinerary_dict


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
    grounding_text = state_accessor.grounding_research_text

    def _finalize(itinerary_dict, suggestions):
        itinerary_dict = downgrade_ungrounded_match_types(itinerary_dict, grounding_text)
        itinerary_dict = sanitize_itinerary_explanations(itinerary_dict)
        return itinerary_dict, suggestions

    # Primary: composer_envelope from session state (set by output_key="composer_envelope")
    envelope_data = state_accessor.composer_envelope
    if envelope_data is not None:
        result = validate_composer_envelope(envelope_data)
        if result is not None:
            logger.info("itinerary_from_envelope")
            return _finalize(*result)

    # Legacy fallback: bare TripItinerary from state (set by output_key="final_itinerary")
    state_itinerary = state_accessor.final_itinerary
    itinerary_result = validate_trip_itinerary(state_itinerary)
    if itinerary_result is not None:
        logger.info("itinerary_from_state")
        return _finalize(itinerary_result, [])

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
                        return _finalize(*env_result)
                    # Then bare itinerary
                    itinerary_result = validate_trip_itinerary(candidate)
                    if itinerary_result is not None:
                        logger.info("itinerary_from_text_fallback")
                        return _finalize(itinerary_result, [])

    return None


def extract_expansion_from_state(state_accessor) -> Optional[dict]:
    """Extract and validate the last expansion result from session state."""
    return sanitize_expansion_explanations(
        validate_expansion_result(state_accessor.last_expansion)
    )


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
    return sanitize_book_recommendations(
        validate_book_recommendations_result(state_accessor.last_book_recommendations)
    )


# Leading/interior articles carry no grounding signal and are the usual source
# of surface-form mismatches ("Pump Room" vs "The Grand Pump Room").
_ARTICLES = frozenset({"the", "a", "an"})

# 1-2 token titles require every token in the grounding text; 3+ token titles
# tolerate exactly this many missing tokens, which absorbs surface variants
# like "The Grand Pump Room" vs "Pump Room". A flat overlap ratio is
# deliberately avoided: it would let long titles pass with several unsupported
# tokens, and long place names are where scattered-word false matches bite.
_GROUNDING_MAX_MISSING = 1


def grounding_token_set(text: object) -> frozenset:
    """Tokenize grounding text once for repeated is_title_grounded() checks.

    Word tokens (unicode-aware, lowercased, punctuation stripped). An empty
    result means "no usable evidence" — every caller treats that as its
    fail-open branch, same as the old empty-string check.
    """
    if not isinstance(text, str):
        return frozenset()
    return frozenset(re.findall(r"\w+", text.lower()))


def is_title_grounded(title: object, haystack_tokens: frozenset) -> bool:
    """Shared grounding-match primitive for the output-side guards.

    Token containment replacing naive substring matching, which failed in
    both directions: "The Mill" false-matched any text containing the
    substring (e.g. "the millionaire"), while a grounded "The Grand Pump
    Room" was missed when the research said "Pump Room". Matching on whole
    word tokens fixes the former; tolerating at most one missing token on
    3+ token titles fixes the latter. The allowance is a fixed count, not a
    ratio, so a long title cannot pass on scattered partial support.

    A title with no significant tokens (empty/None/articles-only) is never
    grounded — identical to the old empty-normalized-title behavior.
    """
    title_tokens = grounding_token_set(title) - _ARTICLES
    if not title_tokens:
        return False
    missing = len(title_tokens - haystack_tokens)
    allowed_missing = _GROUNDING_MAX_MISSING if len(title_tokens) >= 3 else 0
    return missing <= allowed_missing


def filter_grounded_recommendations(
    rec_data: Optional[dict], researcher_text: str
) -> Optional[dict]:
    """Drop recommendations whose title is not grounded in the researcher's text.

    The book-recommendation formatter is tool-less and instructed never to
    invent books, but as a defensive post-validation we drop any recommendation
    whose title is not grounded in the researcher candidate text (per the
    ``is_title_grounded`` token-overlap rule). This
    is the output-side complement to relaxing the schema floor: relaxing the
    floor removes the *pressure* to fabricate; this removes any title that was
    fabricated anyway.

    Conservative and fail-open by design (it must never make results worse):
      * No researcher text captured -> return ``rec_data`` unchanged (we cannot
        prove anything is ungrounded, so we never drop on missing evidence).
      * Filtering that would drop *every* recommendation -> return ``rec_data``
        unchanged (treat as a capture/formatting mismatch, never surface empty).
      * Otherwise return the grounded subset with ``limited_matches=True`` so the
        caller can signal an honest "fewer real matches" state instead of padding.
    """
    if rec_data is None:
        return None

    recs = rec_data.get("recommendations") or []
    haystack = grounding_token_set(researcher_text)
    if not haystack:
        return rec_data

    grounded = [rec for rec in recs if is_title_grounded(rec.get("title"), haystack)]

    if not grounded or len(grounded) == len(recs):
        return rec_data

    result = dict(rec_data)
    result["recommendations"] = grounded
    result["limited_matches"] = True
    logger.info(
        "book_recommendations_grounding_filtered",
        kept=len(grounded),
        dropped=len(recs) - len(grounded),
    )
    return result
