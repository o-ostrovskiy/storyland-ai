"""Output-tone / safety guardrails for reader-facing explanation text.

Currently exposes the rec-explanation *tone* guardrail: the system-instruction
clause that keeps every "why it fits" explanation focused on the book<->place
fit, plus the deterministic, zero-spend output checker that strips any
reader-directed judgement that slips through.
"""

from .tone_guardrail import (
    READER_TONE_GUARDRAIL,
    flag_reader_directed,
    sanitize_explanation,
    sanitize_itinerary_explanations,
    sanitize_expansion_explanations,
    sanitize_book_recommendations,
)

__all__ = [
    "READER_TONE_GUARDRAIL",
    "flag_reader_directed",
    "sanitize_explanation",
    "sanitize_itinerary_explanations",
    "sanitize_expansion_explanations",
    "sanitize_book_recommendations",
]
