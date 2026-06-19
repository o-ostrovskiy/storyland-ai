"""
Place→Book reverse-discovery models (AI candidate layer).

These models describe the *candidate* shape the storyland-ai reverse-routing
capability produces for a destination: grounded books set in (literal) or
evoking (vibe) a place, each with a short "why it fits" explanation.

This is the AI layer only. The Google Books *existence* check and the final
user-facing grounding object live in the BE (storyland-services) place→book
endpoint, which calls this capability and decorates each candidate. Keeping the
existence check out of the AI layer respects the standing rule that book
existence is verified in BE via the Google Books API, not ai-side.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


MatchType = Literal["literal", "vibe"]


class PlaceBookCandidate(BaseModel):
    """A single place→book candidate from the reverse-routing pipeline."""

    title: str = Field(description="Exact book title")
    author: str = Field(description="Primary author full name")
    description: Optional[str] = Field(
        default=None,
        description="Brief, real synopsis from the researcher; null when none was found",
    )
    why_it_fits: str = Field(
        description="1-2 sentences on why this book suits a trip to the place"
    )
    match_type: MatchType = Field(
        description=(
            "'literal' = the story is set in the place | "
            "'vibe' = it evokes the place's mood but is not set there"
        )
    )
    maps_to: Optional[str] = Field(
        default=None,
        description=(
            "Real location the story maps to (literal matches only). "
            "MUST be null for vibe matches."
        ),
    )


class PlaceToBookCandidates(BaseModel):
    """Formatter output: the labelled candidate list for a place.

    The list MAY be empty: an ungroundable / obscure place should yield zero
    candidates (→ a clean not-found state) rather than a fabricated list.
    """

    candidates: List[PlaceBookCandidate] = Field(
        default_factory=list,
        description=(
            "Grounded, labelled book candidates for the place; "
            "empty when none can be grounded (never pad with invented books)"
        ),
    )


class PlaceToBookResult(BaseModel):
    """Resolver result: candidates plus a found/not-found envelope.

    Mirrors the BE place→book contract minus the Google Books ``grounding``
    object (BE adds that). ``found=False`` carries an empty ``candidates`` list
    and a human message — never a fabricated list.
    """

    place: str = Field(description="The raw place string the caller asked for")
    query: str = Field(description="Normalized place key used for caching/lookup")
    found: bool = Field(
        description="True when at least one grounded candidate was produced"
    )
    message: Optional[str] = Field(
        default=None,
        description="Human-readable note for the not-found state; null when found",
    )
    candidates: List[PlaceBookCandidate] = Field(
        default_factory=list,
        description="Labelled candidates; empty when not found",
    )
