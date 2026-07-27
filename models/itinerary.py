"""
Itinerary-related Pydantic models.

Contains models for the final travel itinerary, including city plans and stops.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, List


class CityStop(BaseModel):
    """A stop/place to visit in a city"""

    name: str = Field(description="Name of the place")
    type: str = Field(
        description="Type: landmark, museum, cafe, restaurant, bookstore, filming_location, etc."
    )
    reason: str = Field(description="Why this place matters for the book")
    address: Optional[str] = Field(
        default=None,
        description="Street address or location description (e.g., '221B Baker Street, London' or 'Near Ponte Vecchio, Florence'). Include if known.",
    )
    filming_scene: Optional[str] = Field(
        default=None,
        description="If this is a filming location: describe the specific scene or sequence filmed here (e.g., 'The opening chase scene from the 2005 film adaptation was filmed on these steps'). Leave null if not a filming location.",
    )
    time_of_day: str = Field(
        description='Best time to visit: morning, afternoon, evening, full_day'
    )
    notes: Optional[str] = Field(
        default=None, description="Practical tips for visiting"
    )
    source: Literal["composed", "expansion"] = Field(
        default="composed",
        description="Whether this stop was part of the original itinerary or added via expansion",
    )
    match_type: Literal["literal", "historical", "thematic", "vibe"] = Field(
        default="vibe",
        description=(
            "How this place connects to the book, so the user can calibrate trust: "
            "'literal' = the book explicitly names this exact place; "
            "'historical' = a real place tied to the book's documented history, author, or "
            "real-world events it depicts; "
            "'thematic' = a real place that strongly embodies the book's themes/setting but is "
            "not named in it; "
            "'vibe' = an atmospheric/'similar feel' suggestion. "
            "When unsure, choose the WEAKER claim (prefer 'vibe' over 'literal') — never overstate. "
            "Defaults to 'vibe' so an unlabelled payload degrades to the weakest, safest claim."
        ),
    )
    grounding_source: Optional[str] = Field(
        default=None,
        description=(
            "Optional citation/evidence for a 'literal' or 'historical' match — e.g. a chapter "
            "reference, the documented fact, or the source that ties this place to the book. "
            "Leave null for 'thematic'/'vibe' matches or when no specific source is available."
        ),
    )


class CityPlan(BaseModel):
    """Travel plan for a specific city"""

    name: str = Field(description="City name")
    country: str = Field(description="Country name")
    days_suggested: int = Field(
        description="Suggested number of days to spend", ge=1, le=7
    )

    @field_validator("days_suggested", mode="before")
    @classmethod
    def coerce_days_to_int(cls, v: object) -> int:
        return round(float(v))
    overview: str = Field(description="Brief overview of what to expect in this city")
    stops: List[CityStop] = Field(description="Places to visit in this city")


class TripItinerary(BaseModel):
    """Complete travel itinerary"""

    cities: List[CityPlan] = Field(description="City-by-city travel plans")
    summary_text: str = Field(description="Engaging overview of the entire trip")


class SuggestionChip(BaseModel):
    """A contextual suggestion chip shown to the user after receiving an itinerary."""

    id: str = Field(
        default="",
        description="Server-assigned stable identifier (uuid4). Leave empty — the server stamps this.",
    )
    label: str = Field(
        description="Short chip label shown in the UI (2-4 words, e.g. 'Add restaurants nearby')"
    )
    # MYS-494: deliberately left unbounded (no min_length/max_length).
    # This model is a structured-output response schema for the composer
    # and expansion-formatter agents (see core/executor.py's
    # _clamp_action_prompt docstring) -- a max_length here would fail
    # generation validation on an overlong LLM output rather than
    # truncating it. The 500-char bound lives at persist time instead
    # (core/executor.py::WorkflowExecutor._clamp_action_prompt, applied
    # in _persist_suggestions), which is guaranteed to run: ADK writes
    # this model's raw dict into session state, so a field_validator/
    # computed_field added here would not. Don't "fix" this field
    # directly -- read the ticket first.
    action_prompt: str = Field(
        description="Instruction passed to the expansion agent when this chip is clicked (10-30 words)"
    )


class ComposerEnvelope(BaseModel):
    """Wrapper returned by composer/formatter agents. Split by executor before persisting."""

    itinerary: TripItinerary = Field(description="The full trip itinerary")
    suggestions: List[SuggestionChip] = Field(
        default_factory=list,
        description="2-4 contextual suggestion chips for follow-up expansions",
    )


class ExpansionResult(BaseModel):
    """Result from the expansion agent: new places + follow-up suggestions."""

    parent_city: str = Field(
        description="Name of the city these new places belong to (must match a city in the existing itinerary)"
    )
    places: List[CityStop] = Field(
        description="3-5 new places to add to the itinerary"
    )
    suggestions: List[SuggestionChip] = Field(
        default_factory=list,
        description="2-4 contextual follow-up suggestion chips",
    )
