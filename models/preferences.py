"""
User preferences Pydantic models.

Contains models for storing and managing user travel preferences and history.
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class TravelPreferences(BaseModel):
    """User travel preferences for personalized itineraries"""

    prefers_museums: bool = Field(
        default=True, description="User enjoys visiting museums"
    )
    travels_with_kids: bool = Field(
        default=False, description="User travels with children"
    )
    budget: Literal["budget", "moderate", "luxury"] = Field(
        default="moderate", description="Travel budget level"
    )
    favorite_genres: List[str] = Field(
        default_factory=list, description="Favorite book genres"
    )
    favorite_authors: List[str] = Field(
        default_factory=list, description="Favorite authors"
    )
    dietary_restrictions: List[str] = Field(
        default_factory=list, description="Dietary restrictions or preferences"
    )
    accessibility_needs: bool = Field(
        default=False, description="Requires accessibility accommodations"
    )
    preferred_pace: Literal["relaxed", "moderate", "fast-paced"] = Field(
        default="moderate", description="Preferred travel pace"
    )
