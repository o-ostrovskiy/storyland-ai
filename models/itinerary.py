"""
Itinerary-related Pydantic models.

Contains models for the final travel itinerary, including city plans and stops.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class CityStop(BaseModel):
    """A stop/place to visit in a city"""

    name: str = Field(description="Name of the place")
    type: str = Field(
        description="Type: landmark, museum, cafe, restaurant, bookstore, etc."
    )
    reason: str = Field(description="Why this place matters for the book")
    time_of_day: str = Field(
        description='Best time to visit: morning, afternoon, evening, full_day'
    )
    notes: Optional[str] = Field(
        default=None, description="Practical tips for visiting"
    )


class CityPlan(BaseModel):
    """Travel plan for a specific city"""

    name: str = Field(description="City name")
    country: str = Field(description="Country name")
    days_suggested: int = Field(
        description="Suggested number of days to spend", ge=1, le=7
    )
    overview: str = Field(description="Brief overview of what to expect in this city")
    stops: List[CityStop] = Field(description="Places to visit in this city")


class TripItinerary(BaseModel):
    """Complete travel itinerary"""

    cities: List[CityPlan] = Field(description="City-by-city travel plans")
    summary_text: str = Field(description="Engaging overview of the entire trip")
