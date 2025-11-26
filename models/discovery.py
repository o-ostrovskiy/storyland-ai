"""
Discovery-related Pydantic models.

Contains models for cities, landmarks, and author sites discovered during
the research phase of itinerary creation.
"""

from pydantic import BaseModel, Field
from typing import List


class CityInfo(BaseModel):
    """Information about a city to visit"""

    name: str = Field(description="City name")
    country: str = Field(description="Country name")
    relevance: str = Field(description="How this city relates to the book")


class CityDiscovery(BaseModel):
    """Discovery results for cities"""

    cities: List[CityInfo] = Field(description="List of cities related to the book")


class LandmarkInfo(BaseModel):
    """Information about a landmark"""

    name: str = Field(description="Landmark or place name")
    city: str = Field(description="City where landmark is located")
    connection: str = Field(description="How this landmark relates to the book")


class LandmarkDiscovery(BaseModel):
    """Discovery results for landmarks"""

    landmarks: List[LandmarkInfo] = Field(
        description="List of landmarks related to the book"
    )


class AuthorSiteInfo(BaseModel):
    """Information about author-related sites"""

    name: str = Field(description="Site name")
    type: str = Field(description="Type of site (museum, birthplace, etc.)")
    city: str = Field(description="City where site is located")


class AuthorSites(BaseModel):
    """Discovery results for author-related sites"""

    author_sites: List[AuthorSiteInfo] = Field(
        description="List of author-related sites"
    )


class RegionCity(BaseModel):
    """A city within a travel region"""

    name: str = Field(description="City name")
    country: str = Field(description="Country name")


class RegionOption(BaseModel):
    """A practical travel region grouping nearby cities"""

    region_id: int = Field(description="Unique identifier for the region (1, 2, 3...)")
    region_name: str = Field(
        description="Descriptive name for the region (e.g., 'New England, USA', 'Western Europe')"
    )
    cities: List[RegionCity] = Field(description="Cities in this region")
    estimated_days: int = Field(
        description="Estimated total days to visit all cities in this region", ge=1, le=30
    )
    travel_note: str = Field(
        description="How to travel between cities (e.g., 'All cities accessible by car within 2-3 hours')"
    )
    highlights: str = Field(
        description="Key attractions or reasons to choose this region"
    )


class RegionAnalysis(BaseModel):
    """Analysis of discovered locations grouped into practical travel regions"""

    regions: List[RegionOption] = Field(
        description="List of practical travel regions, each containing cities that can be visited together"
    )
    analysis_note: str = Field(
        description="Brief explanation of why locations were grouped this way"
    )
