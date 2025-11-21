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
