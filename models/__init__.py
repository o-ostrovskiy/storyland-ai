"""
Pydantic data models for StoryLand AI.

This package contains type-safe data structures for:
- Book metadata and context
- Discovery results (cities, landmarks, author sites)
- Travel itineraries
- User preferences
"""

from .book import BookMetadata, BookContext, BookInfo
from .discovery import (
    CityInfo,
    CityDiscovery,
    LandmarkInfo,
    LandmarkDiscovery,
    AuthorSiteInfo,
    AuthorSites,
)
from .itinerary import CityStop, CityPlan, TripItinerary
from .preferences import TravelPreferences

__all__ = [
    # Book models
    "BookMetadata",
    "BookContext",
    "BookInfo",
    # Discovery models
    "CityInfo",
    "CityDiscovery",
    "LandmarkInfo",
    "LandmarkDiscovery",
    "AuthorSiteInfo",
    "AuthorSites",
    # Itinerary models
    "CityStop",
    "CityPlan",
    "TripItinerary",
    # Preferences
    "TravelPreferences",
]
