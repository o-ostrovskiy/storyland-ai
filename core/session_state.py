"""
Typed session state access.

Wraps the ADK session.state dict with typed properties, eliminating
magic string key access scattered across the codebase.
"""

from typing import Optional, List


class SessionStateKeys:
    """Constants for all session state keys used by agents and orchestration."""

    BOOK_TITLE = "book_title"
    AUTHOR = "author"
    BOOK_METADATA = "book_metadata"
    BOOK_CONTEXT = "book_context"
    READER_PROFILE = "reader_profile"
    CITY_DISCOVERY = "city_discovery"
    LANDMARK_DISCOVERY = "landmark_discovery"
    AUTHOR_SITES = "author_sites"
    REGION_ANALYSIS = "region_analysis"
    SELECTED_REGIONS = "selected_regions"
    FINAL_ITINERARY = "final_itinerary"
    USER_PREFERENCES = "user:preferences"


class SessionStateAccessor:
    """Typed wrapper around session.state dict.

    Usage:
        state = SessionStateAccessor(session.state)
        metadata = state.book_metadata  # typed access, no magic strings
        state.selected_regions = [...]  # typed write
    """

    def __init__(self, state: dict):
        self._state = state

    @property
    def book_metadata(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.BOOK_METADATA)

    @book_metadata.setter
    def book_metadata(self, value: dict) -> None:
        self._state[SessionStateKeys.BOOK_METADATA] = value

    @property
    def region_analysis(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.REGION_ANALYSIS)

    @property
    def regions(self) -> List[dict]:
        ra = self.region_analysis or {}
        return ra.get("regions", [])

    @property
    def analysis_note(self) -> str:
        ra = self.region_analysis or {}
        return ra.get("analysis_note", "")

    @property
    def selected_regions(self) -> List[dict]:
        return self._state.get(SessionStateKeys.SELECTED_REGIONS, [])

    @selected_regions.setter
    def selected_regions(self, value: List[dict]) -> None:
        self._state[SessionStateKeys.SELECTED_REGIONS] = value

    @property
    def final_itinerary(self) -> Optional[object]:
        return self._state.get(SessionStateKeys.FINAL_ITINERARY)

    @property
    def user_preferences(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.USER_PREFERENCES)

    @property
    def book_title(self) -> str:
        bm = self.book_metadata or {}
        return bm.get("book_title", "")

    @property
    def author(self) -> str:
        bm = self.book_metadata or {}
        return bm.get("author", "")
