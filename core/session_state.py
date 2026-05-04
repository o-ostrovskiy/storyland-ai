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
    COMPOSER_ENVELOPE = "composer_envelope"
    LAST_SUGGESTIONS = "last_suggestions"
    EXPANSIONS = "expansions"
    EXPANSION_COUNT = "expansion_count"
    EXPANSION_IN_PROGRESS = "expansion_in_progress"
    LAST_EXPANSION = "last_expansion"
    LAST_BOOK_RECOMMENDATIONS = "last_book_recommendations"
    BOOK_RECOMMENDATION_COUNT = "book_recommendation_count"
    BOOK_RECS_IN_PROGRESS = "book_recs_in_progress"
    BOOK_RECOMMENDATION_CHIP_ID = "book_recommendation_chip_id"
    BOOK_RECOMMENDATION_CHIP = "book_recommendation_chip"
    USER_PREFERENCES = "user:preferences"
    USER_LOCATION = "user_location"
    JOB_FAILED = "job_failed"


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

    def clear_final_itinerary(self) -> None:
        """Remove a stale itinerary before a compose retry."""
        self._state.pop(SessionStateKeys.FINAL_ITINERARY, None)

    @property
    def user_preferences(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.USER_PREFERENCES)

    @property
    def failed(self) -> bool:
        return bool(self._state.get(SessionStateKeys.JOB_FAILED))

    @failed.setter
    def failed(self, value: bool) -> None:
        self._state[SessionStateKeys.JOB_FAILED] = value

    @property
    def book_title(self) -> str:
        bm = self.book_metadata or {}
        return bm.get("book_title", "") or self._state.get(SessionStateKeys.BOOK_TITLE, "")

    @property
    def author(self) -> str:
        bm = self.book_metadata or {}
        return bm.get("author", "") or self._state.get(SessionStateKeys.AUTHOR, "")

    @property
    def composer_envelope(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.COMPOSER_ENVELOPE)

    @property
    def last_suggestions(self) -> List[dict]:
        return self._state.get(SessionStateKeys.LAST_SUGGESTIONS, [])

    @property
    def expansions(self) -> List[dict]:
        return self._state.get(SessionStateKeys.EXPANSIONS, [])

    @property
    def expansion_count(self) -> int:
        return int(self._state.get(SessionStateKeys.EXPANSION_COUNT, 0))

    @property
    def expansion_in_progress(self) -> bool:
        return bool(self._state.get(SessionStateKeys.EXPANSION_IN_PROGRESS, False))

    @property
    def last_expansion(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.LAST_EXPANSION)

    @property
    def book_context(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.BOOK_CONTEXT)

    @property
    def last_book_recommendations(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.LAST_BOOK_RECOMMENDATIONS)

    @property
    def book_recommendation_count(self) -> int:
        return int(self._state.get(SessionStateKeys.BOOK_RECOMMENDATION_COUNT, 0))

    @property
    def book_recs_in_progress(self) -> bool:
        return bool(self._state.get(SessionStateKeys.BOOK_RECS_IN_PROGRESS, False))

    @property
    def book_recommendation_chip_id(self) -> Optional[str]:
        return self._state.get(SessionStateKeys.BOOK_RECOMMENDATION_CHIP_ID)

    @property
    def book_recommendation_chip(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.BOOK_RECOMMENDATION_CHIP)
