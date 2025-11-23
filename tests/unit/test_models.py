"""
Unit tests for Pydantic models.

Tests validation, serialization, and edge cases for all data models.
"""

import pytest
from pydantic import ValidationError

from models.book import BookMetadata, BookContext, BookInfo
from models.discovery import (
    CityDiscovery, CityInfo,
    LandmarkDiscovery, LandmarkInfo,
    AuthorSites, AuthorSiteInfo
)
from models.itinerary import TripItinerary, CityPlan, CityStop
from models.preferences import TravelPreferences


# =============================================================================
# BookMetadata Tests
# =============================================================================

class TestBookMetadata:
    """Tests for BookMetadata model."""

    def test_valid_book_metadata(self, sample_book_metadata):
        """Test creating valid BookMetadata."""
        assert sample_book_metadata.book_title == "Pride and Prejudice"
        assert sample_book_metadata.author == "Jane Austen"
        assert "Fiction" in sample_book_metadata.categories

    def test_minimal_book_metadata(self):
        """Test BookMetadata with only required fields."""
        metadata = BookMetadata(
            book_title="Test Book",
            author="Test Author"
        )
        assert metadata.book_title == "Test Book"
        assert metadata.description is None
        assert metadata.categories == []

    def test_book_metadata_serialization(self, sample_book_metadata):
        """Test BookMetadata JSON serialization."""
        json_str = sample_book_metadata.model_dump_json()
        assert "Pride and Prejudice" in json_str
        assert "Jane Austen" in json_str

    def test_book_metadata_missing_required_fields(self):
        """Test BookMetadata fails without required fields."""
        with pytest.raises(ValidationError):
            BookMetadata(book_title="Only Title")

    def test_book_metadata_from_dict(self):
        """Test creating BookMetadata from dictionary."""
        data = {
            "book_title": "Test",
            "author": "Author",
            "categories": ["Fiction"]
        }
        metadata = BookMetadata(**data)
        assert metadata.categories == ["Fiction"]


# =============================================================================
# BookContext Tests
# =============================================================================

class TestBookContext:
    """Tests for BookContext model."""

    def test_valid_book_context(self, sample_book_context):
        """Test creating valid BookContext."""
        assert "Hertfordshire, England" in sample_book_context.primary_locations
        assert "Regency" in sample_book_context.time_period

    def test_book_context_required_fields(self):
        """Test BookContext requires all fields."""
        with pytest.raises(ValidationError):
            BookContext(
                primary_locations=["London"],
                time_period="19th century"
                # Missing themes
            )

    def test_book_context_empty_locations(self):
        """Test BookContext with empty locations list."""
        context = BookContext(
            primary_locations=[],
            time_period="Modern day",
            themes=["Adventure"]
        )
        assert context.primary_locations == []


# =============================================================================
# BookInfo Tests
# =============================================================================

class TestBookInfo:
    """Tests for BookInfo model (internal API representation)."""

    def test_valid_book_info(self, sample_book_info):
        """Test creating valid BookInfo."""
        assert sample_book_info.title == "Pride and Prejudice"
        assert "Jane Austen" in sample_book_info.authors

    def test_book_info_empty_authors(self):
        """Test BookInfo with empty authors list."""
        info = BookInfo(
            title="Anonymous Work",
            authors=[]
        )
        assert info.authors == []

    def test_book_info_optional_fields(self):
        """Test BookInfo with only required fields."""
        info = BookInfo(
            title="Test",
            authors=["Author"]
        )
        assert info.description is None
        assert info.published_date is None
        assert info.image_url is None


# =============================================================================
# CityDiscovery Tests
# =============================================================================

class TestCityDiscovery:
    """Tests for CityDiscovery and CityInfo models."""

    def test_valid_city_discovery(self, sample_city_discovery):
        """Test creating valid CityDiscovery."""
        assert len(sample_city_discovery.cities) == 2
        assert sample_city_discovery.cities[0].name == "Bath"

    def test_city_info_required_fields(self):
        """Test CityInfo requires all fields."""
        city = CityInfo(
            name="Paris",
            country="France",
            relevance="Setting of the novel"
        )
        assert city.name == "Paris"

    def test_city_discovery_empty_list(self):
        """Test CityDiscovery with no cities."""
        discovery = CityDiscovery(cities=[])
        assert discovery.cities == []


# =============================================================================
# LandmarkDiscovery Tests
# =============================================================================

class TestLandmarkDiscovery:
    """Tests for LandmarkDiscovery and LandmarkInfo models."""

    def test_valid_landmark_discovery(self, sample_landmark_discovery):
        """Test creating valid LandmarkDiscovery."""
        assert len(sample_landmark_discovery.landmarks) == 2
        assert "Chatsworth" in sample_landmark_discovery.landmarks[0].name

    def test_landmark_info_required_fields(self):
        """Test LandmarkInfo requires all fields."""
        with pytest.raises(ValidationError):
            LandmarkInfo(
                name="Tower",
                city="London"
                # Missing connection
            )


# =============================================================================
# AuthorSites Tests
# =============================================================================

class TestAuthorSites:
    """Tests for AuthorSites and AuthorSiteInfo models."""

    def test_valid_author_sites(self, sample_author_sites):
        """Test creating valid AuthorSites."""
        assert len(sample_author_sites.author_sites) == 2
        assert sample_author_sites.author_sites[0].type == "museum"

    def test_author_site_info_all_fields(self):
        """Test AuthorSiteInfo with all fields."""
        site = AuthorSiteInfo(
            name="Shakespeare's Birthplace",
            type="birthplace",
            city="Stratford-upon-Avon"
        )
        assert site.name == "Shakespeare's Birthplace"


# =============================================================================
# CityStop Tests
# =============================================================================

class TestCityStop:
    """Tests for CityStop model."""

    def test_valid_city_stop(self, sample_city_stop):
        """Test creating valid CityStop."""
        assert sample_city_stop.name == "Chatsworth House"
        assert sample_city_stop.time_of_day == "morning"

    def test_city_stop_optional_notes(self):
        """Test CityStop without notes."""
        stop = CityStop(
            name="Test Place",
            type="museum",
            reason="Test reason",
            time_of_day="afternoon"
        )
        assert stop.notes is None

    def test_city_stop_all_time_of_day_values(self):
        """Test CityStop accepts various time_of_day values."""
        for time in ["morning", "afternoon", "evening", "full_day"]:
            stop = CityStop(
                name="Place",
                type="landmark",
                reason="Reason",
                time_of_day=time
            )
            assert stop.time_of_day == time


# =============================================================================
# CityPlan Tests
# =============================================================================

class TestCityPlan:
    """Tests for CityPlan model."""

    def test_valid_city_plan(self, sample_city_plan):
        """Test creating valid CityPlan."""
        assert sample_city_plan.name == "Bakewell"
        assert sample_city_plan.days_suggested == 2
        assert len(sample_city_plan.stops) == 1

    def test_city_plan_days_validation_min(self):
        """Test CityPlan days_suggested minimum is 1."""
        with pytest.raises(ValidationError):
            CityPlan(
                name="City",
                country="Country",
                days_suggested=0,  # Invalid: must be >= 1
                overview="Overview",
                stops=[]
            )

    def test_city_plan_days_validation_max(self):
        """Test CityPlan days_suggested maximum is 7."""
        with pytest.raises(ValidationError):
            CityPlan(
                name="City",
                country="Country",
                days_suggested=10,  # Invalid: must be <= 7
                overview="Overview",
                stops=[]
            )

    def test_city_plan_valid_days_range(self):
        """Test CityPlan accepts valid days_suggested values."""
        for days in range(1, 8):  # 1 to 7
            plan = CityPlan(
                name="City",
                country="Country",
                days_suggested=days,
                overview="Overview",
                stops=[]
            )
            assert plan.days_suggested == days


# =============================================================================
# TripItinerary Tests
# =============================================================================

class TestTripItinerary:
    """Tests for TripItinerary model."""

    def test_valid_trip_itinerary(self, sample_trip_itinerary):
        """Test creating valid TripItinerary."""
        assert len(sample_trip_itinerary.cities) == 1
        assert "Elizabeth Bennet" in sample_trip_itinerary.summary_text

    def test_trip_itinerary_empty_cities(self):
        """Test TripItinerary with empty cities list."""
        itinerary = TripItinerary(
            cities=[],
            summary_text="An empty itinerary."
        )
        assert itinerary.cities == []

    def test_trip_itinerary_serialization(self, sample_trip_itinerary):
        """Test TripItinerary serialization round-trip."""
        json_data = sample_trip_itinerary.model_dump()
        restored = TripItinerary(**json_data)
        assert restored.summary_text == sample_trip_itinerary.summary_text
        assert len(restored.cities) == len(sample_trip_itinerary.cities)


# =============================================================================
# TravelPreferences Tests
# =============================================================================

class TestTravelPreferences:
    """Tests for TravelPreferences model."""

    def test_valid_preferences(self, sample_preferences):
        """Test creating valid TravelPreferences."""
        assert sample_preferences.budget == "moderate"
        assert sample_preferences.preferred_pace == "relaxed"

    def test_preferences_defaults(self):
        """Test TravelPreferences default values."""
        prefs = TravelPreferences()
        assert prefs.prefers_museums is True
        assert prefs.travels_with_kids is False
        assert prefs.budget == "moderate"
        assert prefs.preferred_pace == "moderate"
        assert prefs.accessibility_needs is False

    def test_preferences_budget_validation(self):
        """Test TravelPreferences budget must be valid literal."""
        with pytest.raises(ValidationError):
            TravelPreferences(budget="cheap")  # Invalid: not in Literal

    def test_preferences_valid_budget_values(self):
        """Test all valid budget values."""
        for budget in ["budget", "moderate", "luxury"]:
            prefs = TravelPreferences(budget=budget)
            assert prefs.budget == budget

    def test_preferences_pace_validation(self):
        """Test TravelPreferences pace must be valid literal."""
        with pytest.raises(ValidationError):
            TravelPreferences(preferred_pace="slow")  # Invalid

    def test_preferences_valid_pace_values(self):
        """Test all valid pace values."""
        for pace in ["relaxed", "moderate", "fast-paced"]:
            prefs = TravelPreferences(preferred_pace=pace)
            assert prefs.preferred_pace == pace

    def test_preferences_from_dict(self, sample_preferences_dict):
        """Test creating TravelPreferences from dictionary."""
        prefs = TravelPreferences(**sample_preferences_dict)
        assert prefs.budget == "moderate"
        assert "Jane Austen" in prefs.favorite_authors
