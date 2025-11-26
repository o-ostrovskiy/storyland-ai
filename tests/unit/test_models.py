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
    AuthorSites, AuthorSiteInfo,
    RegionCity, RegionOption, RegionAnalysis,
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


# =============================================================================
# RegionCity Tests
# =============================================================================

class TestRegionCity:
    """Tests for RegionCity model."""

    def test_valid_region_city(self):
        """Test creating valid RegionCity."""
        city = RegionCity(name="Boston", country="USA")
        assert city.name == "Boston"
        assert city.country == "USA"

    def test_region_city_requires_name(self):
        """Test RegionCity requires name field."""
        with pytest.raises(ValidationError):
            RegionCity(country="USA")

    def test_region_city_requires_country(self):
        """Test RegionCity requires country field."""
        with pytest.raises(ValidationError):
            RegionCity(name="Boston")


# =============================================================================
# RegionOption Tests
# =============================================================================

class TestRegionOption:
    """Tests for RegionOption model."""

    def test_valid_region_option(self):
        """Test creating valid RegionOption."""
        region = RegionOption(
            region_id=1,
            region_name="New England, USA",
            cities=[
                RegionCity(name="Boston", country="USA"),
                RegionCity(name="Providence", country="USA"),
            ],
            estimated_days=5,
            travel_note="All cities accessible by car within 2 hours",
            highlights="Historic sites, literary landmarks, coastal scenery",
        )
        assert region.region_id == 1
        assert region.region_name == "New England, USA"
        assert len(region.cities) == 2
        assert region.estimated_days == 5

    def test_region_option_days_validation_min(self):
        """Test RegionOption estimated_days minimum is 1."""
        with pytest.raises(ValidationError):
            RegionOption(
                region_id=1,
                region_name="Test",
                cities=[],
                estimated_days=0,  # Invalid: must be >= 1
                travel_note="Note",
                highlights="Highlights",
            )

    def test_region_option_days_validation_max(self):
        """Test RegionOption estimated_days maximum is 30."""
        with pytest.raises(ValidationError):
            RegionOption(
                region_id=1,
                region_name="Test",
                cities=[],
                estimated_days=31,  # Invalid: must be <= 30
                travel_note="Note",
                highlights="Highlights",
            )

    def test_region_option_empty_cities(self):
        """Test RegionOption with empty cities list."""
        region = RegionOption(
            region_id=1,
            region_name="Empty Region",
            cities=[],
            estimated_days=1,
            travel_note="No cities",
            highlights="None",
        )
        assert region.cities == []

    def test_region_option_serialization(self):
        """Test RegionOption serialization round-trip."""
        region = RegionOption(
            region_id=1,
            region_name="Test Region",
            cities=[RegionCity(name="City", country="Country")],
            estimated_days=3,
            travel_note="Note",
            highlights="Highlights",
        )
        json_data = region.model_dump()
        restored = RegionOption(**json_data)
        assert restored.region_name == region.region_name
        assert len(restored.cities) == 1


# =============================================================================
# RegionAnalysis Tests
# =============================================================================

class TestRegionAnalysis:
    """Tests for RegionAnalysis model."""

    def test_valid_region_analysis(self):
        """Test creating valid RegionAnalysis."""
        analysis = RegionAnalysis(
            regions=[
                RegionOption(
                    region_id=1,
                    region_name="Region A",
                    cities=[RegionCity(name="CityA", country="CountryA")],
                    estimated_days=3,
                    travel_note="Note A",
                    highlights="Highlights A",
                ),
                RegionOption(
                    region_id=2,
                    region_name="Region B",
                    cities=[RegionCity(name="CityB", country="CountryB")],
                    estimated_days=4,
                    travel_note="Note B",
                    highlights="Highlights B",
                ),
            ],
            analysis_note="Cities grouped by geographic proximity",
        )
        assert len(analysis.regions) == 2
        assert analysis.analysis_note == "Cities grouped by geographic proximity"

    def test_region_analysis_empty_regions(self):
        """Test RegionAnalysis with no regions."""
        analysis = RegionAnalysis(
            regions=[],
            analysis_note="No regions found",
        )
        assert analysis.regions == []

    def test_region_analysis_serialization(self):
        """Test RegionAnalysis JSON serialization round-trip."""
        analysis = RegionAnalysis(
            regions=[
                RegionOption(
                    region_id=1,
                    region_name="Test",
                    cities=[],
                    estimated_days=1,
                    travel_note="Note",
                    highlights="Highlights",
                )
            ],
            analysis_note="Test analysis",
        )
        json_data = analysis.model_dump()
        restored = RegionAnalysis(**json_data)
        assert len(restored.regions) == 1
        assert restored.analysis_note == analysis.analysis_note
