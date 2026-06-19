"""
Unit tests for Pydantic models.

Tests validation, serialization, and edge cases for all data models.
"""

import pytest
from pydantic import ValidationError

from models.book import BookMetadata, BookContext, BookInfo, BookRecommendation, BookRecommendationsResult
from models.discovery import (
    CityDiscovery, CityInfo,
    LandmarkDiscovery, LandmarkInfo,
    AuthorSites, AuthorSiteInfo,
    RegionCity, RegionOption, RegionAnalysis,
)
from models.itinerary import (
    TripItinerary, CityPlan, CityStop,
    SuggestionChip, ComposerEnvelope, ExpansionResult,
)
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
        """Test LandmarkInfo requires name and connection."""
        with pytest.raises(ValidationError):
            LandmarkInfo(
                name="Tower",
                city="London"
                # Missing connection
            )

    def test_landmark_info_non_urban_stop(self):
        """Test LandmarkInfo accepts non-urban stops with region instead of city."""
        stop = LandmarkInfo(
            name="Forester Pass Trailhead",
            region="Sierra Nevada, California",
            connection="Key waypoint on Cheryl Strayed's PCT route in Wild"
        )
        assert stop.city is None
        assert stop.region == "Sierra Nevada, California"
        assert stop.landmark_type is None

    def test_landmark_info_with_landmark_type(self):
        """Test LandmarkInfo captures landmark_type for scenic/route stops."""
        stop = LandmarkInfo(
            name="Kennedy Meadows",
            region="Sequoia National Forest, California",
            connection="Resupply stop mentioned in Wild",
            landmark_type="route_point"
        )
        assert stop.landmark_type == "route_point"


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

    def test_city_stop_match_type_defaults_to_vibe(self):
        """Unlabelled stops degrade to the weakest claim ('vibe')."""
        stop = CityStop(
            name="Some Cafe",
            type="cafe",
            reason="A cosy spot that fits the book's mood",
            time_of_day="afternoon",
        )
        assert stop.match_type == "vibe"
        assert stop.grounding_source is None

    def test_city_stop_accepts_all_match_types(self):
        """All four match_type values are valid."""
        for mt in ["literal", "historical", "thematic", "vibe"]:
            stop = CityStop(
                name="Place",
                type="landmark",
                reason="Reason",
                time_of_day="morning",
                match_type=mt,
            )
            assert stop.match_type == mt

    def test_city_stop_rejects_invalid_match_type(self):
        """An out-of-enum match_type is rejected."""
        with pytest.raises(ValidationError):
            CityStop(
                name="Place",
                type="landmark",
                reason="Reason",
                time_of_day="morning",
                match_type="exact",
            )

    def test_city_stop_grounding_source_optional(self):
        """grounding_source can carry a citation for literal/historical matches."""
        stop = CityStop(
            name="221B Baker Street",
            type="landmark",
            reason="The address Sherlock Holmes is said to live at",
            time_of_day="morning",
            match_type="literal",
            grounding_source="Named explicitly throughout the Sherlock Holmes canon",
        )
        assert stop.match_type == "literal"
        assert "Sherlock" in stop.grounding_source

    def test_city_stop_serialization_includes_new_fields(self):
        """model_dump exposes the new fields so be/fe can forward and render them."""
        stop = CityStop(
            name="Place",
            type="museum",
            reason="Reason",
            time_of_day="evening",
            match_type="thematic",
        )
        dumped = stop.model_dump()
        assert dumped["match_type"] == "thematic"
        assert "grounding_source" in dumped
        assert dumped["grounding_source"] is None


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


# =============================================================================
# CityStop source field Tests
# =============================================================================

class TestCityStopSource:
    """Tests for CityStop.source field (expansion provenance)."""

    def test_default_source_is_composed(self):
        stop = CityStop(name="Place", type="museum", reason="Reason", time_of_day="morning")
        assert stop.source == "composed"

    def test_expansion_source(self):
        stop = CityStop(name="Place", type="cafe", reason="Reason", time_of_day="afternoon", source="expansion")
        assert stop.source == "expansion"

    def test_invalid_source_raises(self):
        with pytest.raises(ValidationError):
            CityStop(name="Place", type="cafe", reason="R", time_of_day="morning", source="unknown")

    def test_source_round_trips(self):
        stop = CityStop(name="Place", type="cafe", reason="R", time_of_day="morning", source="expansion")
        data = stop.model_dump()
        restored = CityStop(**data)
        assert restored.source == "expansion"


# =============================================================================
# SuggestionChip Tests
# =============================================================================

class TestSuggestionChip:
    """Tests for SuggestionChip model."""

    def test_minimal_chip(self):
        chip = SuggestionChip(label="Add restaurants", action_prompt="Find atmospheric restaurants near the stops that evoke the book's mood.")
        assert chip.label == "Add restaurants"
        assert chip.id == ""

    def test_chip_with_id(self):
        chip = SuggestionChip(id="abc-123", label="Hidden gems", action_prompt="Find lesser-known spots.")
        assert chip.id == "abc-123"

    def test_chip_requires_label(self):
        with pytest.raises(ValidationError):
            SuggestionChip(action_prompt="Find places.")

    def test_chip_requires_action_prompt(self):
        with pytest.raises(ValidationError):
            SuggestionChip(label="Add restaurants")

    def test_chip_serialization(self):
        chip = SuggestionChip(id="x", label="Bookshops", action_prompt="Find literary bookshops nearby.")
        data = chip.model_dump()
        assert data["id"] == "x"
        assert data["label"] == "Bookshops"


# =============================================================================
# ComposerEnvelope Tests
# =============================================================================

class TestComposerEnvelope:
    """Tests for ComposerEnvelope model."""

    def _make_itinerary(self):
        stop = CityStop(name="Baker St", type="landmark", reason="Holmes", time_of_day="morning")
        city = CityPlan(name="London", country="England", days_suggested=2, overview="Great city", stops=[stop])
        return TripItinerary(cities=[city], summary_text="A literary journey.")

    def test_valid_envelope(self):
        itinerary = self._make_itinerary()
        chip = SuggestionChip(label="Add restaurants", action_prompt="Find restaurants near Baker Street.")
        env = ComposerEnvelope(itinerary=itinerary, suggestions=[chip])
        assert env.itinerary.summary_text == "A literary journey."
        assert len(env.suggestions) == 1

    def test_envelope_default_suggestions(self):
        itinerary = self._make_itinerary()
        env = ComposerEnvelope(itinerary=itinerary)
        assert env.suggestions == []

    def test_envelope_requires_itinerary(self):
        with pytest.raises(ValidationError):
            ComposerEnvelope(suggestions=[])

    def test_envelope_round_trip(self):
        itinerary = self._make_itinerary()
        chip = SuggestionChip(id="x", label="Test", action_prompt="Test prompt.")
        env = ComposerEnvelope(itinerary=itinerary, suggestions=[chip])
        data = env.model_dump()
        restored = ComposerEnvelope(**data)
        assert restored.itinerary.summary_text == "A literary journey."
        assert restored.suggestions[0].id == "x"


# =============================================================================
# ExpansionResult Tests
# =============================================================================

class TestExpansionResult:
    """Tests for ExpansionResult model."""

    def _make_stop(self, name="New Café"):
        return CityStop(name=name, type="cafe", reason="Mood", time_of_day="afternoon", source="expansion")

    def test_valid_result(self):
        result = ExpansionResult(
            parent_city="London",
            places=[self._make_stop()],
            suggestions=[SuggestionChip(label="More cafés", action_prompt="Find more atmospheric cafés.")]
        )
        assert result.parent_city == "London"
        assert len(result.places) == 1
        assert result.places[0].source == "expansion"

    def test_result_default_suggestions(self):
        result = ExpansionResult(parent_city="Paris", places=[self._make_stop()])
        assert result.suggestions == []

    def test_result_requires_parent_city(self):
        with pytest.raises(ValidationError):
            ExpansionResult(places=[self._make_stop()])

    def test_result_requires_places(self):
        with pytest.raises(ValidationError):
            ExpansionResult(parent_city="London")

    def test_result_round_trip(self):
        result = ExpansionResult(
            parent_city="Edinburgh",
            places=[self._make_stop("The Elephant House")],
        )
        data = result.model_dump()
        restored = ExpansionResult(**data)
        assert restored.parent_city == "Edinburgh"
        assert restored.places[0].name == "The Elephant House"


# =============================================================================
# BookRecommendation Tests
# =============================================================================


class TestBookRecommendation:
    """Tests for BookRecommendation model."""

    def _make_rec(self, **kwargs):
        defaults = {
            "title": "A Tale of Two Cities",
            "author": "Charles Dickens",
            "reason": "Set in the same revolutionary Paris you will explore.",
            "recommendation_basis": "destination",
        }
        defaults.update(kwargs)
        return BookRecommendation(**defaults)

    def test_valid_recommendation(self):
        rec = self._make_rec()
        assert rec.title == "A Tale of Two Cities"
        assert rec.recommendation_basis == "destination"

    def test_optional_fields_default_to_none(self):
        rec = self._make_rec()
        assert rec.description is None
        assert rec.published_date is None
        assert rec.image_url is None

    def test_all_fields(self):
        rec = BookRecommendation(
            title="Les Misérables",
            author="Victor Hugo",
            description="Epic novel set in Paris.",
            published_date="1862",
            image_url="https://example.com/cover.jpg",
            reason="Also set in Paris during a turbulent era.",
            recommendation_basis="destination",
        )
        assert rec.description == "Epic novel set in Paris."
        assert rec.image_url == "https://example.com/cover.jpg"

    def test_invalid_recommendation_basis(self):
        with pytest.raises(ValidationError):
            self._make_rec(recommendation_basis="mood")

    def test_all_valid_bases(self):
        for basis in ("destination", "themes", "author"):
            rec = self._make_rec(recommendation_basis=basis)
            assert rec.recommendation_basis == basis

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            BookRecommendation(title="Only Title")

    def test_serialization_roundtrip(self):
        rec = self._make_rec()
        data = rec.model_dump()
        restored = BookRecommendation(**data)
        assert restored.title == rec.title
        assert restored.recommendation_basis == rec.recommendation_basis


class TestBookRecommendationsResult:
    """Tests for BookRecommendationsResult model."""

    def _make_rec(self, title="Book", basis="themes"):
        return {
            "title": title,
            "author": "Author",
            "reason": "A reason.",
            "recommendation_basis": basis,
        }

    def test_valid_result(self):
        result = BookRecommendationsResult(
            recommendations=[self._make_rec(f"Book {i}") for i in range(5)]
        )
        assert len(result.recommendations) == 5
        assert result.recommendations[0].title == "Book 0"

    def test_empty_recommendations_rejected(self):
        with pytest.raises(ValidationError):
            BookRecommendationsResult(recommendations=[])

    def test_three_to_five_accepted(self):
        # Floor relaxed from a hard 5 to REC_MIN_RESULTS (default 3): 3-5 valid,
        # so the tool-less formatter is never forced to invent a 5th book.
        for n in (3, 4, 5):
            result = BookRecommendationsResult(
                recommendations=[self._make_rec(f"Book {i}") for i in range(n)]
            )
            assert len(result.recommendations) == n

    def test_below_floor_rejected(self):
        with pytest.raises(ValidationError):
            BookRecommendationsResult(
                recommendations=[self._make_rec(f"Book {i}") for i in range(2)]
            )

    def test_more_than_five_rejected(self):
        with pytest.raises(ValidationError):
            BookRecommendationsResult(
                recommendations=[self._make_rec(f"Book {i}") for i in range(6)]
            )

    def test_missing_recommendations_raises(self):
        with pytest.raises(ValidationError):
            BookRecommendationsResult()

    def test_round_trip(self):
        result = BookRecommendationsResult(
            recommendations=[
                self._make_rec(f"Book {i}", "destination") for i in range(5)
            ]
        )
        data = result.model_dump()
        restored = BookRecommendationsResult(**data)
        assert len(restored.recommendations) == 5
        assert restored.recommendations[0].title == "Book 0"
