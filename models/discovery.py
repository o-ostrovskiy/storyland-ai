"""
Discovery-related Pydantic models.

Contains models for cities, landmarks, and author sites discovered during
the research phase of itinerary creation.
"""

from pydantic import BaseModel, Field, computed_field
from typing import List, Optional

from .place_key import mint_checked_place_key


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
    city: Optional[str] = Field(
        default=None,
        description="City where landmark is located. Null for non-urban stops (trailheads, scenic overlooks, route stops)."
    )
    region: Optional[str] = Field(
        default=None,
        description="Region or area for non-urban stops (e.g., 'Sierra Nevada, California', 'Pacific Crest Trail, Section C')."
    )
    connection: str = Field(description="How this landmark relates to the book")
    landmark_type: Optional[str] = Field(
        default=None,
        description="Type of stop: 'mentioned in book', 'related to setting', 'suggested for atmosphere', 'scenic_stop', 'route_point'."
    )


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
    """A practical travel region grouping nearby cities.

    ``region_id`` is an ORDINAL WITHIN ONE RESPONSE, not an identity: it is
    meaningless across two jobs (see models/place_key.py). The cross-job identity
    is ``place_key``, minted from the structured geo fields below — never from
    ``region_name``, which is prose.
    """

    region_id: int = Field(description="Unique identifier for the region (1, 2, 3...)")
    region_name: str = Field(
        description="Descriptive name for the region (e.g., 'New England, USA', 'Western Europe')"
    )
    country_code: Optional[str] = Field(
        default=None,
        description=(
            "ISO 3166-1 alpha-2 country code of this region's primary locality, "
            "uppercase (e.g. 'FR', 'US', 'JP'). Omit ONLY if genuinely unknown — never guess."
        ),
    )
    primary_locality: Optional[str] = Field(
        default=None,
        description=(
            "The single principal city this region is anchored on, as a bare place name "
            "with no country and no qualifier (e.g. 'Paris', 'Boston', 'Kyoto'). "
            "It MUST be one of the cities listed in `cities`."
        ),
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def place_key(self) -> Optional[str]:
        """Canonical cross-job identity, DERIVED — never emitted by the model.

        A computed field is absent from the *validation* JSON schema (which is
        what ADK hands the LLM as ``output_schema``) and present in the
        *serialization* dump. So the model is asked for the two grounded
        descriptions it can actually know, and the identity every downstream
        intersection keys on is minted by us, deterministically. The model can
        neither invent a key nor collide two places by emitting the same one.

        Routed through ``mint_checked_place_key`` — the SAME checked seam
        ``core/regions.py``'s ``enrich_region_analysis`` uses — not
        ``mint_place_key`` directly. ``mint_place_key`` alone only refuses
        missing/invalid fields; it has no way to know whether
        ``primary_locality`` is even one of this region's own ``cities``. Two
        mint paths with two different rules is exactly how a caller ends up
        reading the unchecked answer (MYS-460 review). Today ADK writes the
        agent's raw parsed JSON dict into session state, so this property
        never materialises at runtime on its own — but the first caller that
        constructs a ``RegionOption`` and reads ``.place_key`` (a future PR2
        codepath, a test, a script) must get the checked answer, not a second
        set of rules.
        """
        cities = [c.model_dump() for c in self.cities]
        return mint_checked_place_key(self.country_code, self.primary_locality, cities)


class RegionAnalysis(BaseModel):
    """Analysis of discovered locations grouped into practical travel regions"""

    regions: List[RegionOption] = Field(
        description="List of practical travel regions, each containing cities that can be visited together"
    )
    analysis_note: str = Field(
        description="Brief explanation of why locations were grouped this way"
    )
