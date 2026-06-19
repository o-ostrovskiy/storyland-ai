"""
Workflow orchestrator.

Creates workflows that coordinate all agents to produce complete literary
travel itineraries.

Two-phase architecture (HTTP API with HITL):
1. Discovery workflow - finds locations and groups into travel regions
2. Composition workflow - creates itinerary for selected region(s)

WHY TWO PHASES WITH HITL?
- Problem: Books like "Gone with the Wind" span Georgia (USA) and have author
  sites in Atlanta. Auto-generating itineraries for ALL regions would create
  impractical multi-continent trips.
- Solution: After discovery, show user region options (e.g., "England", "Scotland")
  and let them choose which region(s) to explore. This prevents wasting tokens
  on unwanted regions and gives users control over trip scope.
- Trade-off: Requires human input (not fully autonomous) but produces much more
  practical and personalized itineraries.
"""

from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.tools import google_search

from .book_context_agent import create_book_context_pipeline
from .discovery_agents import (
    create_city_pipeline,
    create_landmark_pipeline,
    create_author_pipeline,
)
from .book_recommendation_agent import create_book_recommendation_pipeline
from .place_to_book_agent import create_place_to_book_pipeline
from .expansion_agent import create_expansion_pipeline
from .local_atmosphere_agent import create_local_atmosphere_pipeline
from .trip_composer_agent import create_trip_composer_agent
from .reader_profile_agent import create_reader_profile_agent
from .region_analyzer_agent import create_region_analyzer_agent
from .prompts import AgentPrompts, load_prompts


def create_discovery_workflow(
    model,
    book_title: str,
    author: str,
    prompts: AgentPrompts | None = None,
):
    """
    Create the discovery workflow that finds locations and analyzes regions.

    This workflow runs after the book metadata has been confirmed and before
    user region selection. It discovers cities, landmarks, and author sites,
    then groups them into practical travel regions for the user to choose from.

    Architecture:
        SequentialAgent (discovery_workflow)
        ├─ book_context_pipeline [research → format] → state["book_context"]
        ├─ reader_profile_agent [read preferences] → state["reader_profile"]
        ├─ ParallelAgent (parallel_discovery) ⚡ CONCURRENT
        │  ├─ city_pipeline [research → format] → state["city_discovery"]
        │  ├─ landmark_pipeline [research → format] → state["landmark_discovery"]
        │  └─ author_pipeline [research → format] → state["author_sites"]
        └─ region_analyzer_agent → state["region_analysis"]

    Args:
        model: The LLM model to use
        book_title: Exact book title (pre-confirmed by caller)
        author: Exact author name (pre-confirmed by caller)
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        SequentialAgent orchestrating the discovery workflow
    """
    if prompts is None:
        prompts = load_prompts()

    # Create pipelines with exact book info
    book_context_pipeline = create_book_context_pipeline(
        model, google_search, book_title=book_title, author=author, prompts=prompts
    )

    city_pipeline = create_city_pipeline(model, google_search, prompts=prompts)
    landmark_pipeline = create_landmark_pipeline(model, google_search, prompts=prompts)
    author_pipeline = create_author_pipeline(model, google_search, prompts=prompts)

    reader_profile = create_reader_profile_agent(model, prompts=prompts)
    region_analyzer = create_region_analyzer_agent(model, prompts=prompts)

    # Create parallel discovery agent for concurrent execution
    # WHY PARALLEL: Running city/landmark/author agents concurrently provides 3x speedup
    # (15s vs 45s) with no additional token cost. Each agent makes independent Google
    # Search queries that don't depend on each other's results.
    parallel_discovery = ParallelAgent(
        name="parallel_discovery",
        sub_agents=[city_pipeline, landmark_pipeline, author_pipeline],
    )

    # Build discovery workflow
    # WHY SEQUENTIAL: Each stage depends on previous stage's output:
    # - book_context provides setting/theme → discovery agents use this context
    # - reader_profile provides preferences → trip composer uses these
    # - parallel_discovery provides locations → region_analyzer groups them
    sub_agents = [
        book_context_pipeline,
        reader_profile,
        parallel_discovery,
        region_analyzer,
    ]

    return SequentialAgent(
        name="discovery_workflow",
        sub_agents=sub_agents,
    )


def create_local_atmosphere_workflow(
    model,
    book_title: str,
    author: str,
    location_label: str,
    radius_km: int,
    prompts: AgentPrompts | None = None,
):
    """
    Create the local-atmosphere workflow (single-phase).

    Used when the reader cannot travel to the book's actual setting and wants
    an itinerary near their current location whose mood and sensory character
    evoke the book.

    Architecture:
        SequentialAgent (local_atmosphere_workflow)
        ├─ book_context_pipeline [research → format] → state["book_context"]
        ├─ reader_profile_agent → state["reader_profile"]
        └─ local_atmosphere_pipeline [research → format] → state["final_itinerary"]

    There is no city/landmark/author/region discovery: those agents look at the
    book's geography, which is irrelevant here.

    Args:
        model: The LLM model to use.
        book_title: Exact book title.
        author: Exact author name.
        location_label: Human-readable user location (e.g. "New York, NY 10013").
        radius_km: Maximum distance in km from the user's location.
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        SequentialAgent orchestrating the local-atmosphere workflow.
    """
    if prompts is None:
        prompts = load_prompts()

    book_context_pipeline = create_book_context_pipeline(
        model, google_search, book_title=book_title, author=author, prompts=prompts
    )
    reader_profile = create_reader_profile_agent(model, prompts=prompts)
    local_pipeline = create_local_atmosphere_pipeline(
        model,
        google_search,
        location_label=location_label,
        radius_km=radius_km,
        prompts=prompts,
    )

    return SequentialAgent(
        name="local_atmosphere_workflow",
        sub_agents=[book_context_pipeline, reader_profile, local_pipeline],
    )


def create_expansion_workflow(
    model,
    google_search_tool,
    book_title: str,
    author: str,
    parent_city: str,
    action_prompt: str,
    existing_places: str,
    prompts: AgentPrompts | None = None,
):
    """
    Create the expansion workflow that adds new places to an existing itinerary city.

    Called after composition when the user clicks a suggestion chip. Runs a
    researcher → formatter pipeline scoped to the chosen city and action.

    Architecture:
        SequentialAgent (expansion_workflow)
        └─ expansion_pipeline
           ├─ expansion_researcher [google_search] → research notes
           └─ expansion_formatter [output_schema=ExpansionResult] → state["last_expansion"]

    Args:
        model: The LLM model to use.
        google_search_tool: The Google Search tool.
        book_title: Exact book title.
        author: Exact author name.
        parent_city: City where new places should be located.
        action_prompt: The expansion instruction from the suggestion chip.
        existing_places: Newline-separated "Name (City)" strings to avoid repeating.
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        SequentialAgent orchestrating the expansion pipeline.
    """
    if prompts is None:
        prompts = load_prompts()

    expansion_pipeline = create_expansion_pipeline(
        model,
        google_search_tool,
        book_title=book_title,
        author=author,
        parent_city=parent_city,
        action_prompt=action_prompt,
        existing_places=existing_places,
        prompts=prompts,
    )

    return SequentialAgent(
        name="expansion_workflow",
        sub_agents=[expansion_pipeline],
    )


def create_book_recommendation_workflow(
    model,
    google_search_tool,
    book_title: str,
    author: str,
    destinations: str,
    themes: str,
    prompts: AgentPrompts | None = None,
):
    """
    Create the book recommendation workflow.

    Called after composition when the user clicks the "Find books like this" chip.
    Runs a single agent that searches for and returns 5 book recommendations.

    Architecture:
        SequentialAgent (book_recommendation_workflow)
        └─ book_recommendation_pipeline
           ├─ book_recommendation_researcher [google_search] → research notes
           └─ book_recommendation_formatter [output_schema=BookRecommendationsResult]
              → state["last_book_recommendations"]

    Args:
        model: The LLM model to use.
        google_search_tool: The Google Search tool.
        book_title: Exact book title.
        author: Exact author name.
        destinations: Comma-separated city names from the user's itinerary.
        themes: Comma-separated themes from the book context.
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        SequentialAgent orchestrating the book recommendation pipeline.
    """
    if prompts is None:
        prompts = load_prompts()

    pipeline = create_book_recommendation_pipeline(
        model,
        google_search_tool,
        book_title=book_title,
        author=author,
        destinations=destinations,
        themes=themes,
        prompts=prompts,
    )

    return SequentialAgent(
        name="book_recommendation_workflow",
        sub_agents=[pipeline],
    )


def create_composition_workflow(model, prompts: AgentPrompts | None = None):
    """
    Create the composition workflow that generates the final itinerary.

    This workflow runs after the user has selected a region.
    It expects the selected region to be stored in session state as "selected_region".

    Architecture:
        SequentialAgent (composition_workflow)
        └─ trip_composer_agent → state["final_itinerary"]

    Args:
        model: The LLM model to use
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        SequentialAgent orchestrating the composition workflow
    """
    if prompts is None:
        prompts = load_prompts()

    trip_composer = create_trip_composer_agent(model, prompts=prompts)

    return SequentialAgent(
        name="composition_workflow",
        sub_agents=[trip_composer],
    )


def create_place_to_book_workflow(
    model,
    google_search_tool,
    place: str,
    prompts: AgentPrompts | None = None,
):
    """
    Create the place→book reverse-routing workflow (AI candidate layer).

    The reverse of discovery: a destination is the input and grounded,
    literal/vibe-labelled book candidates are the output. Wraps the
    place_to_book pipeline (researcher [google_search] → formatter
    [output_schema=PlaceToBookCandidates]).

    Isolated capability — not yet wired into the HTTP API; the BE place→book
    endpoint will call it (PR 3 of the reverse-flow feature).

    Architecture:
        SequentialAgent (place_to_book_workflow)
        └─ place_to_book_pipeline
           ├─ place_to_book_researcher [google_search] → research notes
           └─ place_to_book_formatter [output_schema=PlaceToBookCandidates]
              → state["last_place_to_book"]

    Args:
        model: The LLM model to use.
        google_search_tool: The Google Search tool.
        place: Free-text destination.
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        SequentialAgent orchestrating the place→book pipeline.
    """
    if prompts is None:
        prompts = load_prompts()

    pipeline = create_place_to_book_pipeline(
        model, google_search_tool, place=place, prompts=prompts
    )

    return SequentialAgent(
        name="place_to_book_workflow",
        sub_agents=[pipeline],
    )
