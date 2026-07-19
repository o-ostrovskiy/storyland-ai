"""
Workflow orchestrator.

Builds the ADK 2 graph workflows (``google.adk.workflow.Workflow``) that
coordinate all agents. Agents themselves are plain ``LlmAgent`` pairs built by
the ``create_*_agents`` factories; this module owns ALL composition (chains,
fan-out, fan-in) as explicit graph edges — no Sequential/ParallelAgent
templates (removed in the ADK 2 graph rewrite, ADR #24).

NAMING: the primary capability — a book in, real places out — is
``book_to_place``, the symmetric counterpart of the ``place_to_book`` reverse
flow. It runs as two phase workflows because of the human-in-the-loop region
selection between them (ADR #1):

1. ``book_to_place_discovery``  — finds locations, groups into travel regions
2. ``book_to_place_composition`` — creates the itinerary for selected region(s)

WHY TWO PHASES WITH HITL?
- Problem: Books like "Gone with the Wind" span Georgia (USA) and have author
  sites in Atlanta. Auto-generating itineraries for ALL regions would create
  impractical multi-continent trips.
- Solution: After discovery, show user region options (e.g., "England",
  "Scotland") and let them choose which region(s) to explore. This prevents
  wasting tokens on unwanted regions and gives users control over trip scope.
- Trade-off: Requires human input (not fully autonomous) but produces much
  more practical and personalized itineraries.
"""

from google.adk.tools import google_search
from google.adk.workflow import JoinNode, START, Workflow

from .book_context_agent import create_book_context_agents
from .discovery_agents import (
    create_city_agents,
    create_landmark_agents,
    create_author_agents,
)
from .book_recommendation_agent import create_book_recommendation_agents
from .place_to_book_agent import create_place_to_book_agents
from .expansion_agent import create_expansion_agents
from .local_atmosphere_agent import create_local_atmosphere_agents
from .trip_composer_agent import create_trip_composer_agent
from .region_analyzer_agent import create_region_analyzer_agent
from .prompts import AgentPrompts, load_prompts


def create_book_to_place_discovery_workflow(
    model,
    book_title: str,
    author: str,
    vibe: str | None = None,
    taste_context: dict | None = None,
    prompts: AgentPrompts | None = None,
):
    """
    Create the book→place discovery workflow (phase 1 of the primary flow).

    Runs after the book metadata has been confirmed and before user region
    selection. It discovers cities, landmarks, and author sites, then groups
    them into practical travel regions for the user to choose from.

    Graph:
        START → book_context_researcher → book_context_formatter
              → ⚡ fan-out: city_researcher   → city_formatter   ─┐
                          landmark_researcher → landmark_formatter ┼→ join
                          author_researcher   → author_formatter  ─┘
              → region_analyzer                                  (fan-in)

    Formatters write state["book_context"] / ["city_discovery"] /
    ["landmark_discovery"] / ["author_sites"]; region_analyzer writes
    state["region_analysis"]. The JoinNode is required: a plain node fires on
    ANY predecessor, and region_analyzer must wait for ALL three branches
    (pinned by tests/unit/test_graph_workflows.py).

    WHY the fan-out: the three discovery branches make independent
    google_search queries with no cross-dependencies — running them as
    parallel graph branches is a 3x latency win at identical token cost
    (ADR #3).

    Args:
        model: The LLM model to use
        book_title: Exact book title (pre-confirmed by caller)
        author: Exact author name (pre-confirmed by caller)
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        Workflow graph for book→place discovery
    """
    if prompts is None:
        prompts = load_prompts()

    book_context_researcher, book_context_formatter = create_book_context_agents(
        model, google_search, book_title=book_title, author=author, prompts=prompts
    )
    city_researcher, city_formatter = create_city_agents(
        model,
        google_search,
        book_title=book_title,
        author=author,
        vibe=vibe,
        taste_context=taste_context,
        prompts=prompts,
    )
    landmark_researcher, landmark_formatter = create_landmark_agents(
        model,
        google_search,
        book_title=book_title,
        author=author,
        vibe=vibe,
        taste_context=taste_context,
        prompts=prompts,
    )
    author_researcher, author_formatter = create_author_agents(
        model,
        google_search,
        book_title=book_title,
        author=author,
        vibe=vibe,
        taste_context=taste_context,
        prompts=prompts,
    )
    region_analyzer = create_region_analyzer_agent(model, prompts=prompts)
    join = JoinNode(name="discovery_join")

    return Workflow(
        name="book_to_place_discovery",
        edges=[
            (
                START,
                book_context_researcher,
                book_context_formatter,
                (city_researcher, landmark_researcher, author_researcher),
            ),
            (city_researcher, city_formatter, join),
            (landmark_researcher, landmark_formatter, join),
            (author_researcher, author_formatter, join),
            (join, region_analyzer),
        ],
    )


def create_book_to_place_composition_workflow(
    model, prompts: AgentPrompts | None = None
):
    """
    Create the book→place composition workflow (phase 2 of the primary flow).

    Runs after the user has selected region(s); expects the selection in
    session state (written by the executor before the run).

    Graph:
        START → trip_composer → state["composer_envelope"]

    Args:
        model: The LLM model to use
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        Workflow graph for book→place composition
    """
    if prompts is None:
        prompts = load_prompts()

    trip_composer = create_trip_composer_agent(model, prompts=prompts)

    return Workflow(
        name="book_to_place_composition",
        edges=[(START, trip_composer)],
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

    Graph:
        START → book_context_researcher → book_context_formatter
              → local_atmosphere_researcher → local_atmosphere_formatter

    There is no city/landmark/author/region discovery: those agents look at
    the book's geography, which is irrelevant here (ADR #13).

    Args:
        model: The LLM model to use.
        book_title: Exact book title.
        author: Exact author name.
        location_label: Human-readable user location (e.g. "New York, NY 10013").
        radius_km: Maximum distance in km from the user's location.
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        Workflow graph for the local-atmosphere flow.
    """
    if prompts is None:
        prompts = load_prompts()

    book_context_researcher, book_context_formatter = create_book_context_agents(
        model, google_search, book_title=book_title, author=author, prompts=prompts
    )
    local_researcher, local_formatter = create_local_atmosphere_agents(
        model,
        google_search,
        location_label=location_label,
        radius_km=radius_km,
        prompts=prompts,
    )

    return Workflow(
        name="local_atmosphere_workflow",
        edges=[
            (
                START,
                book_context_researcher,
                book_context_formatter,
                local_researcher,
                local_formatter,
            )
        ],
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

    Called after composition when the user clicks a suggestion chip.

    Graph:
        START → expansion_researcher [google_search]
              → expansion_formatter [output_schema=ExpansionResult]
              → state["last_expansion"]

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
        Workflow graph for the expansion flow.
    """
    if prompts is None:
        prompts = load_prompts()

    researcher, formatter = create_expansion_agents(
        model,
        google_search_tool,
        book_title=book_title,
        author=author,
        parent_city=parent_city,
        action_prompt=action_prompt,
        existing_places=existing_places,
        prompts=prompts,
    )

    return Workflow(
        name="expansion_workflow",
        edges=[(START, researcher, formatter)],
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

    Graph:
        START → book_recommendation_researcher [google_search]
              → book_recommendation_formatter [output_schema=BookRecommendationsResult]
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
        Workflow graph for the book-recommendation flow.
    """
    if prompts is None:
        prompts = load_prompts()

    researcher, formatter = create_book_recommendation_agents(
        model,
        google_search_tool,
        book_title=book_title,
        author=author,
        destinations=destinations,
        themes=themes,
        prompts=prompts,
    )

    return Workflow(
        name="book_recommendation_workflow",
        edges=[(START, researcher, formatter)],
    )


def create_place_to_book_workflow(
    model,
    google_search_tool,
    place: str,
    prompts: AgentPrompts | None = None,
):
    """
    Create the place→book reverse-routing workflow (AI candidate layer).

    The reverse of the primary book→place flow: a destination is the input
    and grounded, literal/vibe-labelled book candidates are the output.
    Exposed as ``POST /place-to-book`` (gateway secret enforced); the
    storyland-services gateway runs the authoritative Google Books existence
    check downstream.

    Graph:
        START → place_to_book_researcher [google_search]
              → place_to_book_formatter [output_schema=PlaceToBookCandidates]
              → state["last_place_to_book"]

    Args:
        model: The LLM model to use.
        google_search_tool: The Google Search tool.
        place: Free-text destination.
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        Workflow graph for the place→book flow.
    """
    if prompts is None:
        prompts = load_prompts()

    researcher, formatter = create_place_to_book_agents(
        model, google_search_tool, place=place, prompts=prompts
    )

    return Workflow(
        name="place_to_book_workflow",
        edges=[(START, researcher, formatter)],
    )
