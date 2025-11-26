"""
Workflow orchestrator.

Creates workflows that coordinate all agents to produce complete literary
travel itineraries.

Three-phase architecture (CLI with HITL):
1. Metadata stage - fetches book metadata (title, author)
2. Discovery workflow - finds locations and groups into travel regions
3. Composition workflow - creates itinerary for selected region(s)

WHY THREE PHASES WITH HITL?
- Problem: Books like "Gone with the Wind" span Georgia (USA) and have author
  sites in Atlanta. Auto-generating itineraries for ALL regions would create
  impractical multi-continent trips.
- Solution: After discovery, show user region options (e.g., "England", "Scotland")
  and let them choose which region(s) to explore. This prevents wasting tokens
  on unwanted regions and gives users control over trip scope.
- Trade-off: Requires human input (not fully autonomous) but produces much more
  practical and personalized itineraries.

Eval workflow (automated):
- Single workflow with region analysis but auto-selects all regions
- Used for ADK evals where HITL interaction isn't possible
"""

from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.tools import google_search

from .book_metadata_agent import create_book_metadata_pipeline
from .book_context_agent import create_book_context_pipeline
from .discovery_agents import (
    create_city_pipeline,
    create_landmark_pipeline,
    create_author_pipeline,
)
from .trip_composer_agent import create_trip_composer_agent
from .reader_profile_agent import create_reader_profile_agent
from .region_analyzer_agent import create_region_analyzer_agent


def create_metadata_stage(model, google_books_tool):
    """
    Create the metadata extraction stage.

    This stage runs first to get the exact book title and author,
    which are then used to create the main workflow.

    WHY SEPARATE METADATA STAGE?
    - Problem: Books like "The Nightingale" have multiple authors (Kristin Hannah
      vs. Hans Christian Andersen). User input may be ambiguous.
    - Solution: Use Google Books API early to disambiguate and get exact title/author
      before running expensive discovery workflows.
    - Benefit: Prevents wrong book discovery (e.g., searching for fairy tale locations
      instead of WWII France).

    Architecture:
        SequentialAgent (metadata_stage)
        └─ book_metadata_pipeline [fetch → format] → state["book_metadata"]

    Args:
        model: The LLM model to use
        google_books_tool: The Google Books FunctionTool

    Returns:
        SequentialAgent that extracts book metadata
    """
    book_metadata_pipeline = create_book_metadata_pipeline(model, google_books_tool)

    return SequentialAgent(
        name="metadata_stage",
        sub_agents=[book_metadata_pipeline],
    )


def create_discovery_workflow(model, book_title: str, author: str):
    """
    Create the discovery workflow that finds locations and analyzes regions.

    This workflow runs after metadata extraction and before user region selection.
    It discovers cities, landmarks, and author sites, then groups them into
    practical travel regions for the user to choose from.

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
        book_title: Exact book title from metadata stage
        author: Exact author name from metadata stage

    Returns:
        SequentialAgent orchestrating the discovery workflow
    """
    # Create pipelines with exact book info
    book_context_pipeline = create_book_context_pipeline(
        model, google_search, book_title=book_title, author=author
    )

    city_pipeline = create_city_pipeline(model, google_search)
    landmark_pipeline = create_landmark_pipeline(model, google_search)
    author_pipeline = create_author_pipeline(model, google_search)

    reader_profile = create_reader_profile_agent(model)
    region_analyzer = create_region_analyzer_agent(model)

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


def create_composition_workflow(model):
    """
    Create the composition workflow that generates the final itinerary.

    This workflow runs after the user has selected a region.
    It expects the selected region to be stored in session state as "selected_region".

    Architecture:
        SequentialAgent (composition_workflow)
        └─ trip_composer_agent → state["final_itinerary"]

    Args:
        model: The LLM model to use

    Returns:
        SequentialAgent orchestrating the composition workflow
    """
    trip_composer = create_trip_composer_agent(model)

    return SequentialAgent(
        name="composition_workflow",
        sub_agents=[trip_composer],
    )


def create_eval_workflow(model, google_books_tool):
    """
    Create evaluation workflow with automated region selection.

    This workflow is designed for ADK evals where human-in-the-loop
    interaction is not possible. It includes region analysis but the
    trip composer will receive all discovered regions automatically.

    Architecture:
        SequentialAgent (eval_workflow)
        ├─ book_metadata_pipeline [fetch → format] → state["book_metadata"]
        ├─ book_context_pipeline [research → format] → state["book_context"]
        ├─ reader_profile_agent [read preferences] → state["reader_profile"]
        ├─ ParallelAgent (parallel_discovery) ⚡ CONCURRENT
        │  ├─ city_pipeline [research → format] → state["city_discovery"]
        │  ├─ landmark_pipeline [research → format] → state["landmark_discovery"]
        │  └─ author_pipeline [research → format] → state["author_sites"]
        ├─ region_analyzer_agent → state["region_analysis"]
        └─ trip_composer_agent → state["final_itinerary"]

    Args:
        model: The LLM model to use for all agents
        google_books_tool: The Google Books FunctionTool

    Returns:
        SequentialAgent orchestrating the complete eval workflow
    """
    # Create all pipelines
    book_metadata_pipeline = create_book_metadata_pipeline(model, google_books_tool)
    book_context_pipeline = create_book_context_pipeline(
        model, google_search, book_title="[from conversation]", author="[from conversation]"
    )

    city_pipeline = create_city_pipeline(model, google_search)
    landmark_pipeline = create_landmark_pipeline(model, google_search)
    author_pipeline = create_author_pipeline(model, google_search)

    reader_profile = create_reader_profile_agent(model)
    region_analyzer = create_region_analyzer_agent(model)
    trip_composer = create_trip_composer_agent(model)

    # Create parallel discovery agent
    parallel_discovery = ParallelAgent(
        name="parallel_discovery",
        sub_agents=[city_pipeline, landmark_pipeline, author_pipeline],
    )

    # Build complete eval workflow with region analysis
    sub_agents = [
        book_metadata_pipeline,
        book_context_pipeline,
        reader_profile,
        parallel_discovery,
        region_analyzer,
        trip_composer,
    ]

    return SequentialAgent(
        name="eval_workflow",
        sub_agents=sub_agents,
    )
