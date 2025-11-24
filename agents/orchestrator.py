"""
Workflow orchestrator.

Creates the main sequential workflow that coordinates all agents to produce
a complete literary travel itinerary.

Two-phase architecture:
1. Metadata stage - fetches book metadata (title, author)
2. Main workflow - uses exact title/author for accurate searches
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


def create_metadata_stage(model, google_books_tool):
    """
    Create the metadata extraction stage.

    This stage runs first to get the exact book title and author,
    which are then used to create the main workflow.

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


def create_main_workflow(model, book_title: str, author: str):
    """
    Create the main workflow with exact book title and author.

    This stage runs after metadata is extracted, using the exact
    title/author for accurate context and discovery searches.

    Architecture:
        SequentialAgent (main_workflow)
        ├─ book_context_pipeline [research → format] → state["book_context"]
        ├─ reader_profile_agent [read preferences] → state["reader_profile"]
        ├─ ParallelAgent (parallel_discovery) ⚡ CONCURRENT
        │  ├─ city_pipeline [research → format] → state["city_discovery"]
        │  ├─ landmark_pipeline [research → format] → state["landmark_discovery"]
        │  └─ author_pipeline [research → format] → state["author_sites"]
        └─ trip_composer_agent → state["final_itinerary"]

    Args:
        model: The LLM model to use
        book_title: Exact book title from metadata stage
        author: Exact author name from metadata stage

    Returns:
        SequentialAgent orchestrating the main workflow
    """
    # Create pipelines with exact book info
    book_context_pipeline = create_book_context_pipeline(
        model, google_search, book_title=book_title, author=author
    )

    city_pipeline = create_city_pipeline(model, google_search)
    landmark_pipeline = create_landmark_pipeline(model, google_search)
    author_pipeline = create_author_pipeline(model, google_search)

    trip_composer = create_trip_composer_agent(model)
    reader_profile = create_reader_profile_agent(model)

    # Create parallel discovery agent
    parallel_discovery = ParallelAgent(
        name="parallel_discovery",
        sub_agents=[city_pipeline, landmark_pipeline, author_pipeline],
    )

    # Build workflow
    sub_agents = [
        book_context_pipeline,
        reader_profile,
        parallel_discovery,
        trip_composer,
    ]

    return SequentialAgent(
        name="main_workflow",
        sub_agents=sub_agents,
    )


# Legacy function for backwards compatibility (ADK web UI, tests)
def create_workflow(model, google_books_tool):
    """
    Create the complete workflow (legacy single-phase version).

    NOTE: This version uses conversation history for book context.
    For better accuracy, use create_metadata_stage() + create_main_workflow().

    Architecture:
        SequentialAgent (workflow)
        ├─ book_metadata_pipeline [fetch → format] → state["book_metadata"]
        ├─ book_context_pipeline [research → format] → state["book_context"]
        ├─ reader_profile_agent [read preferences] → state["reader_profile"]
        ├─ ParallelAgent (parallel_discovery) ⚡ CONCURRENT
        │  ├─ city_pipeline [research → format] → state["city_discovery"]
        │  ├─ landmark_pipeline [research → format] → state["landmark_discovery"]
        │  └─ author_pipeline [research → format] → state["author_sites"]
        └─ trip_composer_agent → state["final_itinerary"] (uses preferences!)

    Args:
        model: The LLM model to use for all agents
        google_books_tool: The Google Books FunctionTool

    Returns:
        SequentialAgent orchestrating the complete workflow
    """
    # For legacy mode, use placeholder values - agent will use conversation history
    book_metadata_pipeline = create_book_metadata_pipeline(model, google_books_tool)
    book_context_pipeline = create_book_context_pipeline(
        model, google_search, book_title="[from conversation]", author="[from conversation]"
    )

    city_pipeline = create_city_pipeline(model, google_search)
    landmark_pipeline = create_landmark_pipeline(model, google_search)
    author_pipeline = create_author_pipeline(model, google_search)

    trip_composer = create_trip_composer_agent(model)
    reader_profile = create_reader_profile_agent(model)

    # Create parallel discovery agent
    parallel_discovery = ParallelAgent(
        name="parallel_discovery",
        sub_agents=[city_pipeline, landmark_pipeline, author_pipeline],
    )

    # Build workflow with reader profile for personalization
    sub_agents = [
        book_metadata_pipeline,
        book_context_pipeline,
        reader_profile,
        parallel_discovery,
        trip_composer,
    ]

    # Create the main workflow
    workflow = SequentialAgent(
        name="workflow",
        sub_agents=sub_agents,
    )

    return workflow
