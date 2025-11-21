"""
Workflow orchestrator.

Creates the main sequential workflow that coordinates all agents to produce
a complete literary travel itinerary.
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


def create_workflow(model, google_books_tool):
    """
    Create the complete workflow for generating literary travel itineraries.

    Architecture:
        SequentialAgent (workflow)
        ├─ book_metadata_pipeline [fetch → format] → state["book_metadata"]
        ├─ book_context_pipeline [research → format] → state["book_context"]
        ├─ ParallelAgent (parallel_discovery) ⚡ CONCURRENT
        │  ├─ city_pipeline [research → format] → state["city_discovery"]
        │  ├─ landmark_pipeline [research → format] → state["landmark_discovery"]
        │  └─ author_pipeline [research → format] → state["author_sites"]
        └─ trip_composer_agent → state["final_itinerary"]

    Args:
        model: The LLM model to use for all agents
        google_books_tool: The Google Books FunctionTool

    Returns:
        SequentialAgent orchestrating the complete workflow
    """
    # Create all agent pipelines
    book_metadata_pipeline = create_book_metadata_pipeline(model, google_books_tool)
    book_context_pipeline = create_book_context_pipeline(model, google_search)

    city_pipeline = create_city_pipeline(model, google_search)
    landmark_pipeline = create_landmark_pipeline(model, google_search)
    author_pipeline = create_author_pipeline(model, google_search)

    trip_composer = create_trip_composer_agent(model)

    # Create parallel discovery agent
    parallel_discovery = ParallelAgent(
        name="parallel_discovery",
        sub_agents=[city_pipeline, landmark_pipeline, author_pipeline],
    )

    # Create the main workflow
    workflow = SequentialAgent(
        name="workflow",
        sub_agents=[
            book_metadata_pipeline,
            book_context_pipeline,
            parallel_discovery,
            trip_composer,
        ],
    )

    return workflow
