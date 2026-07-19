"""
Agent definitions for StoryLand AI.

This package contains agent factory functions for:
- Book context research
- Discovery agents (cities, landmarks, author sites)
- Trip composition
- Workflow orchestration
"""

from .book_context_agent import create_book_context_agents
from .discovery_agents import (
    create_city_agents,
    create_landmark_agents,
    create_author_agents,
)
from .expansion_agent import create_expansion_agents
from .local_atmosphere_agent import create_local_atmosphere_agents
from .trip_composer_agent import create_trip_composer_agent
from .region_analyzer_agent import create_region_analyzer_agent
from .book_recommendation_agent import create_book_recommendation_agents
from .place_to_book_agent import create_place_to_book_agents
from .orchestrator import (
    create_book_recommendation_workflow,
    create_expansion_workflow,
    create_place_to_book_workflow,
    create_book_to_place_discovery_workflow,
    create_book_to_place_composition_workflow,
    create_local_atmosphere_workflow,
)

__all__ = [
    "create_book_context_agents",
    "create_city_agents",
    "create_landmark_agents",
    "create_author_agents",
    "create_expansion_agents",
    "create_local_atmosphere_agents",
    "create_trip_composer_agent",
    "create_region_analyzer_agent",
    "create_book_recommendation_agents",
    "create_place_to_book_agents",
    "create_book_recommendation_workflow",
    "create_expansion_workflow",
    "create_place_to_book_workflow",
    "create_book_to_place_discovery_workflow",
    "create_book_to_place_composition_workflow",
    "create_local_atmosphere_workflow",
]
