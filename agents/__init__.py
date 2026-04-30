"""
Agent definitions for StoryLand AI.

This package contains agent factory functions for:
- Book context research
- Discovery agents (cities, landmarks, author sites)
- Trip composition
- Reader profile and personalization
- Workflow orchestration
"""

from .book_context_agent import create_book_context_pipeline
from .discovery_agents import (
    create_city_pipeline,
    create_landmark_pipeline,
    create_author_pipeline,
)
from .local_atmosphere_agent import create_local_atmosphere_pipeline
from .trip_composer_agent import create_trip_composer_agent
from .reader_profile_agent import create_reader_profile_agent
from .region_analyzer_agent import create_region_analyzer_agent
from .orchestrator import (
    create_discovery_workflow,
    create_composition_workflow,
    create_local_atmosphere_workflow,
)

__all__ = [
    "create_book_context_pipeline",
    "create_city_pipeline",
    "create_landmark_pipeline",
    "create_author_pipeline",
    "create_local_atmosphere_pipeline",
    "create_trip_composer_agent",
    "create_reader_profile_agent",
    "create_region_analyzer_agent",
    "create_discovery_workflow",
    "create_composition_workflow",
    "create_local_atmosphere_workflow",
]
