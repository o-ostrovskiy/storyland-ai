"""
Agent definitions for StoryLand AI.

This package contains agent factory functions for:
- Book metadata extraction
- Book context research
- Discovery agents (cities, landmarks, author sites)
- Trip composition
- Reader profile and personalization
- Workflow orchestration

For ADK Web UI:
    Run `adk web agents/` or `python main.py --dev`
"""

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
from .orchestrator import (
    create_metadata_stage,
    create_discovery_workflow,
    create_composition_workflow,
    create_eval_workflow,
)

__all__ = [
    "create_book_metadata_pipeline",
    "create_book_context_pipeline",
    "create_city_pipeline",
    "create_landmark_pipeline",
    "create_author_pipeline",
    "create_trip_composer_agent",
    "create_reader_profile_agent",
    "create_region_analyzer_agent",
    "create_metadata_stage",
    "create_discovery_workflow",
    "create_composition_workflow",
    "create_eval_workflow",
]
