"""
StoryLand AI Core — Python SDK for literary travel itinerary generation.

Usage:
    from core import WorkflowExecutor, ExecutorConfig
    from core.events import MetadataReady, RegionsReady, ItineraryReady

    config = ExecutorConfig(model_name="gemini-2.0-flash", google_api_key="...")
    executor = WorkflowExecutor(config)

    async for event in executor.discover(book_title="1984"):
        ...
"""

from .executor import WorkflowExecutor
from .types import ExecutorConfig
from .events import (
    DomainEvent,
    Phase,
    ProgressEvent,
    MetadataReady,
    RegionsReady,
    ItineraryReady,
    WorkflowError,
    WorkflowComplete,
)

__all__ = [
    "WorkflowExecutor",
    "ExecutorConfig",
    "DomainEvent",
    "Phase",
    "ProgressEvent",
    "MetadataReady",
    "RegionsReady",
    "ItineraryReady",
    "WorkflowError",
    "WorkflowComplete",
]
