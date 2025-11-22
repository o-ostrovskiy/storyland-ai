"""
Plugins for StoryLand AI.

This package contains ADK plugins for extending agent functionality:
- ObservabilityPlugin: Production logging and tracing
"""

from .observability import ObservabilityPlugin

__all__ = ["ObservabilityPlugin"]
