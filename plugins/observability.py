"""
Observability Plugin for StoryLand AI.

Provides simple logging for agent and tool activity.
"""

import time
from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin

from common.logging import get_logger


class ObservabilityPlugin(BasePlugin):
    """
    Simple observability plugin - logs agent and tool activity.

    Example:
        >>> runner = Runner(agent=workflow, plugins=[ObservabilityPlugin()])
    """

    def __init__(self) -> None:
        super().__init__(name="observability")
        self.logger = get_logger("observability")
        self._agent_times: dict[str, float] = {}
        self._tool_times: dict[str, float] = {}

    async def before_agent_callback(self, **kwargs) -> None:
        agent = kwargs.get("agent")
        name = getattr(agent, "name", "unknown") if agent else "unknown"
        self._agent_times[name] = time.time()
        self.logger.info("agent_started", agent=name)

    async def after_agent_callback(self, **kwargs) -> Optional[Any]:
        agent = kwargs.get("agent")
        name = getattr(agent, "name", "unknown") if agent else "unknown"
        duration = (time.time() - self._agent_times.pop(name, time.time())) * 1000
        self.logger.info("agent_completed", agent=name, duration_ms=round(duration))
        return None

    async def before_tool_callback(self, **kwargs) -> Optional[Any]:
        tool = kwargs.get("tool")
        name = getattr(tool, "name", str(tool)) if tool else "unknown"
        self._tool_times[name] = time.time()
        self.logger.info("tool_started", tool=name)
        return None

    async def after_tool_callback(self, **kwargs) -> Optional[Any]:
        tool = kwargs.get("tool")
        name = getattr(tool, "name", str(tool)) if tool else "unknown"
        duration = (time.time() - self._tool_times.pop(name, time.time())) * 1000
        self.logger.info("tool_completed", tool=name, duration_ms=round(duration))
        return None

    async def on_tool_error_callback(self, **kwargs) -> Optional[Any]:
        tool = kwargs.get("tool")
        error = kwargs.get("error")
        name = getattr(tool, "name", str(tool)) if tool else "unknown"
        self.logger.error("tool_error", tool=name, error=str(error))
        return None

    async def on_model_error_callback(self, **kwargs) -> Optional[Any]:
        error = kwargs.get("error")
        self.logger.error("model_error", error=str(error))
        return None
