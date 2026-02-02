"""
Langfuse observability plugin for Google ADK.

Tracks token usage, costs, and agent performance metrics in Langfuse.
"""

import asyncio
from typing import Any, Optional
from dataclasses import dataclass

from google.adk.plugins import Plugin
from google.adk.runners.events import (
    Event,
    AgentStartEvent,
    AgentCompleteEvent,
    ModelRequestEvent,
    ModelResponseEvent,
    ToolStartEvent,
    ToolCompleteEvent,
)

try:
    from langfuse import Langfuse
    from langfuse.decorators import langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


@dataclass
class TokenUsage:
    """Token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        """
        Calculate cost in USD based on Gemini 2.0 Flash pricing.

        Pricing (as of Jan 2026):
        - Input: $0.075 per 1M tokens
        - Output: $0.30 per 1M tokens
        """
        input_cost = (self.input_tokens / 1_000_000) * 0.075
        output_cost = (self.output_tokens / 1_000_000) * 0.30
        return input_cost + output_cost


class LangfusePlugin(Plugin):
    """
    Langfuse observability plugin for tracking token usage and costs.

    Integrates with Google ADK's event system to capture:
    - Token usage per model call (input/output)
    - Cost calculation based on Gemini pricing
    - Agent execution traces
    - Tool usage metrics

    Usage:
        config = load_config()
        langfuse_plugin = LangfusePlugin(
            secret_key=config.langfuse_secret_key,
            public_key=config.langfuse_public_key,
            host=config.langfuse_host,
        )

        runner = Runner(
            agent=my_agent,
            plugins=[LoggingPlugin(), langfuse_plugin],
        )
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        public_key: Optional[str] = None,
        host: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize Langfuse plugin.

        Args:
            secret_key: Langfuse secret key (from env LANGFUSE_SECRET_KEY)
            public_key: Langfuse public key (from env LANGFUSE_PUBLIC_KEY)
            host: Langfuse host URL (from env LANGFUSE_HOST)
            enabled: Enable/disable plugin (auto-disabled if credentials missing)
        """
        self.enabled = enabled and LANGFUSE_AVAILABLE
        self.client: Optional[Langfuse] = None
        self._token_usage = TokenUsage()
        self._current_trace = None
        self._current_generation = None
        self._agent_stack = []  # Track nested agent calls

        if not LANGFUSE_AVAILABLE:
            return

        # Only initialize if all credentials provided
        if secret_key and public_key:
            try:
                self.client = Langfuse(
                    secret_key=secret_key,
                    public_key=public_key,
                    host=host,
                )
                self.enabled = True
            except Exception as e:
                print(f"⚠️  Langfuse initialization failed: {e}")
                self.enabled = False
        else:
            self.enabled = False

    async def on_event(self, event: Event) -> None:
        """
        Handle ADK events and track in Langfuse.

        Args:
            event: ADK event (agent start/complete, model request/response, etc.)
        """
        if not self.enabled or not self.client:
            return

        try:
            # Handle different event types
            if isinstance(event, AgentStartEvent):
                await self._handle_agent_start(event)
            elif isinstance(event, AgentCompleteEvent):
                await self._handle_agent_complete(event)
            elif isinstance(event, ModelRequestEvent):
                await self._handle_model_request(event)
            elif isinstance(event, ModelResponseEvent):
                await self._handle_model_response(event)
            elif isinstance(event, ToolStartEvent):
                await self._handle_tool_start(event)
            elif isinstance(event, ToolCompleteEvent):
                await self._handle_tool_complete(event)
        except Exception as e:
            # Don't break the workflow if observability fails
            # Log error with context for debugging
            print(f"⚠️  Langfuse event handling error: {e}")

            # Try to log the error to Langfuse if possible
            try:
                if self._current_trace:
                    self._current_trace.update(
                        metadata={
                            "langfuse_plugin_error": str(e),
                            "error_type": type(e).__name__,
                            "event_type": type(event).__name__ if event else "unknown",
                        }
                    )
            except Exception:
                # Silently fail if we can't even log the error
                pass

    async def _handle_agent_start(self, event: AgentStartEvent) -> None:
        """Track agent execution start."""
        agent_name = getattr(event, 'agent_name', 'unknown_agent')

        # Create trace for top-level agent, span for nested agents
        if not self._agent_stack:
            self._current_trace = self.client.trace(
                name=agent_name,
                metadata={
                    "invocation_id": getattr(event, 'invocation_id', None),
                    "agent_type": "google_adk",
                },
            )
        else:
            # Nested agent - create span
            if self._current_trace:
                self.client.span(
                    trace_id=self._current_trace.id,
                    name=agent_name,
                    metadata={"invocation_id": getattr(event, 'invocation_id', None)},
                )

        self._agent_stack.append(agent_name)

    async def _handle_agent_complete(self, event: AgentCompleteEvent) -> None:
        """Track agent execution completion."""
        if self._agent_stack:
            self._agent_stack.pop()

        # Finalize trace when top-level agent completes
        if not self._agent_stack and self._current_trace:
            self._current_trace.update(
                output={"status": "completed"},
                metadata={
                    "total_input_tokens": self._token_usage.input_tokens,
                    "total_output_tokens": self._token_usage.output_tokens,
                    "total_cost_usd": self._token_usage.cost_usd,
                },
            )
            self._current_trace = None

    async def _handle_model_request(self, event: ModelRequestEvent) -> None:
        """Track model request (LLM call start)."""
        if not self._current_trace:
            return

        # Extract prompt from event
        prompt = getattr(event, 'request', None)
        model_name = getattr(event, 'model', 'gemini-2.0-flash-lite')

        # Create generation tracking
        self._current_generation = self.client.generation(
            trace_id=self._current_trace.id,
            name=f"{model_name}_call",
            model=model_name,
            input=prompt,
            metadata={
                "agent": self._agent_stack[-1] if self._agent_stack else "unknown",
            },
        )

    async def _handle_model_response(self, event: ModelResponseEvent) -> None:
        """
        Track model response and extract token usage.

        This is where we capture actual token counts from Gemini API.
        """
        if not self._current_generation:
            return

        # Extract response content
        response = getattr(event, 'response', None)

        # Extract token usage from Gemini response
        usage = self._extract_token_usage(response)

        if usage:
            # Update running totals
            self._token_usage.input_tokens += usage.input_tokens
            self._token_usage.output_tokens += usage.output_tokens
            self._token_usage.total_tokens += usage.total_tokens

            # Update generation with token usage and cost
            self._current_generation.update(
                output=response,
                usage={
                    "input": usage.input_tokens,
                    "output": usage.output_tokens,
                    "total": usage.total_tokens,
                    "unit": "TOKENS",
                },
                metadata={
                    "cost_usd": usage.cost_usd,
                    "model_pricing": {
                        "input_per_1m": 0.075,
                        "output_per_1m": 0.30,
                    },
                },
            )

        self._current_generation = None

    def _extract_token_usage(self, response: Any) -> Optional[TokenUsage]:
        """
        Extract token counts from Gemini API response.

        Gemini responses include usage_metadata with:
        - prompt_token_count (input tokens)
        - candidates_token_count (output tokens)
        - total_token_count

        Args:
            response: Gemini API response object

        Returns:
            TokenUsage object or None if not available
        """
        try:
            # Try to extract from usage_metadata attribute
            if hasattr(response, 'usage_metadata'):
                metadata = response.usage_metadata
                return TokenUsage(
                    input_tokens=getattr(metadata, 'prompt_token_count', 0),
                    output_tokens=getattr(metadata, 'candidates_token_count', 0),
                    total_tokens=getattr(metadata, 'total_token_count', 0),
                )

            # Alternative: Try dict-like access
            if isinstance(response, dict) and 'usage_metadata' in response:
                metadata = response['usage_metadata']
                return TokenUsage(
                    input_tokens=metadata.get('prompt_token_count', 0),
                    output_tokens=metadata.get('candidates_token_count', 0),
                    total_tokens=metadata.get('total_token_count', 0),
                )

            return None
        except Exception:
            return None

    async def _handle_tool_start(self, event: ToolStartEvent) -> None:
        """Track tool execution start."""
        if not self._current_trace:
            return

        tool_name = getattr(event, 'tool_name', 'unknown_tool')
        self.client.span(
            trace_id=self._current_trace.id,
            name=f"tool_{tool_name}",
            metadata={
                "tool_type": "adk_tool",
                "tool_name": tool_name,
            },
        )

    async def _handle_tool_complete(self, event: ToolCompleteEvent) -> None:
        """Track tool execution completion."""
        # Tool completion is tracked via span end
        pass

    def get_session_stats(self) -> dict[str, Any]:
        """
        Get current session token usage statistics.

        Returns:
            Dictionary with token counts and cost
        """
        return {
            "input_tokens": self._token_usage.input_tokens,
            "output_tokens": self._token_usage.output_tokens,
            "total_tokens": self._token_usage.total_tokens,
            "cost_usd": self._token_usage.cost_usd,
        }

    def reset_stats(self) -> None:
        """Reset token usage statistics."""
        self._token_usage = TokenUsage()

    async def flush(self) -> None:
        """Flush pending Langfuse events."""
        if self.client:
            try:
                await asyncio.to_thread(self.client.flush)
            except Exception as e:
                print(f"⚠️  Langfuse flush error: {e}")
