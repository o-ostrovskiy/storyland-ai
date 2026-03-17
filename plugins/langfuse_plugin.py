"""
Langfuse observability plugin for Google ADK.

Tracks token usage, costs, and agent performance metrics in Langfuse.
"""

import asyncio
from typing import Any, Optional
from dataclasses import dataclass

from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.genai import types
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from common.logging import get_logger

logger = get_logger(__name__)

try:
    from langfuse import Langfuse
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


class LangfusePlugin(BasePlugin):
    """
    Langfuse observability plugin for tracking token usage and costs.

    Integrates with Google ADK's callback system to capture:
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
        super().__init__(name="langfuse_plugin")
        self.enabled = enabled and LANGFUSE_AVAILABLE
        self.client: Optional[Langfuse] = None
        self._token_usage = TokenUsage()
        self._current_trace = None
        self._current_generation = None
        self._agent_stack = []  # Track nested agent calls
        self._current_span = None

        if not LANGFUSE_AVAILABLE:
            logger.info("langfuse_unavailable", reason="package_not_installed")
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
                logger.info("langfuse_plugin_initialized", host=host)
            except Exception as e:
                logger.warning("langfuse_init_failed", error=str(e), error_type=type(e).__name__)
                self.enabled = False
        else:
            logger.info("langfuse_disabled", reason="missing_credentials")
            self.enabled = False

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        """Log user message and create trace."""
        if not self.enabled or not self.client:
            return None

        try:
            # Create a top-level span (auto-creates a trace in Langfuse v3)
            agent_name = getattr(invocation_context.agent, 'name', 'unknown_agent')
            user_id = invocation_context.user_id
            self._current_trace = self.client.start_span(
                name=f"{agent_name}_invocation",
                metadata={
                    "invocation_id": invocation_context.invocation_id,
                    "session_id": invocation_context.session.id if invocation_context.session else None,
                    "agent_type": "google_adk",
                },
                input=str(user_message),
            )
            # Set user_id on the trace so it appears as the Langfuse User field
            self._current_trace.update_trace(user_id=user_id)
            logger.debug("langfuse_trace_created", trace_id=self._current_trace.trace_id, user_id=user_id)
        except Exception as e:
            logger.warning("langfuse_trace_error", error=str(e), error_type=type(e).__name__)

        return None

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        """Track agent execution start."""
        if not self.enabled or not self.client or not self._current_trace:
            return None

        try:
            agent_name = getattr(agent, 'name', 'unknown_agent')

            # Create child span for agent under the current trace span
            span = self._current_trace.start_span(
                name=agent_name,
                metadata={
                    "agent_type": type(agent).__name__,
                    "callback_context": str(callback_context),
                },
            )
            self._agent_stack.append((agent_name, span))
            logger.debug("langfuse_agent_start", agent=agent_name)
        except Exception as e:
            logger.warning("langfuse_agent_start_error", error=str(e), error_type=type(e).__name__)

        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        """Track agent execution completion."""
        if not self.enabled or not self.client:
            return None

        try:
            if self._agent_stack:
                agent_name, span = self._agent_stack.pop()
                span.end()
                logger.debug("langfuse_agent_complete", agent=agent_name)
        except Exception as e:
            logger.warning("langfuse_agent_complete_error", error=str(e), error_type=type(e).__name__)

        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        """Track model request (LLM call start)."""
        if not self.enabled or not self.client or not self._current_trace:
            return None

        try:
            model_name = getattr(llm_request, 'model', 'unknown_model')

            # Create generation tracking under the current agent span or trace span
            parent = self._agent_stack[-1][1] if self._agent_stack else self._current_trace
            self._current_generation = parent.start_observation(
                as_type="generation",
                name=f"{model_name}_call",
                model=model_name,
                input=str(llm_request),
                metadata={
                    "agent": self._agent_stack[-1][0] if self._agent_stack else "unknown",
                },
            )
            logger.debug("langfuse_model_request", model=model_name)
        except Exception as e:
            logger.warning("langfuse_model_request_error", error=str(e), error_type=type(e).__name__)

        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        """
        Track model response and extract token usage.

        This is where we capture actual token counts from Gemini API.
        """
        if not self.enabled or not self.client or not self._current_generation:
            return None

        try:
            # Extract token usage from response
            usage = self._extract_token_usage(llm_response)

            if usage:
                # Update running totals
                self._token_usage.input_tokens += usage.input_tokens
                self._token_usage.output_tokens += usage.output_tokens
                self._token_usage.total_tokens += usage.total_tokens

                # Update generation with token usage and cost
                self._current_generation.update(
                    output=str(llm_response),
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
                logger.debug(
                    "langfuse_model_response",
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                )

            self._current_generation.end()
            self._current_generation = None
        except Exception as e:
            logger.warning("langfuse_model_response_error", error=str(e), error_type=type(e).__name__)

        return None

    def _extract_token_usage(self, response: LlmResponse) -> Optional[TokenUsage]:
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
        except Exception as e:
            logger.debug("token_usage_extraction_failed", error=str(e))
            return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        """Track tool execution start."""
        if not self.enabled or not self.client or not self._current_trace:
            return None

        try:
            tool_name = getattr(tool, 'name', 'unknown_tool')
            parent = self._agent_stack[-1][1] if self._agent_stack else self._current_trace
            self._current_span = parent.start_span(
                name=f"tool_{tool_name}",
                metadata={
                    "tool_type": "adk_tool",
                    "tool_name": tool_name,
                    "tool_args": str(tool_args),
                },
            )
            logger.debug("langfuse_tool_start", tool=tool_name)
        except Exception as e:
            logger.warning("langfuse_tool_start_error", error=str(e), error_type=type(e).__name__)

        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        """Track tool execution completion."""
        if not self.enabled or not self.client or not self._current_span:
            return None

        try:
            tool_name = getattr(tool, 'name', 'unknown_tool')
            self._current_span.update(output=str(result))
            self._current_span.end()
            self._current_span = None
            logger.debug("langfuse_tool_complete", tool=tool_name)
        except Exception as e:
            logger.warning("langfuse_tool_complete_error", error=str(e), error_type=type(e).__name__)

        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """Finalize trace when invocation completes."""
        if not self.enabled or not self.client or not self._current_trace:
            return

        try:
            self._current_trace.update(
                output={"status": "completed"},
                metadata={
                    "total_input_tokens": self._token_usage.input_tokens,
                    "total_output_tokens": self._token_usage.output_tokens,
                    "total_cost_usd": self._token_usage.cost_usd,
                },
            )
            self._current_trace.end()
            logger.info(
                "langfuse_invocation_complete",
                input_tokens=self._token_usage.input_tokens,
                output_tokens=self._token_usage.output_tokens,
                cost_usd=self._token_usage.cost_usd,
            )
            self._current_trace = None
        except Exception as e:
            logger.warning("langfuse_finalize_error", error=str(e), error_type=type(e).__name__)

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
                logger.debug("langfuse_flushed")
            except Exception as e:
                logger.warning("langfuse_flush_error", error=str(e), error_type=type(e).__name__)

    async def close(self) -> None:
        """Close plugin and flush events."""
        await self.flush()
