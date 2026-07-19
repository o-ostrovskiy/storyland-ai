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
    from langfuse import Langfuse, propagate_attributes
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


# Gemini standard (non-batch) pricing per 1M tokens (as of May 2026).
# Source: https://ai.google.dev/gemini-api/docs/pricing
_GEMINI_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1m, output_per_1m)
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
}
_GEMINI_DEFAULT_PRICING = (0.30, 2.50)  # fall back to gemini-2.5-flash rates


def _pricing_for(model_name: str) -> tuple[float, float]:
    """Return (input_per_1m, output_per_1m) for the given model name."""
    if not model_name:
        return _GEMINI_DEFAULT_PRICING
    lower = model_name.lower()
    # MYS-398 follow-up: match longest key first. `_GEMINI_PRICING` is
    # keyed by substring ("key in lower"), and "gemini-2.5-flash" is
    # itself a substring of "gemini-2.5-flash-lite" -- checking keys in
    # plain insertion order let the shorter, wrong entry match first, so
    # every flash-lite call was silently priced at the plain-flash rate
    # (pre-existing, caught by the MYS-398 concurrency test asserting two
    # different flash variants' costs side by side for the first time).
    for key in sorted(_GEMINI_PRICING, key=len, reverse=True):
        if key in lower:
            return _GEMINI_PRICING[key]
    return _GEMINI_DEFAULT_PRICING


def _calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """Calculate cost in USD given token counts and model name."""
    input_rate, output_rate = _pricing_for(model_name)
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


@dataclass
class TokenUsage:
    """Token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


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
        root_name: Optional[str] = None,
    ):
        """
        Initialize Langfuse plugin.

        Args:
            secret_key: Langfuse secret key (from env LANGFUSE_SECRET_KEY)
            public_key: Langfuse public key (from env LANGFUSE_PUBLIC_KEY)
            host: Langfuse host URL (from env LANGFUSE_HOST)
            enabled: Enable/disable plugin (auto-disabled if credentials missing)
            root_name: Trace-name identity for the run's root workflow. Under
                Runner(node=...) ADK 2.5 builds InvocationContext with
                agent=None (and node_path=None at user-message time), so the
                plugin cannot recover the root identity from the context —
                the caller that builds the Runner KNOWS the workflow and
                injects its name here (see WorkflowExecutor._build_runner).
        """
        super().__init__(name="langfuse_plugin")
        self.root_name = root_name
        self.enabled = enabled and LANGFUSE_AVAILABLE
        self.client: Optional[Langfuse] = None
        self._token_usage = TokenUsage()
        self._total_cost_usd: float = 0.0
        self._current_trace = None
        # MYS-398: discovery's parallel_discovery ParallelAgent runs three
        # sub-agent branches (city/landmark/author pipelines) concurrently
        # against this ONE plugin instance (core/executor.py creates one
        # LangfusePlugin per workflow *run*, not per sub-agent). The fields
        # below used to be shared scalars/a single LIFO list, so an
        # interleaved before_model/after_model pair from one branch could
        # overwrite or pop state that belonged to a different, still-open
        # branch -- see _branch_key() for how they're now scoped per branch.
        self._generations: dict[str, Any] = {}  # branch key -> open generation observation
        self._models: dict[str, str] = {}  # branch key -> model name of the open generation
        self._agent_stacks: dict[str, list] = {}  # branch key -> that branch's own LIFO agent stack
        self._spans: dict[str, Any] = {}  # branch key -> open tool span
        self._propagation_cm = None

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

    def _resolve_root_name(self, invocation_context: InvocationContext) -> str:
        """Trace-name identity for this invocation, explicit-first.

        NOT the getattr(x, 'attr', default) idiom: when
        ``invocation_context.agent`` is None (every Runner(node=...) root on
        ADK 2.5), that idiom silently returns the default — the third
        silent-wrong-answer of its kind in this migration (after the
        usage_metadata declared-field trap). Explicit chain instead: real
        agent name → injected root_name → loud WARNING + 'unknown_agent'.
        """
        agent = invocation_context.agent
        if agent is not None:
            name = getattr(agent, "name", None)
            if name:
                return name
        if self.root_name:
            return self.root_name
        logger.warning(
            "langfuse_root_identity_missing",
            invocation_id=getattr(invocation_context, "invocation_id", None),
            hint="pass root_name= or set plugin.root_name before Runner construction",
        )
        return "unknown_agent"

    @staticmethod
    def _branch_key(callback_context: CallbackContext) -> str:
        """Per-concurrent-call key so parallel sub-agent branches never
        share the same generation/agent-stack/span slot (MYS-398).

        ADK's own ParallelAgent already solves this problem for itself:
        every sub-agent gets a private InvocationContext copy carrying a
        unique, stable `branch` string for the sub-agent's entire run (see
        google.adk.agents.parallel_agent._create_branch_ctx_for_sub_agent,
        e.g. "parallel_discovery.city_pipeline"). Nested SequentialAgent
        steps within a branch (research -> format) do not fork a new
        branch, so they safely share one key -- only ParallelAgent forks,
        and this plugin only needs to isolate exactly that boundary.

        ADK 2.x exposes `branch` as a PUBLIC property on Context (and thus
        CallbackContext/ToolContext), which removed the private
        `_invocation_context` coupling this plugin carried on the 1.x line
        (the coupling that motivated the old `<2` pin). The private
        attribute is kept as a fallback only for defense in depth. Falls
        back to a constant key when `branch` is unset (sequential-only
        paths) -- a single shared key is exactly what non-parallel
        invocations already had, and correct for them since there's never
        more than one in-flight generation there.
        """
        branch = getattr(callback_context, "branch", None)
        if branch is None:
            try:
                branch = callback_context._invocation_context.branch
            except AttributeError:
                branch = None
        return branch or "_root"

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
            agent_name = self._resolve_root_name(invocation_context)
            user_id = invocation_context.user_id
            session_id = invocation_context.session.id if invocation_context.session else None

            # Propagate trace-level attributes to root span and all children
            self._propagation_cm = propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                trace_name=f"{agent_name}_invocation",
            )
            self._propagation_cm.__enter__()

            self._current_trace = self.client.start_observation(
                as_type="span",
                name=f"{agent_name}_invocation",
                metadata={
                    "invocation_id": invocation_context.invocation_id,
                    "session_id": session_id,
                    "agent_type": "google_adk",
                },
                input=str(user_message),
            )
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
            span = self._current_trace.start_observation(
                as_type="span",
                name=agent_name,
                metadata={
                    "agent_type": type(agent).__name__,
                    "callback_context": str(callback_context),
                },
            )
            key = self._branch_key(callback_context)
            self._agent_stacks.setdefault(key, []).append((agent_name, span))
            logger.debug("langfuse_agent_start", agent=agent_name, branch=key)
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
            key = self._branch_key(callback_context)
            stack = self._agent_stacks.get(key)
            if stack:
                agent_name, span = stack.pop()
                if not stack:
                    self._agent_stacks.pop(key, None)
                span.end()
                logger.debug("langfuse_agent_complete", agent=agent_name, branch=key)
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
            model_name = getattr(llm_request, 'model', 'unknown_model') or 'unknown_model'
            key = self._branch_key(callback_context)
            self._models[key] = model_name

            # Create generation tracking under the current agent span (this
            # branch's own stack top) or the shared trace span.
            stack = self._agent_stacks.get(key) or []
            parent = stack[-1][1] if stack else self._current_trace
            self._generations[key] = parent.start_observation(
                as_type="generation",
                name=f"{model_name}_call",
                model=model_name,
                input=str(llm_request),
                metadata={
                    "agent": stack[-1][0] if stack else "unknown",
                },
            )
            logger.debug("langfuse_model_request", model=model_name, branch=key)
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
        key = self._branch_key(callback_context)
        generation = self._generations.get(key)
        if not self.enabled or not self.client or not generation:
            return None

        try:
            # Extract token usage from response
            usage = self._extract_token_usage(llm_response)
            model_name = self._models.get(key, "")

            # Explicit None check: _extract_token_usage never returns an
            # all-zero TokenUsage, and a dataclass instance is always truthy,
            # so `if usage:` would mask a missing-usage response.
            if usage is not None:
                # Update running totals. Plain += on dataclass fields/floats
                # is safe to share across branches: asyncio is single-
                # threaded and these statements contain no `await`, so each
                # increment runs to completion without interleaving with
                # another branch's increment.
                self._token_usage.input_tokens += usage.input_tokens
                self._token_usage.output_tokens += usage.output_tokens
                self._token_usage.total_tokens += usage.total_tokens

                cost = _calculate_cost(usage.input_tokens, usage.output_tokens, model_name)
                self._total_cost_usd += cost
                input_rate, output_rate = _pricing_for(model_name)
                generation.update(
                    output=str(llm_response),
                    usage_details={
                        "input": usage.input_tokens,
                        "output": usage.output_tokens,
                        "total": usage.total_tokens,
                    },
                    cost_details={
                        "total_cost": cost,
                    },
                    metadata={
                        "model_pricing": {
                            "input_per_1m": input_rate,
                            "output_per_1m": output_rate,
                        },
                    },
                )
                logger.debug(
                    "langfuse_model_response",
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=cost,
                    branch=key,
                )
            else:
                # A None here means the response carried no recognizable
                # usage_metadata — after an SDK/ADK upgrade that's the first
                # (and previously ONLY silent) sign that cost tracking has
                # zeroed out. Loud by design; never raises.
                logger.warning(
                    "langfuse_token_usage_missing",
                    model=model_name,
                    branch=key,
                    response_type=type(llm_response).__name__,
                )

            generation.end()
            self._generations.pop(key, None)
            self._models.pop(key, None)
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
            TokenUsage object, or None if usage is absent OR carries no
            counts. LlmResponse is pydantic, so ``usage_metadata`` is a
            declared field and ``hasattr`` is ALWAYS true — when the value is
            None, ``getattr(None, ..., 0)`` used to fabricate a truthy
            TokenUsage(0, 0, 0) that sailed through the ``if usage`` success
            branch and recorded a zero-cost generation without ever firing
            the missing-usage warning. A real generation can never be 0/0/0
            (the prompt alone costs tokens), so all-zero is normalized to
            None here and the caller's warning path owns it.
        """
        try:
            usage = None
            # Try the usage_metadata attribute (guarding the None value the
            # declared-field hasattr check can't catch).
            metadata = getattr(response, 'usage_metadata', None)
            if metadata is not None:
                usage = TokenUsage(
                    input_tokens=getattr(metadata, 'prompt_token_count', 0) or 0,
                    output_tokens=getattr(metadata, 'candidates_token_count', 0) or 0,
                    total_tokens=getattr(metadata, 'total_token_count', 0) or 0,
                )
            # Alternative: dict-like access
            elif isinstance(response, dict) and response.get('usage_metadata'):
                md = response['usage_metadata']
                usage = TokenUsage(
                    input_tokens=md.get('prompt_token_count') or 0,
                    output_tokens=md.get('candidates_token_count') or 0,
                    total_tokens=md.get('total_token_count') or 0,
                )

            if usage is None or (
                usage.input_tokens == 0
                and usage.output_tokens == 0
                and usage.total_tokens == 0
            ):
                return None
            return usage
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
            key = self._branch_key(tool_context)
            stack = self._agent_stacks.get(key) or []
            parent = stack[-1][1] if stack else self._current_trace
            self._spans[key] = parent.start_observation(
                as_type="span",
                name=f"tool_{tool_name}",
                metadata={
                    "tool_type": "adk_tool",
                    "tool_name": tool_name,
                    "tool_args": str(tool_args),
                },
            )
            logger.debug("langfuse_tool_start", tool=tool_name, branch=key)
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
        key = self._branch_key(tool_context)
        span = self._spans.get(key)
        if not self.enabled or not self.client or not span:
            return None

        try:
            tool_name = getattr(tool, 'name', 'unknown_tool')
            span.update(output=str(result))
            span.end()
            self._spans.pop(key, None)
            logger.debug("langfuse_tool_complete", tool=tool_name, branch=key)
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
                    "total_cost_usd": self._total_cost_usd,
                },
            )
            self._current_trace.end()
            logger.info(
                "langfuse_invocation_complete",
                input_tokens=self._token_usage.input_tokens,
                output_tokens=self._token_usage.output_tokens,
                cost_usd=self._total_cost_usd,
            )
            self._current_trace = None
        except Exception as e:
            logger.warning("langfuse_finalize_error", error=str(e), error_type=type(e).__name__)
        finally:
            if self._propagation_cm:
                self._propagation_cm.__exit__(None, None, None)
                self._propagation_cm = None

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
            "cost_usd": self._total_cost_usd,
        }

    def reset_stats(self) -> None:
        """Reset token usage statistics."""
        self._token_usage = TokenUsage()
        self._total_cost_usd = 0.0

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
