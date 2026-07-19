"""
Shared runner harness for WorkflowExecutor flows.

Every executor flow (discover / compose / local_atmosphere / expand /
recommend_books, plus the place→book resolver) drains an ADK Runner the same
way: map agent authors to ProgressEvents, optionally capture researcher text
and the final response, then close out with token usage and a terminal event —
all under one workflow timeout with a per-flow cleanup policy.

This module owns that scaffolding so the ADK-facing surface (event iteration,
timeout/error mapping) lives in ONE place. Flow-specific business logic
(guards, caching, merging, grounding filters) stays in the executor. Runner
*construction* deliberately stays in the calling module so existing test
seams (monkeypatching ``core.executor.Runner`` / ``core.place_to_book.Runner``)
keep working at their historical patch points.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, AsyncIterator, Awaitable, Callable, Optional

from async_timeout import timeout as async_timeout

from common.logging import get_logger

from .events import (
    DomainEvent,
    Phase,
    ProgressEvent,
    WorkflowComplete,
    WorkflowError,
)

logger = get_logger("storyland.core.run_harness")


@dataclass
class RunCapture:
    """Mutable sink filled while draining a runner's event stream."""

    final_response: object = None
    captured_texts: dict[str, list[str]] = field(default_factory=dict)

    def text_for(self, author: str) -> str:
        """Joined captured text for one author ('' when none captured)."""
        return "\n".join(self.captured_texts.get(author, []))


async def pump_events(
    runner,
    *,
    user_id: str,
    session_id: str,
    message,
    agent_steps: dict[str, str],
    phase: Optional[Phase] = None,
    capture: Optional[RunCapture] = None,
    capture_authors: tuple[str, ...] = (),
    track_final_response: bool = False,
) -> AsyncGenerator[ProgressEvent, None]:
    """Drain ``runner.run_async`` yielding at most one ProgressEvent per agent.

    ``phase`` is required only when ``agent_steps`` is non-empty (it tags the
    ProgressEvents); a pure drain (empty ``agent_steps`` — e.g. the place→book
    resolver) passes no phase rather than inventing one.

    Capture semantics (both require ``capture``):
      * ``capture_authors``: text parts from these authors are accumulated into
        ``capture.captured_texts`` (grounding post-validation input).
      * ``track_final_response``: the last event for which
        ``is_final_response()`` is true is stored on ``capture.final_response``.
    """
    if agent_steps and phase is None:
        raise ValueError("pump_events: phase is required when agent_steps is non-empty")
    reported: set[str] = set()
    async with runner:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            author = getattr(event, "author", None)
            if (
                capture is not None
                and author in capture_authors
                and event.content
                and event.content.parts
            ):
                for part in event.content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        capture.captured_texts.setdefault(author, []).append(text)
            if author and author in agent_steps and author not in reported:
                reported.add(author)
                yield ProgressEvent(
                    phase=phase,
                    step=agent_steps[author],
                    detail=author,
                )
            if (
                track_final_response
                and capture is not None
                and event.is_final_response()
            ):
                capture.final_response = event


# Async cleanup hook run before the terminal error events are emitted.
Cleanup = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class GuardSpec:
    """Per-flow policy for the shared timeout/exception boundary.

    ``map_exception`` (when set) owns BOTH the logging and the WorkflowError
    for generic exceptions — used by discover() to classify TaskGroup
    failures into client-safe typed errors. Flows without it get the default
    ``{flow_name}_error`` log and a str(e)/type-name WorkflowError, preserving
    each flow's historical behavior.
    """

    flow_name: str
    job_id: str
    timeout_seconds: float
    timeout_message: str
    phase: Optional[Phase] = None
    on_timeout: Optional[Cleanup] = None
    on_cancel: Optional[Cleanup] = None
    on_error: Optional[Cleanup] = None
    map_exception: Optional[Callable[[Exception], WorkflowError]] = None


async def run_guarded(
    body: AsyncIterator[DomainEvent], spec: GuardSpec
) -> AsyncGenerator[DomainEvent, None]:
    """Run a flow body under the workflow timeout with per-flow error policy.

    The timeout context wraps the *iteration* of ``body``, so consumer time
    (e.g. a slow SSE client between events) counts against the workflow
    timeout — identical to the historical inline ``async with timeout(...)``
    placement. CancelledError performs cleanup and re-raises (the caller's
    cancellation must propagate); Timeout and generic exceptions terminate the
    stream with WorkflowError + WorkflowComplete.
    """
    try:
        async with async_timeout(spec.timeout_seconds):
            async for ev in body:
                yield ev
    except TimeoutError:
        logger.error(f"{spec.flow_name}_timeout", job_id=spec.job_id)
        if spec.on_timeout is not None:
            await spec.on_timeout()
        yield WorkflowError(
            message=spec.timeout_message,
            error_type="WorkflowTimeoutError",
            phase=spec.phase,
        )
        yield WorkflowComplete(job_id=spec.job_id)
    except asyncio.CancelledError:
        logger.warning(f"{spec.flow_name}_cancelled", job_id=spec.job_id)
        if spec.on_cancel is not None:
            await spec.on_cancel()
        raise
    except Exception as e:
        if spec.map_exception is not None:
            error_event = spec.map_exception(e)
        else:
            logger.error(
                f"{spec.flow_name}_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            error_event = WorkflowError(
                message=str(e),
                error_type=type(e).__name__,
                phase=spec.phase,
            )
        if spec.on_error is not None:
            await spec.on_error()
        yield error_event
        yield WorkflowComplete(job_id=spec.job_id)


async def collect_token_usage(langfuse_plugin) -> Optional[dict]:
    """Session token stats + flush when the plugin is enabled, else None."""
    if langfuse_plugin.enabled:
        token_usage = langfuse_plugin.get_session_stats()
        await langfuse_plugin.flush()
        return token_usage
    return None


def error_events(
    job_id: str,
    message: str,
    error_type: str,
    phase: Optional[Phase] = None,
    **extra,
) -> tuple[DomainEvent, ...]:
    """The standard terminal pair for a rejected/failed request."""
    return (
        WorkflowError(message=message, error_type=error_type, phase=phase, **extra),
        WorkflowComplete(job_id=job_id),
    )
