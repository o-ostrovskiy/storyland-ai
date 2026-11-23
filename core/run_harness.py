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

import anyio
from async_timeout import timeout as async_timeout

from common.logging import get_logger
from common.search_grounding import extract_search_metadata

from .events import (
    BookRecommendationsReady,
    DomainEvent,
    ExpansionReady,
    ItineraryReady,
    Phase,
    ProgressEvent,
    RegionsReady,
    WorkflowComplete,
    WorkflowError,
)

logger = get_logger("storyland.core.run_harness")


@dataclass
class RunCapture:
    """Mutable sink filled while draining a runner's event stream."""

    final_response: object = None
    captured_texts: dict[str, list[str]] = field(default_factory=dict)
    captured_queries: dict[str, list[str]] = field(default_factory=dict)

    def text_for(self, author: str) -> str:
        """Joined captured text for one author ('' when none captured)."""
        return "\n".join(self.captured_texts.get(author, []))

    def queries_for(self, author: str) -> list[str]:
        """Search queries one author ran, in order ([] when none captured)."""
        return list(self.captured_queries.get(author, []))


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
        ``capture.captured_texts`` (grounding post-validation input), and the
        google_search queries they ran (either receipt channel, see
        ``common.search_grounding``) into ``capture.captured_queries``.
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
            if capture is not None and author in capture_authors:
                receipts = extract_search_metadata(event)
                if receipts is not None and receipts.queries:
                    seen = capture.captured_queries.setdefault(author, [])
                    seen.extend(q for q in receipts.queries if q not in seen)
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

# The success payload each flow yields once. After one of these went out, the
# flow's work is delivered: a later cancel skips on_cancel.
RESULT_EVENTS = (
    RegionsReady,
    ItineraryReady,
    ExpansionReady,
    BookRecommendationsReady,
)

# Upper bound on a cancel cleanup (a session read + one append on SQLite take
# milliseconds). The shield must never turn a cancel into a hang.
CANCEL_CLEANUP_TIMEOUT_S = 5.0


async def _run_cancel_cleanup(spec: "GuardSpec", cleanup: Cleanup) -> None:
    """Run ``cleanup`` to completion even though the caller was cancelled.

    sse_starlette cancels the stream through an anyio cancel scope, and anyio
    re-delivers that cancel at EVERY later ``await`` in the scope, so a plain
    ``await cleanup()`` here died at its first DB call: the job was never
    marked failed and the expand / book-recs in-progress flags were never
    cleared (a dropped expand then refused every later expand with
    ExpansionInProgress). A shielded scope with its own deadline lets the
    cleanup finish, bounded; the caller re-raises the cancel right after.
    """
    with anyio.move_on_after(CANCEL_CLEANUP_TIMEOUT_S, shield=True) as scope:
        try:
            await cleanup()
        except Exception as e:
            logger.warning(
                f"{spec.flow_name}_cancel_cleanup_failed",
                job_id=spec.job_id,
                error_type=type(e).__name__,
            )
    if scope.cancelled_caught:
        logger.warning(f"{spec.flow_name}_cancel_cleanup_timeout", job_id=spec.job_id)


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
    placement. CancelledError (and GeneratorExit, a consumer that stopped at a
    yield) runs ``on_cancel`` shielded and bounded, then re-raises (the
    caller's cancellation must propagate); ``on_cancel`` is skipped once the
    flow's result event was already handed on. Timeout and generic exceptions
    terminate the stream with WorkflowError + WorkflowComplete.
    """
    # Set once the flow's result has been handed to the consumer. A cancel
    # after that point cut only bookkeeping (e.g. the Langfuse flush before
    # `done`) -- the MCP server closes discover right after `regions` -- so
    # on_cancel must not mark a delivered job failed.
    result_sent = False
    try:
        async with async_timeout(spec.timeout_seconds):
            async for ev in body:
                yield ev
                if isinstance(ev, RESULT_EVENTS):
                    result_sent = True
    except GeneratorExit:
        # The consumer stopped iterating at a yield (the SSE transport was
        # cancelled mid-send and the generator is being closed). Same cleanup
        # as a cancel; awaiting is allowed here, yielding is not.
        if spec.on_cancel is not None and not result_sent:
            await _run_cancel_cleanup(spec, spec.on_cancel)
        raise
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
        logger.warning(
            f"{spec.flow_name}_cancelled", job_id=spec.job_id, result_sent=result_sent
        )
        if spec.on_cancel is not None and not result_sent:
            await _run_cancel_cleanup(spec, spec.on_cancel)
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
