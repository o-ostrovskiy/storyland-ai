"""
Server-side core-funnel telemetry (additive, feature-flagged).

Why this exists
---------------
GA4 already runs on the frontend, but client-side analytics is unreliable for
exactly the signals the Product Analyst needs in the weekly review: failed
searches, real backend latency, and the book-vs-place entry split. This module
emits those funnel stages from the server, where they can always be observed.

It is intentionally a thin logging seam. When ``ANALYTICS_ENABLED`` is false
(the default) every call is a no-op, so wiring it in changes nothing in
production until the flag is flipped. A later slice can forward these same
events to the GA4 Measurement Protocol or PostHog without touching call sites.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import AsyncGenerator, Callable, Optional, Tuple, Type

from common.logging import get_logger
from core.events import DomainEvent, WorkflowError

logger = get_logger("storyland.telemetry.funnel")


class FunnelStage(str, Enum):
    """Core funnel stages emitted from the backend."""

    SEARCH_SUBMITTED = "search_submitted"
    RESULT_SHOWN = "result_shown"
    SEARCH_EMPTY = "search_empty"
    SEARCH_FAILED = "search_failed"


class SearchEntry(str, Enum):
    """How the search was started -- the funnel's first dimension."""

    BOOK = "book"  # discover flow: starts from a book
    PLACE = "place"  # local-atmosphere flow: starts from a place/location


class FunnelTelemetry:
    """Emits structured funnel events. No-op unless explicitly enabled."""

    def __init__(self, enabled: bool, environment: str = "local") -> None:
        self._enabled = enabled
        self._environment = environment

    @property
    def enabled(self) -> bool:
        return self._enabled

    def emit(
        self,
        stage: FunnelStage,
        *,
        entry: SearchEntry,
        user_id: str,
        latency_ms: Optional[int] = None,
        **fields: object,
    ) -> None:
        """Record one funnel event. Silently does nothing when disabled."""
        if not self._enabled:
            return
        extra = {k: v for k, v in fields.items() if v is not None}
        logger.info(
            "funnel_event",
            stage=stage.value,
            entry=entry.value,
            user_id=user_id,
            latency_ms=latency_ms,
            environment=self._environment,
            **extra,
        )


def get_funnel_telemetry(config: object) -> FunnelTelemetry:
    """Build a FunnelTelemetry from app config (safe defaults if absent)."""
    return FunnelTelemetry(
        enabled=bool(getattr(config, "analytics_enabled", False)),
        environment=str(getattr(config, "environment", "local")),
    )


async def track_funnel(
    stream: AsyncGenerator[DomainEvent, None],
    *,
    telemetry: FunnelTelemetry,
    entry: SearchEntry,
    user_id: str,
    result_types: Tuple[Type[DomainEvent], ...],
    is_empty: Callable[[DomainEvent], bool],
) -> AsyncGenerator[DomainEvent, None]:
    """
    Wrap a domain-event stream and emit funnel stages around it.

    Emits ``search_submitted`` on entry, then exactly one terminal stage:
    ``result_shown`` (a non-empty result event was seen), ``search_empty``
    (the flow finished with no usable result), or ``search_failed`` (a
    WorkflowError was emitted or an exception propagated). Events pass through
    untouched, so this is transparent to consumers.
    """
    start = time.monotonic()
    saw_result = False
    saw_error = False
    telemetry.emit(FunnelStage.SEARCH_SUBMITTED, entry=entry, user_id=user_id)
    try:
        async for event in stream:
            if isinstance(event, result_types) and not is_empty(event):
                saw_result = True
            elif isinstance(event, WorkflowError):
                saw_error = True
            yield event
    except Exception:
        saw_error = True
        raise
    finally:
        latency_ms = int((time.monotonic() - start) * 1000)
        if saw_error:
            stage = FunnelStage.SEARCH_FAILED
        elif saw_result:
            stage = FunnelStage.RESULT_SHOWN
        else:
            stage = FunnelStage.SEARCH_EMPTY
        telemetry.emit(stage, entry=entry, user_id=user_id, latency_ms=latency_ms)
