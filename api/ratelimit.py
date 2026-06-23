"""
Application-level load-shedding for the expensive AI endpoints.

The discovery chain (``/discover``, ``/compose``, ``/expand``,
``/local-atmosphere``, ``/recommend-books``, ``/place-to-book``) drives a full
multi-agent Gemini workflow per call on a single-process uvicorn box. Without a
guard, one authenticated caller (or a misbehaving gateway) can issue unbounded
distinct-query searches and run many heavy chains in parallel — inflating Gemini
spend and exhausting the one Lightsail box (OOM / cascading workflow timeouts).
The result cache only absorbs *identical* repeats, not a flood of distinct
titles.

Two independent, in-process, additive guards live here. Both are config-gated
and **disabled by default** (limit ``<= 0``), so production behaviour is
unchanged until limits are set via env — additive and reversible:

1. :class:`SlidingWindowRateLimiter` — per-identity request-rate cap. Caps the
   number of requests one user/IP may make per rolling window (HTTP 429 when
   exceeded). Bounds worst-case spend per caller.
2. :class:`InFlightLimiter` — bounded concurrent-in-flight cap. Limits how many
   heavy chains run at once (HTTP 503 when full). It **sheds** load immediately
   rather than queueing, so excess requests fail fast instead of piling onto the
   single event loop.

Single-process design note: a plain ``int`` counter and per-key ``deque`` are
safe here because the asyncio event loop is single-threaded and none of these
methods ``await`` between the capacity check and the mutation.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional


class InFlightLimiter:
    """Bounds the number of concurrently in-flight expensive requests.

    ``max_in_flight <= 0`` disables the cap (every acquire succeeds), so the
    guard is a no-op until an operator opts in.
    """

    def __init__(self, max_in_flight: int) -> None:
        self._max = max_in_flight
        self._active = 0

    @property
    def enabled(self) -> bool:
        return self._max > 0

    @property
    def active(self) -> int:
        return self._active

    @property
    def max_in_flight(self) -> int:
        return self._max

    def try_acquire(self) -> bool:
        """Reserve a slot. Returns ``False`` (shed) when at capacity."""
        if not self.enabled:
            return True
        if self._active >= self._max:
            return False
        self._active += 1
        return True

    def release(self) -> None:
        """Free a previously-acquired slot (idempotent below zero)."""
        if not self.enabled:
            return
        if self._active > 0:
            self._active -= 1


class SlidingWindowRateLimiter:
    """Per-key sliding-window request-rate limiter.

    Allows at most ``max_requests`` calls per ``window_seconds`` for each key
    (typically a user id or client IP). ``max_requests <= 0`` disables it.

    Memory is bounded: per-key hit timestamps older than the window are evicted
    on access, and an occasional sweep drops keys that have gone idle so a flood
    of distinct keys can't grow the map without bound.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        gc_threshold: int = 1024,
    ) -> None:
        self._max = max_requests
        self._window = float(window_seconds)
        self._gc_threshold = gc_threshold
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    @property
    def enabled(self) -> bool:
        return self._max > 0 and self._window > 0

    @property
    def tracked_keys(self) -> int:
        return len(self._hits)

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        """Record a hit for ``key`` and report whether it is within the limit.

        ``now`` is injectable for deterministic tests; it defaults to a
        monotonic clock so wall-clock changes can't distort the window.
        """
        if not self.enabled:
            return True
        now = time.monotonic() if now is None else now

        if len(self._hits) > self._gc_threshold:
            self._gc(now)

        hits = self._hits[key]
        cutoff = now - self._window
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self._max:
            # Over the limit: don't record this hit. Drop the key if it carries
            # no live timestamps so a rejected one-off caller leaves no residue.
            if not hits:
                del self._hits[key]
            return False

        hits.append(now)
        return True

    def _gc(self, now: float) -> None:
        """Drop keys whose most recent hit has aged out of the window."""
        cutoff = now - self._window
        stale = [k for k, dq in self._hits.items() if not dq or dq[-1] <= cutoff]
        for k in stale:
            del self._hits[k]
