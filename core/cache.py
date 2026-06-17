"""
Tiny in-process TTL + LRU cache for expensive recommendation results.

Scope (G1, opportunity "Cache expensive Gemini / Google Books recommendation
calls"): an in-memory, per-process cache with no external dependencies. It is
used to short-circuit the most-repeated, highest-cost chain (Discovery:
book -> regions) when the same book/author/preferences are requested again.

Design notes:
- Monotonic-clock expiry (immune to wall-clock changes).
- ``max_entries`` cap with oldest-first ("LRU-ish": insertion/refresh order)
  eviction.
- An ``asyncio.Lock`` guards all mutations so concurrent requests are safe.
- Values are returned as-is; callers cache only already-validated data, so a
  hit can never introduce a *new* fabrication. Staleness is bounded by the TTL.

This cache is intentionally simple. A shared/persistent cache (Redis/KV) is an
explicit follow-up and would add infrastructure (G4), so it is out of scope.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any, Optional


class TTLCache:
    """An async-safe, bounded TTL cache with oldest-first eviction.

    Args:
        ttl_seconds: Time-to-live for each entry, in seconds. Entries older
            than this are treated as misses and dropped on access.
        max_entries: Maximum number of live entries. When exceeded, the
            oldest entry is evicted.
        clock: Monotonic time source (override for tests). Defaults to
            ``time.monotonic``.
    """

    def __init__(
        self,
        ttl_seconds: int,
        max_entries: int,
        clock=time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock
        self._lock = asyncio.Lock()
        # key -> (expires_at, value); ordered oldest-first for eviction.
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()

    async def get(self, key: str) -> Optional[Any]:
        """Return the cached value for ``key`` or ``None`` on miss/expiry."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._clock() >= expires_at:
                # Expired: drop and report a miss.
                self._store.pop(key, None)
                return None
            # Refresh recency so frequently used entries survive eviction.
            self._store.move_to_end(key)
            return value

    async def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``, evicting the oldest if over capacity."""
        async with self._lock:
            expires_at = self._clock() + self._ttl
            if key in self._store:
                self._store.pop(key, None)
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                # Evict oldest (FIFO/LRU-ish).
                self._store.popitem(last=False)

    async def clear(self) -> None:
        """Drop all entries (used by tests and the rollback path)."""
        async with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
