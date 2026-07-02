"""Persistent, async-safe TTL cache backed by ``diskcache`` (SQLite on disk).

This is the persistent sibling of the in-process :class:`core.cache.TTLCache`.
It exposes the *same* async interface (``get`` / ``set`` / ``clear`` /
``close`` / ``len``) so it is a drop-in replacement for the Discovery result
cache, but stores entries in a SQLite database under a directory that is mounted
on a docker volume. Entries therefore survive:

* container restart, and
* redeploy (image rebuild),

which fixes the cold-on-restart failure mode where every deploy re-paid live
Gemini + Google Books cost. A single on-disk store is also shared across worker
processes on the box, so a hit computed by one worker serves them all.

Design notes:

* ``diskcache`` is synchronous, so every operation is dispatched to a worker
  thread via ``asyncio.to_thread`` to avoid blocking the event loop.
* TTL is enforced by diskcache's per-entry ``expire``; staleness is bounded by
  the TTL exactly as before.
* Growth is bounded by a byte ``size_limit`` derived from ``max_entries`` with
  least-recently-used eviction. The bound moves from an exact entry *count* (the
  old process-local dict) to an equivalent on-disk *byte budget* — the right
  primitive for a shared, persistent store. Region-analysis entries are small
  (a few KB), so the budget comfortably holds ``max_entries`` of them.
* Values are returned as-is; callers cache only already-validated data, so a hit
  can never introduce a new fabrication (identical guarantee to TTLCache).
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

# Byte budget assumed per cached region-analysis entry when deriving the on-disk
# size limit from the configured entry count. Generous headroom over a typical
# few-KB region set.
_BYTES_PER_ENTRY = 32 * 1024
_MIN_SIZE_LIMIT = 8 * 1024 * 1024  # never smaller than 8 MB


class DiskTTLCache:
    """An async-safe, persistent TTL cache with LRU + byte-bounded eviction.

    Args:
        directory: On-disk directory for the SQLite store (created if absent);
            mount this on a docker volume for cross-restart persistence.
        ttl_seconds: Time-to-live for each entry, in seconds.
        max_entries: Approximate entry budget; converted to an on-disk byte
            ``size_limit`` with LRU eviction.
        size_limit: Explicit byte size limit (overrides the ``max_entries``
            derivation; mainly for tests).
    """

    def __init__(
        self,
        directory: str,
        ttl_seconds: int,
        max_entries: int,
        size_limit: Optional[int] = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        # Lazy import so environments that never select the disk backend don't
        # need the dependency importable at module load.
        import diskcache

        self._ttl = ttl_seconds
        self._max = max_entries
        self._size_limit = size_limit or max(max_entries * _BYTES_PER_ENTRY, _MIN_SIZE_LIMIT)
        self._cache = diskcache.Cache(
            directory=str(directory),
            size_limit=self._size_limit,
            eviction_policy="least-recently-used",
        )

    @property
    def directory(self) -> str:
        return self._cache.directory

    async def get(self, key: str) -> Optional[Any]:
        """Return the cached value for ``key`` or ``None`` on miss/expiry."""
        return await asyncio.to_thread(self._cache.get, key)

    async def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` with the configured TTL."""
        await asyncio.to_thread(self._cache.set, key, value, self._ttl)

    async def clear(self) -> None:
        """Drop all entries (used by tests and the rollback path)."""
        await asyncio.to_thread(self._cache.clear)

    def close(self) -> None:
        """Release the SQLite handle. Safe to call more than once."""
        try:
            self._cache.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self._cache)
