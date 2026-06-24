"""
Bounded session retention for the in-memory / SQLite ADK session stores.

The ADK ``InMemorySessionService`` and SQLite ``DatabaseSessionService`` have no
TTL, eviction, or cleanup of their own, so on the single self-hosted box the
discover -> compose job state (regions, composer envelopes, full itineraries
keyed by ``job_id``) accumulates forever:

* in-memory -> RAM grows monotonically with traffic until an OOM kill drops
  *all* live jobs at once;
* SQLite    -> the ``sessions`` table grows unbounded, degrading lookup latency.

This module adds a single periodic background sweep (started from the FastAPI
lifespan) that evicts sessions older than ``SESSION_TTL_SECONDS`` and, for the
in-memory store, caps the total count at ``SESSION_MAX_ENTRIES`` (oldest-update
first). It is additive and reversible:

* ``SESSION_TTL_SECONDS=0`` disables the sweep entirely (config kill-switch ->
  byte-identical legacy behaviour);
* the default TTL (24h) is comfortably longer than a full discover -> compose ->
  expand lifetime, and every ``append_event`` refreshes a session's
  ``last_update_time``, so an actively-used mid-flow session is never evicted.

The sweep is intentionally tolerant: any unexpected store shape is logged and
skipped rather than raised, so a retention hiccup can never take down request
serving.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, List, Sequence, Tuple, Union

from common.logging import get_logger

logger = get_logger(__name__)


def _iter_inmemory_sessions(store) -> List[Tuple[str, str, str, object]]:
    """Flatten an InMemorySessionService's nested store into a list.

    ADK stores in-memory sessions as ``store.sessions[app_name][user_id][session_id]``.
    Returns ``(app_name, user_id, session_id, session)`` tuples. A snapshot list
    (not a live generator) so we can delete while iterating safely.
    """
    out: List[Tuple[str, str, str, object]] = []
    sessions = getattr(store, "sessions", None)
    if not isinstance(sessions, dict):
        return out
    for app_name, users in list(sessions.items()):
        if not isinstance(users, dict):
            continue
        for user_id, sess_map in list(users.items()):
            if not isinstance(sess_map, dict):
                continue
            for session_id, session in list(sess_map.items()):
                out.append((app_name, user_id, session_id, session))
    return out


def _session_age_seconds(session, now: float) -> float:
    """Seconds since a session was last updated.

    Reads ADK's ``last_update_time`` (epoch seconds, bumped on every
    ``append_event``). Unknown/garbage timestamps return 0.0 (treated as
    "fresh" so we never evict a session we can't reason about).
    """
    ts = getattr(session, "last_update_time", None)
    try:
        if ts is None:
            return 0.0
        return max(0.0, now - float(ts))
    except (TypeError, ValueError):
        return 0.0


async def _prune_inmemory(store, ttl_seconds: int, max_entries: int) -> int:
    """Evict expired + over-cap sessions from an in-memory store. Returns count."""
    entries = _iter_inmemory_sessions(store)
    if not entries:
        return 0

    now = time.time()
    evicted = 0
    survivors: List[Tuple[str, str, str, object]] = []

    # 1) Age-based expiry.
    for app_name, user_id, session_id, session in entries:
        if ttl_seconds > 0 and _session_age_seconds(session, now) > ttl_seconds:
            if await _delete_session(store, app_name, user_id, session_id):
                evicted += 1
        else:
            survivors.append((app_name, user_id, session_id, session))

    # 2) Hard cap: if still over the cap, evict oldest-update first (age/LRU).
    if max_entries > 0 and len(survivors) > max_entries:
        survivors.sort(key=lambda e: getattr(e[3], "last_update_time", 0.0) or 0.0)
        overflow = len(survivors) - max_entries
        for app_name, user_id, session_id, _session in survivors[:overflow]:
            if await _delete_session(store, app_name, user_id, session_id):
                evicted += 1

    return evicted


async def _delete_session(store, app_name: str, user_id: str, session_id: str) -> bool:
    """Best-effort ``delete_session`` (ADK's API is async). Never raises."""
    try:
        await store.delete_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        return True
    except Exception as exc:  # noqa: BLE001 - retention must never crash serving
        logger.warning(
            "session_evict_failed",
            app_name=app_name,
            session_id=session_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return False


def _engine_of(store):
    """Return the SQLAlchemy engine backing a DatabaseSessionService, if any."""
    for attr in ("db_engine", "_engine", "engine"):
        engine = getattr(store, attr, None)
        if engine is not None:
            return engine
    return None


async def _prune_database(store, ttl_seconds: int) -> int:
    """Best-effort age-based prune of the SQLite/SQL ``sessions`` table.

    Runs ``DELETE FROM sessions WHERE update_time < :cutoff`` against the ADK
    engine. Handles both sync and async SQLAlchemy engines. Any failure
    (unexpected schema, engine shape, driver) is logged and swallowed - the
    in-memory path is the primary OOM guard, and a DB prune hiccup must not
    break serving.
    """
    if ttl_seconds <= 0:
        return 0
    engine = _engine_of(store)
    if engine is None:
        return 0

    try:
        from sqlalchemy import text

        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            seconds=ttl_seconds
        )
        stmt = text("DELETE FROM sessions WHERE update_time < :cutoff")
        params = {"cutoff": cutoff}

        # Async engine (e.g. sqlite+aiosqlite).
        try:
            from sqlalchemy.ext.asyncio import AsyncEngine

            if isinstance(engine, AsyncEngine):
                async with engine.begin() as conn:
                    result = await conn.execute(stmt, params)
                return int(getattr(result, "rowcount", 0) or 0)
        except ImportError:
            pass

        # Sync engine.
        with engine.begin() as conn:
            result = conn.execute(stmt, params)
        return int(getattr(result, "rowcount", 0) or 0)
    except Exception as exc:  # noqa: BLE001 - best-effort, never crash the sweep
        logger.warning(
            "session_db_prune_skipped",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return 0


async def prune_sessions(session_service, ttl_seconds: int, max_entries: int) -> int:
    """Evict expired (and, in-memory, over-cap) sessions from one store.

    Dispatches on the store shape: an in-memory store (has a ``.sessions``
    dict) gets age + cap eviction; anything exposing a SQL engine gets the
    age-based table prune. Returns the number of sessions evicted. Never raises.
    """
    try:
        if isinstance(getattr(session_service, "sessions", None), dict):
            return await _prune_inmemory(session_service, ttl_seconds, max_entries)
        if _engine_of(session_service) is not None:
            return await _prune_database(session_service, ttl_seconds)
    except Exception as exc:  # noqa: BLE001 - defensive belt-and-suspenders
        logger.warning(
            "session_prune_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
    return 0


# A provider yields the session services that are currently live. We use a
# callable (not a fixed list) because some stores - e.g. the place->book
# resolver's isolated in-memory store - are created lazily on first request,
# after the sweep has already started.
ServicesProvider = Union[Callable[[], Iterable[object]], Sequence[object]]


class SessionSweeper:
    """Periodic background task that prunes live session stores.

    Started/stopped from the FastAPI lifespan. Disabled (no task) when
    ``ttl_seconds <= 0`` so ``SESSION_TTL_SECONDS=0`` is a true kill-switch.
    """

    def __init__(
        self,
        services: ServicesProvider,
        ttl_seconds: int,
        max_entries: int,
        interval_seconds: int,
    ):
        self._services = services
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.interval_seconds = interval_seconds
        self._task = None

    def _resolve_services(self) -> List[object]:
        provider = self._services
        try:
            items = provider() if callable(provider) else provider
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "session_services_provider_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []
        # De-dupe by identity (two callers may share a service instance).
        seen: List[object] = []
        for svc in items or []:
            if svc is not None and all(svc is not s for s in seen):
                seen.append(svc)
        return seen

    def start(self) -> None:
        """Launch the sweep loop (no-op when retention is disabled)."""
        if self.ttl_seconds <= 0:
            logger.info("session_sweep_disabled", ttl_seconds=self.ttl_seconds)
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())
        logger.info(
            "session_sweep_started",
            ttl_seconds=self.ttl_seconds,
            max_entries=self.max_entries,
            interval_seconds=self.interval_seconds,
        )

    async def stop(self) -> None:
        """Cancel the sweep loop and wait for it to unwind."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def sweep_once(self) -> int:
        """Run one prune pass across all live stores. Returns total evicted."""
        total = 0
        for svc in self._resolve_services():
            total += await prune_sessions(svc, self.ttl_seconds, self.max_entries)
        if total:
            logger.info("session_sweep_evicted", count=total)
        return total

    async def _run(self) -> None:
        interval = max(1, self.interval_seconds)
        while True:
            try:
                await asyncio.sleep(interval)
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                logger.warning(
                    "session_sweep_iteration_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
