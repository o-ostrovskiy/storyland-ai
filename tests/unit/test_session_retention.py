"""
Unit tests for bounded session retention (services/session_retention.py).

Exercises the in-memory eviction policy (age TTL + max-entry cap), the
config kill-switch, and the SessionSweeper lifecycle against a real ADK
InMemorySessionService.
"""

import time

from services.session_service import create_session_service
from services.session_retention import SessionSweeper, prune_sessions

APP = "storyland"
USER = "u1"


def _stored(svc, sid):
    """Reach the live stored Session (get_session returns a deep copy in ADK)."""
    return svc.sessions[APP][USER][sid]


async def _inmemory_with(specs):
    """Build an in-memory service with sessions aged by `age_seconds`.

    specs: list of (session_id, age_seconds-since-last-update).
    """
    svc = create_session_service(use_database=False)
    now = time.time()
    for sid, age in specs:
        await svc.create_session(app_name=APP, user_id=USER, session_id=sid)
        _stored(svc, sid).last_update_time = now - age
    return svc


async def _exists(svc, sid):
    return (
        await svc.get_session(app_name=APP, user_id=USER, session_id=sid)
    ) is not None


class TestPruneInMemory:
    async def test_expired_evicted_survivors_untouched(self):
        svc = await _inmemory_with([("old", 100_000), ("fresh", 10)])
        evicted = await prune_sessions(svc, ttl_seconds=86400, max_entries=10_000)
        assert evicted == 1
        assert not await _exists(svc, "old")
        assert await _exists(svc, "fresh")

    async def test_recently_refreshed_session_survives(self):
        # A session updated 5s ago survives a 60s TTL even though we asked to
        # prune — mirrors append_event refreshing last_update_time on active use.
        svc = await _inmemory_with([("active", 5)])
        evicted = await prune_sessions(svc, ttl_seconds=60, max_entries=10_000)
        assert evicted == 0
        assert await _exists(svc, "active")

    async def test_over_cap_evicts_oldest_update_first(self):
        # s0 newest update ... s4 oldest; all within TTL. cap=2 -> drop 3 oldest.
        svc = await _inmemory_with([(f"s{i}", float(i)) for i in range(5)])
        evicted = await prune_sessions(svc, ttl_seconds=86400, max_entries=2)
        assert evicted == 3
        survivors = {sid for sid in (f"s{i}" for i in range(5)) if await _exists(svc, sid)}
        assert survivors == {"s0", "s1"}

    async def test_ttl_zero_is_noop(self):
        svc = await _inmemory_with([("old", 1_000_000)])
        evicted = await prune_sessions(svc, ttl_seconds=0, max_entries=10_000)
        assert evicted == 0
        assert await _exists(svc, "old")

    async def test_max_entries_zero_disables_cap(self):
        svc = await _inmemory_with([(f"s{i}", float(i)) for i in range(5)])
        evicted = await prune_sessions(svc, ttl_seconds=86400, max_entries=0)
        assert evicted == 0

    async def test_empty_store_is_noop(self):
        svc = create_session_service(use_database=False)
        assert await prune_sessions(svc, ttl_seconds=86400, max_entries=10) == 0


class TestSessionSweeper:
    def test_disabled_when_ttl_zero(self):
        sweeper = SessionSweeper(
            services=[], ttl_seconds=0, max_entries=10, interval_seconds=1
        )
        sweeper.start()
        assert sweeper._task is None  # no background task spawned (kill-switch)

    async def test_sweep_once_across_callable_provider(self):
        svc = await _inmemory_with([("old", 100_000), ("fresh", 1)])
        sweeper = SessionSweeper(
            services=lambda: [svc],
            ttl_seconds=86400,
            max_entries=10_000,
            interval_seconds=300,
        )
        assert await sweeper.sweep_once() == 1
        assert not await _exists(svc, "old")
        assert await _exists(svc, "fresh")

    async def test_provider_dedupes_shared_service(self):
        svc = await _inmemory_with([("old", 100_000)])
        sweeper = SessionSweeper(
            services=lambda: [svc, svc],  # same instance twice
            ttl_seconds=86400,
            max_entries=10_000,
            interval_seconds=300,
        )
        # Pruned exactly once — not double-counted, no error on the gone session.
        assert await sweeper.sweep_once() == 1

    async def test_start_then_stop_lifecycle(self):
        svc = create_session_service(use_database=False)
        sweeper = SessionSweeper(
            services=lambda: [svc],
            ttl_seconds=86400,
            max_entries=10,
            interval_seconds=300,
        )
        sweeper.start()
        assert sweeper._task is not None
        await sweeper.stop()
        assert sweeper._task is None
