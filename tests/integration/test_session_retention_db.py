"""Integration test: retention sweep against a REAL ADK 2 SQLite session store.

The DB prune in services/session_retention.py is deliberately fail-open (a
schema/engine mismatch logs a warning and returns 0 rather than crashing
serving) — which means an ADK schema change could silently turn the prune
into a no-op and the only symptom would be unbounded DB growth. This test
converts that from "unknown" into "tested": it drives the real
DatabaseSessionService (the same v-current schema prod gets on a fresh DB),
backdates one session, and asserts the sweep actually deletes it.

No network involved (local SQLite only), so this runs in the default
integration gate — it is NOT a real_api test.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from services.session_retention import prune_sessions

pytestmark = pytest.mark.integration

APP = "storyland"
USER = "retention-test"


async def _make_db_service(tmp_path):
    from google.adk.sessions import DatabaseSessionService

    db_file = tmp_path / "retention_test_sessions.db"
    return DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db_file}")


async def _session_ids(svc):
    engine = svc.db_engine
    async with engine.connect() as conn:
        rows = (
            await conn.execute(text("SELECT id FROM sessions ORDER BY id"))
        ).fetchall()
    return sorted(r[0] for r in rows)


async def _backdate(svc, session_id: str, days: int) -> None:
    engine = svc.db_engine
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE sessions SET update_time = :t WHERE id = :sid"),
            {"t": cutoff, "sid": session_id},
        )


class TestDatabaseRetentionSweep:
    async def test_stale_session_is_pruned_fresh_one_survives(self, tmp_path):
        svc = await _make_db_service(tmp_path)
        await svc.create_session(
            app_name=APP, user_id=USER, session_id="stale", state={"k": 1}
        )
        await svc.create_session(
            app_name=APP, user_id=USER, session_id="fresh", state={"k": 2}
        )
        assert await _session_ids(svc) == ["fresh", "stale"]

        await _backdate(svc, "stale", days=30)

        evicted = await prune_sessions(
            svc, ttl_seconds=7 * 24 * 3600, max_entries=1000
        )

        assert evicted == 1, (
            "the DB prune deleted nothing — the sweeper's raw SQL no longer "
            "matches ADK's session schema (fail-open turned into a silent no-op)"
        )
        assert await _session_ids(svc) == ["fresh"]

    async def test_prune_is_noop_when_nothing_stale(self, tmp_path):
        svc = await _make_db_service(tmp_path)
        await svc.create_session(
            app_name=APP, user_id=USER, session_id="only", state={}
        )
        evicted = await prune_sessions(
            svc, ttl_seconds=7 * 24 * 3600, max_entries=1000
        )
        assert evicted == 0
        assert await _session_ids(svc) == ["only"]

    async def test_engine_discovery_finds_adk2_engine(self, tmp_path):
        """The sweeper probes db_engine/_engine/engine; ADK 2 exposes db_engine."""
        from services.session_retention import _engine_of

        svc = await _make_db_service(tmp_path)
        assert _engine_of(svc) is not None
