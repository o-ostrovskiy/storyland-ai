"""
A client that drops the SSE stream mid-generation must return its DB
connection to the pool (storyland-ai-private#11).

The drop is the real one: an ``EventSourceResponse`` whose ASGI ``receive``
answers ``http.disconnect`` the moment a pooled connection is checked out, so
sse_starlette cancels its anyio task group while ADK's
``DatabaseSessionService`` is mid-query. anyio re-delivers that cancellation
at every later ``await`` in the scope, which is what used to cancel ADK's own
``rollback()`` / ``close()`` and strand the connection (the GC
"non-checked-in connection" and ``no active connection`` Sentry issues).

The assertion is on the pool, not on the absence of a log line.
"""

import asyncio

import pytest
from sqlalchemy import event
from sse_starlette.sse import EventSourceResponse

from services.session_service import create_session_service

APP = "storyland"
USER = "u1"
SID = "s1"


@pytest.fixture
async def db_service(tmp_path):
    svc = create_session_service(
        connection_string=f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}",
        use_database=True,
    )
    await svc.create_session(app_name=APP, user_id=USER, session_id=SID)
    yield svc
    await svc.close()


def _scope():
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/api/v1/itinerary/discover",
        "headers": [],
    }


async def _drop_stream_mid_query(svc) -> None:
    """Serve an SSE stream that reads the session forever; disconnect mid-read."""
    disconnected = asyncio.Event()
    pool = svc.db_engine.sync_engine.pool

    @event.listens_for(pool, "checkout")
    def _on_checkout(*_):
        disconnected.set()

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(_message):
        pass

    async def stream():
        while True:
            await svc.get_session(app_name=APP, user_id=USER, session_id=SID)
            yield {"data": "progress"}

    try:
        await asyncio.wait_for(EventSourceResponse(stream())(_scope(), receive, send), timeout=5)
    finally:
        event.remove(pool, "checkout", _on_checkout)


async def _settle(svc, timeout: float = 2.0) -> int:
    """Let any shielded in-flight DB work finish; return connections still out."""
    pool = svc.db_engine.sync_engine.pool
    deadline = asyncio.get_running_loop().time() + timeout
    while pool.checkedout() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    return pool.checkedout()


async def test_dropped_stream_returns_its_connection_to_the_pool(db_service):
    await _drop_stream_mid_query(db_service)

    assert await _settle(db_service) == 0


async def test_service_still_works_after_a_dropped_stream(db_service):
    await _drop_stream_mid_query(db_service)
    await _settle(db_service)

    session = await db_service.get_session(app_name=APP, user_id=USER, session_id=SID)

    assert session is not None
    assert db_service.db_engine.sync_engine.pool.checkedout() == 0


async def test_cancelled_caller_still_raises_cancelled_error(db_service):
    """The shield protects the DB work, never the caller: a cancel still cancels."""
    task = asyncio.create_task(
        db_service.get_session(app_name=APP, user_id=USER, session_id=SID)
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _settle(db_service) == 0
