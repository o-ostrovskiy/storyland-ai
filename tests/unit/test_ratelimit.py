"""Unit tests for the load-shedding guards (api/ratelimit.py) and the FastAPI
dependencies that wire them in (api/dependencies.py).

These exercise pure in-process logic — no network, no Gemini, no app init — so
they run fast and deterministically. Time is injected so the sliding window is
tested without sleeping.
"""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import api.dependencies as deps
from api.dependencies import AppState, enforce_rate_limit, limit_inflight
from api.ratelimit import InFlightLimiter, SlidingWindowRateLimiter


# InFlightLimiter
class TestInFlightLimiter:
    def test_disabled_always_acquires(self):
        lim = InFlightLimiter(0)
        assert not lim.enabled
        assert all(lim.try_acquire() for _ in range(100))
        assert lim.active == 0  # disabled limiter never tracks

    def test_caps_at_max(self):
        lim = InFlightLimiter(2)
        assert lim.try_acquire() is True
        assert lim.try_acquire() is True
        assert lim.active == 2
        assert lim.try_acquire() is False  # shed
        assert lim.active == 2

    def test_release_frees_a_slot(self):
        lim = InFlightLimiter(1)
        assert lim.try_acquire() is True
        assert lim.try_acquire() is False
        lim.release()
        assert lim.active == 0
        assert lim.try_acquire() is True

    def test_release_never_goes_negative(self):
        lim = InFlightLimiter(1)
        lim.release()
        lim.release()
        assert lim.active == 0

    def test_release_noop_when_disabled(self):
        lim = InFlightLimiter(0)
        lim.release()  # must not raise
        assert lim.active == 0


# SlidingWindowRateLimiter
class TestSlidingWindowRateLimiter:
    def test_disabled_always_allows(self):
        lim = SlidingWindowRateLimiter(max_requests=0, window_seconds=60)
        assert not lim.enabled
        assert all(lim.allow("u") for _ in range(50))
        assert lim.tracked_keys == 0

    def test_zero_window_is_disabled(self):
        lim = SlidingWindowRateLimiter(max_requests=5, window_seconds=0)
        assert not lim.enabled
        assert lim.allow("u") is True

    def test_caps_within_window(self):
        lim = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        assert lim.allow("u", now=1000.0) is True
        assert lim.allow("u", now=1000.1) is True
        assert lim.allow("u", now=1000.2) is True
        assert lim.allow("u", now=1000.3) is False  # 4th in window

    def test_window_slides(self):
        lim = SlidingWindowRateLimiter(max_requests=2, window_seconds=10)
        assert lim.allow("u", now=100.0) is True
        assert lim.allow("u", now=105.0) is True
        assert lim.allow("u", now=109.0) is False
        # First hit (t=100) ages out at t>110; t=111 leaves one live hit (105).
        assert lim.allow("u", now=111.0) is True

    def test_keys_are_isolated(self):
        lim = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        assert lim.allow("a", now=1.0) is True
        assert lim.allow("b", now=1.0) is True
        assert lim.allow("a", now=1.1) is False

    def test_rejected_hit_is_not_recorded(self):
        lim = SlidingWindowRateLimiter(max_requests=1, window_seconds=10)
        assert lim.allow("u", now=0.0) is True
        assert lim.allow("u", now=1.0) is False
        assert lim.allow("u", now=2.0) is False
        # Only the first hit counts; after it ages out a new hit is allowed.
        assert lim.allow("u", now=11.0) is True

    def test_gc_bounds_idle_keys(self):
        lim = SlidingWindowRateLimiter(max_requests=1, window_seconds=5, gc_threshold=10)
        # Seed many distinct keys far in the past.
        for i in range(20):
            lim.allow(f"k{i}", now=0.0)
        assert lim.tracked_keys == 20
        # A later access past the gc threshold sweeps aged-out keys.
        lim.allow("fresh", now=1000.0)
        assert lim.tracked_keys == 1


# FastAPI dependency wiring
@pytest.fixture
def fake_state(monkeypatch):
    """Install a minimal AppState carrying real limiter instances."""
    def _install(rate_limiter, inflight_limiter, window=60):
        state = AppState(
            config=SimpleNamespace(rate_limit_window_seconds=window),
            executor=SimpleNamespace(),
            rate_limiter=rate_limiter,
            inflight_limiter=inflight_limiter,
        )
        monkeypatch.setattr(deps, "_app_state", state)
        return state
    return _install


def _request(host="1.2.3.4"):
    return SimpleNamespace(client=SimpleNamespace(host=host), headers={})


class TestEnforceRateLimit:
    def test_noop_when_disabled(self, fake_state):
        fake_state(SlidingWindowRateLimiter(0, 60), InFlightLimiter(0))
        # No exception even when called many times.
        for _ in range(10):
            enforce_rate_limit(_request(), x_user_id=None)

    def test_raises_429_when_exceeded(self, fake_state):
        fake_state(SlidingWindowRateLimiter(max_requests=1, window_seconds=60), InFlightLimiter(0))
        enforce_rate_limit(_request(), x_user_id="alice")  # first allowed
        with pytest.raises(HTTPException) as exc:
            enforce_rate_limit(_request(), x_user_id="alice")
        assert exc.value.status_code == 429
        assert "Retry-After" in exc.value.headers

    def test_user_and_ip_keys_are_distinct(self, fake_state):
        fake_state(SlidingWindowRateLimiter(max_requests=1, window_seconds=60), InFlightLimiter(0))
        enforce_rate_limit(_request(host="9.9.9.9"), x_user_id="bob")
        # Different identity (anonymous IP) is not throttled by bob's budget.
        enforce_rate_limit(_request(host="9.9.9.9"), x_user_id=None)


class TestLimitInflight:
    async def test_acquires_and_releases(self, fake_state):
        inflight = InFlightLimiter(1)
        fake_state(SlidingWindowRateLimiter(0, 60), inflight)
        gen = limit_inflight()
        await gen.__anext__()              # acquire + yield
        assert inflight.active == 1
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()          # runs finally -> release
        assert inflight.active == 0

    async def test_sheds_with_503_when_full(self, fake_state):
        inflight = InFlightLimiter(1)
        fake_state(SlidingWindowRateLimiter(0, 60), inflight)
        assert inflight.try_acquire() is True  # box already full
        gen = limit_inflight()
        with pytest.raises(HTTPException) as exc:
            await gen.__anext__()
        assert exc.value.status_code == 503
        assert inflight.active == 1  # the rejected request took no slot

    async def test_noop_when_disabled(self, fake_state):
        inflight = InFlightLimiter(0)
        fake_state(SlidingWindowRateLimiter(0, 60), inflight)
        gen = limit_inflight()
        await gen.__anext__()
        assert inflight.active == 0
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()


# MYS-403 gap 3 (disconnect half): limit_inflight() releasing its slot when
# the holder is cancelled mid-stream, not just on normal completion.
#
# TestLimitInflight above (and test_api.py::TestInflightLimiterOverRealStream,
# which drives a real ASGI round-trip) both cover the generator finishing on
# its own. Neither proves the OTHER half of the dependency's own docstring
# contract: "holds a slot for the duration of the request ... releases it
# when the response completes" -- a request that never completes because the
# client walked away must still free the slot.
#
# Two tests below, at two different levels:
#
# 1. test_release_runs_when_the_holder_is_cancelled_mid_stream drives the
#    dependency generator directly (fast, isolated, no app/ASGI machinery) --
#    it proves OUR cleanup code (`finally: limiter.release()`) survives being
#    unwound via `aclose()`, i.e. that our code is not the failure mode.
#
# 2. test_release_runs_on_a_real_http_disconnect_over_raw_asgi (Codex P2,
#    MYS-403 review) proves Starlette's OWN machinery actually reaches that
#    unwind on a genuine disconnect -- test 1 alone would stay green even if
#    a real disconnect never called `aclose()` at all. httpx's ASGITransport
#    could not be used for this (confirmed empirically: an early
#    `client.stream()` context exit does not surface to the app as an ASGI
#    `http.disconnect` in this sandbox -- see the note in
#    test_api.py::TestInflightLimiterOverRealStream) -- but a *raw* ASGI
#    harness (a hand-built scope/receive/send trio calling the real FastAPI
#    app directly, bypassing httpx entirely) can and does simulate a genuine
#    `http.disconnect`, and that's what test 2 drives.
class TestLimitInflightCancellation:
    async def test_release_runs_when_the_holder_is_cancelled_mid_stream(
        self, fake_state
    ):
        inflight = InFlightLimiter(1)
        fake_state(SlidingWindowRateLimiter(0, 60), inflight)

        started = asyncio.Event()

        async def hold_the_slot():
            gen = limit_inflight()
            await gen.__anext__()  # acquire + past the yield
            started.set()
            try:
                await asyncio.sleep(60)  # stands in for "streaming forever"
            finally:
                # What Starlette does to unwind a generator dependency when
                # the request task is cancelled (client disconnect).
                await gen.aclose()

        task = asyncio.ensure_future(hold_the_slot())
        await started.wait()
        assert inflight.active == 1, "the slot must be held while the stream is open"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert inflight.active == 0, (
            "a cancelled (disconnected) request must still release its "
            "inflight slot -- otherwise capacity is wedged until a restart"
        )

    async def test_release_runs_on_a_real_http_disconnect_over_raw_asgi(self):
        """Drives the real FastAPI app via a hand-built ASGI scope/receive/
        send trio -- no httpx, no TestClient -- so the disconnect goes
        through sse_starlette's actual ``_listen_for_disconnect`` ->
        task-group-cancellation path, not a simulated ``aclose()`` call.
        This is the integration behavior the test above only assumes.
        """
        from unittest.mock import MagicMock

        from api.app import create_app
        from api.dependencies import AppState, get_gateway_user_id, verify_gateway_secret
        from core.events import Phase, ProgressEvent

        inflight = InFlightLimiter(1)
        app_state = AppState(
            config=MagicMock(),
            executor=MagicMock(),
            rate_limiter=SlidingWindowRateLimiter(0, 60),
            inflight_limiter=inflight,
        )

        started = asyncio.Event()

        async def mock_discover(**kwargs):
            yield ProgressEvent(phase=Phase.BOOK_SEARCH, step="Searching")
            started.set()
            await asyncio.sleep(60)  # stands in for "streaming forever"
            # pragma: no cover -- unreachable once disconnect cancels this

        app_state.executor.discover = mock_discover

        app = create_app()
        app.dependency_overrides[verify_gateway_secret] = lambda: None
        app.dependency_overrides[get_gateway_user_id] = lambda: "test_user"
        deps._app_state = app_state

        body = b'{"book_title": "1984", "author": "George Orwell"}'
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/itinerary/discover",
            "raw_path": b"/api/v1/itinerary/discover",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("testclient", 123),
            "server": ("testserver", 80),
        }

        request_delivered = False
        disconnect_requested = asyncio.Event()

        async def receive():
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            # Blocks until the test asks for the disconnect -- mirrors a
            # real transport's receive() parking between client messages.
            await disconnect_requested.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        app_task = asyncio.ensure_future(app(scope, receive, send))

        await asyncio.wait_for(started.wait(), timeout=5)
        for _ in range(5):
            await asyncio.sleep(0)
        assert inflight.active == 1, (
            "the slot must be held while genuinely streaming over raw ASGI"
        )

        disconnect_requested.set()
        await asyncio.wait_for(app_task, timeout=5)

        assert inflight.active == 0, (
            "a genuine ASGI http.disconnect must reach limit_inflight's own "
            "cleanup via Starlette's dependency-generator unwind -- not just "
            "a hand-simulated aclose() -- otherwise capacity is wedged until "
            "a restart"
        )
