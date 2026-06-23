"""Unit tests for the load-shedding guards (api/ratelimit.py) and the FastAPI
dependencies that wire them in (api/dependencies.py).

These exercise pure in-process logic — no network, no Gemini, no app init — so
they run fast and deterministically. Time is injected so the sliding window is
tested without sleeping.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import api.dependencies as deps
from api.dependencies import AppState, enforce_rate_limit, limit_inflight
from api.ratelimit import InFlightLimiter, SlidingWindowRateLimiter


# ---------------------------------------------------------------------------
# InFlightLimiter
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# SlidingWindowRateLimiter
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# FastAPI dependency wiring
# ---------------------------------------------------------------------------
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
