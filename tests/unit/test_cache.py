"""Unit tests for the in-process TTL + LRU cache (core/cache.py)."""

import pytest

from core.cache import TTLCache
from core.types import ExecutorConfig


class _FakeClock:
    """Controllable monotonic clock for deterministic TTL tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestTTLCacheBasics:
    async def test_set_then_get_hit(self):
        cache = TTLCache(ttl_seconds=100, max_entries=10)
        await cache.set("k", {"regions": [1, 2]})
        assert await cache.get("k") == {"regions": [1, 2]}

    async def test_missing_key_returns_none(self):
        cache = TTLCache(ttl_seconds=100, max_entries=10)
        assert await cache.get("absent") is None

    async def test_rejects_nonpositive_config(self):
        with pytest.raises(ValueError):
            TTLCache(ttl_seconds=0, max_entries=10)
        with pytest.raises(ValueError):
            TTLCache(ttl_seconds=10, max_entries=0)


class TestTTLCacheExpiry:
    async def test_entry_expires_after_ttl(self):
        clock = _FakeClock()
        cache = TTLCache(ttl_seconds=60, max_entries=10, clock=clock)
        await cache.set("k", "v")
        clock.advance(59)
        assert await cache.get("k") == "v"  # still live
        clock.advance(1)  # now at ttl boundary
        assert await cache.get("k") is None  # expired
        assert len(cache) == 0  # expired entry dropped on access


class TestTTLCacheEviction:
    async def test_oldest_evicted_when_over_capacity(self):
        cache = TTLCache(ttl_seconds=100, max_entries=2)
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)  # should evict "a"
        assert await cache.get("a") is None
        assert await cache.get("b") == 2
        assert await cache.get("c") == 3
        assert len(cache) == 2

    async def test_get_refreshes_recency(self):
        cache = TTLCache(ttl_seconds=100, max_entries=2)
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.get("a")  # refresh "a" so "b" is now oldest
        await cache.set("c", 3)  # should evict "b", not "a"
        assert await cache.get("a") == 1
        assert await cache.get("b") is None
        assert await cache.get("c") == 3

    async def test_clear_empties_cache(self):
        cache = TTLCache(ttl_seconds=100, max_entries=10)
        await cache.set("a", 1)
        await cache.clear()
        assert await cache.get("a") is None
        assert len(cache) == 0


class TestDiscoveryCacheKey:
    """The key builder lives on WorkflowExecutor; verify normalization."""

    def _key(self, title, author, prefs):
        from core.executor import WorkflowExecutor

        return WorkflowExecutor._discovery_cache_key(title, author, prefs)

    def test_normalizes_case_and_whitespace(self):
        assert self._key("  1984 ", "George ORWELL", None) == self._key(
            "1984", "george orwell", None
        )

    def test_preferences_order_independent(self):
        k1 = self._key("Dune", "Herbert", {"pace": "slow", "mood": "epic"})
        k2 = self._key("Dune", "Herbert", {"mood": "epic", "pace": "slow"})
        assert k1 == k2

    def test_different_preferences_differ(self):
        k1 = self._key("Dune", "Herbert", {"mood": "epic"})
        k2 = self._key("Dune", "Herbert", {"mood": "cozy"})
        assert k1 != k2

    def test_versioned_prefix(self):
        assert self._key("Dune", "Herbert", None).startswith("discover:v1:")

    def test_absent_vibe_key_is_unchanged(self):
        # An absent vibe must produce the exact pre-vibe key (cache continuity).
        from core.executor import WorkflowExecutor

        with_default = WorkflowExecutor._discovery_cache_key("Dune", "Herbert", None)
        explicit_none = WorkflowExecutor._discovery_cache_key(
            "Dune", "Herbert", None, None
        )
        assert with_default == explicit_none == "discover:v1:dune|herbert|" + (
            with_default.split("|")[-1]
        )

    def test_present_vibe_changes_key(self):
        from core.executor import WorkflowExecutor

        base = WorkflowExecutor._discovery_cache_key("Dune", "Herbert", None)
        cozy = WorkflowExecutor._discovery_cache_key("Dune", "Herbert", None, "cozy")
        assert cozy != base
        assert cozy.endswith("|vibe=cozy")

    def test_different_vibes_differ(self):
        from core.executor import WorkflowExecutor

        k1 = WorkflowExecutor._discovery_cache_key("Dune", "Herbert", None, "cozy")
        k2 = WorkflowExecutor._discovery_cache_key(
            "Dune", "Herbert", None, "hopeful"
        )
        assert k1 != k2


class TestExecutorConfigCacheDefaults:
    def test_cache_config_defaults(self):
        config = ExecutorConfig(
            model_name="gemini-2.0-flash", google_api_key="test-key"
        )
        assert config.cache_ttl_seconds == 86400
        assert config.cache_max_entries == 500


class TestExecutorCacheHit:
    """End-to-end: a primed cache short-circuits discover() with zero LLM work."""

    async def test_cache_hit_skips_discovery_chain(self, monkeypatch):
        import core.executor as ex
        from core.executor import WorkflowExecutor, APP_NAME
        from core.events import RegionsReady, WorkflowComplete
        from services.session_service import create_session_service

        # Any attempt to build/run the Gemini discovery chain must fail the test.
        def _boom(*args, **kwargs):
            raise AssertionError("discovery chain invoked on a cache hit")

        monkeypatch.setattr(ex, "create_discovery_workflow", _boom)
        monkeypatch.setattr(ex, "Runner", _boom)

        config = ExecutorConfig(
            model_name="gemini-2.0-flash",
            google_api_key="test-key",
        )
        executor = WorkflowExecutor(
            config=config,
            session_service=create_session_service(use_database=False),
            model=object(),  # never used on a hit; bypass real Gemini construction
        )

        cached = {
            "regions": [{"region_id": 1, "name": "Atlanta, United States"}],
            "analysis_note": "cached note",
        }
        key = WorkflowExecutor._discovery_cache_key("1984", "George Orwell", None)
        await executor._discovery_cache.set(key, cached)

        events = [
            e
            async for e in executor.discover(
                book_title="1984", author="George Orwell"
            )
        ]

        regions_events = [e for e in events if isinstance(e, RegionsReady)]
        assert len(regions_events) == 1
        assert regions_events[0].regions == cached["regions"]
        assert regions_events[0].analysis_note == "cached note"
        assert any(isinstance(e, WorkflowComplete) for e in events)

    async def test_cache_always_enabled(self):
        from core.cache import TTLCache
        from core.executor import WorkflowExecutor
        from services.session_service import create_session_service

        config = ExecutorConfig(
            model_name="gemini-2.0-flash", google_api_key="test-key"
        )
        executor = WorkflowExecutor(
            config=config,
            session_service=create_session_service(use_database=False),
            model=object(),
        )
        assert isinstance(executor._discovery_cache, TTLCache)
