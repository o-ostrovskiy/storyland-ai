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
        assert self._key("Dune", "Herbert", None).startswith("discover:v2:")

    def test_absent_vibe_key_is_unchanged(self):
        # An absent vibe must produce the exact pre-vibe key (cache continuity).
        from core.executor import WorkflowExecutor

        with_default = WorkflowExecutor._discovery_cache_key("Dune", "Herbert", None)
        explicit_none = WorkflowExecutor._discovery_cache_key(
            "Dune", "Herbert", None, None
        )
        assert with_default == explicit_none == "discover:v2:dune|herbert|" + (
            with_default.split("|")[-1]
        )

    def test_present_vibe_changes_key(self):
        from core.executor import WorkflowExecutor

        base = WorkflowExecutor._discovery_cache_key("Dune", "Herbert", None)
        cozy = WorkflowExecutor._discovery_cache_key("Dune", "Herbert", None, "cozy")
        assert cozy != base
        assert cozy.endswith("|vibe=cozy")

    def test_absent_taste_key_is_unchanged(self):
        from core.executor import WorkflowExecutor

        base = WorkflowExecutor._discovery_cache_key("Dune", "Herbert", None)
        with_none = WorkflowExecutor._discovery_cache_key(
            "Dune", "Herbert", None, None, None
        )
        assert base == with_none

    def test_present_taste_changes_key(self):
        from core.executor import WorkflowExecutor

        base = WorkflowExecutor._discovery_cache_key("Dune", "Herbert", None)
        tasted = WorkflowExecutor._discovery_cache_key(
            "Dune", "Herbert", None, None, {"titles": ["1984"], "moods": ["bleak"]}
        )
        assert tasted != base
        assert "|taste=" in tasted

    def test_different_taste_differ(self):
        from core.executor import WorkflowExecutor

        k1 = WorkflowExecutor._discovery_cache_key(
            "Dune", "Herbert", None, None, {"titles": ["1984"]}
        )
        k2 = WorkflowExecutor._discovery_cache_key(
            "Dune", "Herbert", None, None, {"titles": ["Beloved"]}
        )
        assert k1 != k2

    def test_vibe_and_taste_compose_in_key(self):
        from core.executor import WorkflowExecutor

        key = WorkflowExecutor._discovery_cache_key(
            "Dune", "Herbert", None, "cozy", {"moods": ["warm"]}
        )
        assert "|vibe=cozy" in key
        assert "|taste=" in key

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
        from core.executor import WorkflowExecutor
        from core.events import RegionsReady, WorkflowComplete
        from services.session_service import create_session_service

        # Any attempt to build/run the Gemini discovery chain must fail the test.
        def _boom(*args, **kwargs):
            raise AssertionError("discovery chain invoked on a cache hit")

        monkeypatch.setattr(ex, "create_book_to_place_discovery_workflow", _boom)
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

        # MYS-460: a cached entry carries the grounded geo fields, and the replay
        # must mint the place_key from them. This is the AC stated as behaviour —
        # a HIT can never hand the wire a region with no cross-job identity.
        # v2 bundle shape: region_analysis + the grounding payloads compose()
        # reads from replayed state (ADR #24).
        region_analysis = {
            "regions": [
                {
                    "region_id": 1,
                    "name": "Atlanta, United States",
                    "country_code": "US",
                    "primary_locality": "Atlanta",
                    # MYS-460 fix-list #3: primary_locality is self-consistency
                    # checked against the region's own `cities` before minting.
                    "cities": [{"name": "Atlanta", "country": "United States"}],
                }
            ],
            "analysis_note": "cached note",
        }
        cached = {
            "region_analysis": region_analysis,
            "book_context": {"themes": ["southern gothic"]},
        }
        # The executor namespaces keys with the model/prompt version, so prime
        # the store through the same helper discover() uses.
        key = executor._versioned_key(
            WorkflowExecutor._discovery_cache_key("1984", "George Orwell", None)
        )
        await executor._discovery_cache.set(key, cached)

        events = [
            e
            async for e in executor.discover(
                book_title="1984", author="George Orwell"
            )
        ]

        regions_events = [e for e in events if isinstance(e, RegionsReady)]
        assert len(regions_events) == 1
        replayed = regions_events[0].regions
        # The cached payload is relayed intact...
        assert replayed == [{**region_analysis["regions"][0], "place_key": "us:atlanta"}]
        # ...and the identity is on it. Cache hits land hardest on popular,
        # repeated titles — exactly the book-club case the combined readaway is
        # for — so a keyless replay would kill the feature on precisely the books
        # it was built for.
        assert all(r["place_key"] for r in replayed)
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


# Cache master-switch config (MYS-153 / MYS-9 PR1)
#
# The Discovery result cache must NEVER be silently disabled by a dropped env
# var: an absent CACHE_ENABLED falls back to on. These tests pin that contract
# and the flag's propagation Config -> ExecutorConfig -> WorkflowExecutor.

import os
from unittest.mock import MagicMock

from common.config import _env_bool, load_config


def _base_env(monkeypatch) -> None:
    """Set the minimal REQUIRED env so load_config() succeeds; leave the
    optional cache vars to each test."""
    required = {
        "GOOGLE_API_KEY": "test-key",
        "USE_DATABASE": "false",
        "SESSION_MAX_EVENTS": "20",
        "MAX_CONTEXT_TOKENS": "30000",
        "MODEL_NAME": "gemini-test",
        "WORKFLOW_TIMEOUT": "600",
        "AGENT_TIMEOUT": "120",
        "LOG_LEVEL": "INFO",
        "ENABLE_ADK_DEBUG": "false",
    }
    for k, v in required.items():
        monkeypatch.setenv(k, v)
    # Ensure the optional switch starts unset for the fallback tests.
    monkeypatch.delenv("CACHE_ENABLED", raising=False)


class TestCacheEnabledFallback:
    def test_env_bool_missing_defaults_true(self, monkeypatch):
        monkeypatch.delenv("CACHE_ENABLED", raising=False)
        assert _env_bool("CACHE_ENABLED", True) is True

    def test_env_bool_false_is_false(self, monkeypatch):
        monkeypatch.setenv("CACHE_ENABLED", "false")
        assert _env_bool("CACHE_ENABLED", True) is False

    def test_env_bool_true_is_true(self, monkeypatch):
        monkeypatch.setenv("CACHE_ENABLED", "true")
        assert _env_bool("CACHE_ENABLED", True) is True

    def test_load_config_missing_var_stays_on(self, monkeypatch):
        """A dropped CACHE_ENABLED must degrade to ON, never silently off."""
        _base_env(monkeypatch)
        assert load_config().cache_enabled is True

    def test_load_config_explicit_false_disables(self, monkeypatch):
        _base_env(monkeypatch)
        monkeypatch.setenv("CACHE_ENABLED", "false")
        assert load_config().cache_enabled is False


class TestExecutorConfigCacheEnabled:
    def test_default_is_enabled(self):
        cfg = ExecutorConfig(model_name="gemini-test", google_api_key="k")
        assert cfg.cache_enabled is True

    def test_from_config_carries_flag(self):
        mock_config = MagicMock()
        mock_config.cache_enabled = False
        assert ExecutorConfig.from_config(mock_config).cache_enabled is False

    def test_from_config_defaults_true_when_absent(self):
        # A source config object that predates the flag (no cache_enabled
        # attribute) must resolve to the getattr default of True.
        from types import SimpleNamespace

        legacy = SimpleNamespace(
            model_name="gemini-test",
            google_api_key="k",
            workflow_timeout=600,
            database_url=None,
            use_database=False,
            langfuse_secret_key=None,
            langfuse_public_key=None,
            langfuse_host=None,
            environment="local",
        )
        assert ExecutorConfig.from_config(legacy).cache_enabled is True


class TestExecutorCacheGating:
    """The executor must honor cache_enabled without touching Gemini."""

    def _executor(self, enabled: bool):
        from core.executor import WorkflowExecutor

        cfg = ExecutorConfig(
            model_name="gemini-test",
            google_api_key="k",
            cache_enabled=enabled,
            use_database=False,
        )
        # Inject a fake model so no real Gemini client is created.
        return WorkflowExecutor(cfg, model=MagicMock())

    def test_flag_propagates_enabled(self):
        assert self._executor(True)._cache_enabled is True

    def test_flag_propagates_disabled(self):
        assert self._executor(False)._cache_enabled is False
