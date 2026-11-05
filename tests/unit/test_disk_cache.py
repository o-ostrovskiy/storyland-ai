"""Unit tests for the persistent disk-backed Discovery cache (core/disk_cache.py)
and the backend factory (core.cache.build_discovery_cache)."""

import asyncio

import pytest

from core.cache import TTLCache, build_discovery_cache, build_place_to_book_cache
from core.disk_cache import DiskTTLCache
from core.types import ExecutorConfig


class TestDiskTTLCacheBasics:
    async def test_set_then_get_hit(self, tmp_path):
        cache = DiskTTLCache(str(tmp_path / "d"), ttl_seconds=100, max_entries=10)
        try:
            await cache.set("k", {"regions": [1, 2]})
            assert await cache.get("k") == {"regions": [1, 2]}
        finally:
            cache.close()

    async def test_missing_key_returns_none(self, tmp_path):
        cache = DiskTTLCache(str(tmp_path / "d"), ttl_seconds=100, max_entries=10)
        try:
            assert await cache.get("absent") is None
        finally:
            cache.close()

    async def test_rejects_nonpositive_config(self, tmp_path):
        with pytest.raises(ValueError):
            DiskTTLCache(str(tmp_path / "d"), ttl_seconds=0, max_entries=10)
        with pytest.raises(ValueError):
            DiskTTLCache(str(tmp_path / "d"), ttl_seconds=10, max_entries=0)

    async def test_clear_empties_cache(self, tmp_path):
        cache = DiskTTLCache(str(tmp_path / "d"), ttl_seconds=100, max_entries=10)
        try:
            await cache.set("a", 1)
            await cache.set("b", 2)
            await cache.clear()
            assert await cache.get("a") is None
            assert len(cache) == 0
        finally:
            cache.close()


class TestDiskTTLCachePersistence:
    async def test_survives_a_simulated_restart(self, tmp_path):
        """The whole point: a fresh cache object pointed at the same directory
        (a process restart / redeploy) still serves entries written before."""
        directory = str(tmp_path / "store")
        first = DiskTTLCache(directory, ttl_seconds=100, max_entries=10)
        try:
            await first.set("book|orwell", {"regions": ["London"]})
        finally:
            first.close()  # simulate process exit

        # New object, same directory == restarted process.
        second = DiskTTLCache(directory, ttl_seconds=100, max_entries=10)
        try:
            assert await second.get("book|orwell") == {"regions": ["London"]}
        finally:
            second.close()


class TestDiskTTLCacheExpiry:
    async def test_entry_expires_after_ttl(self, tmp_path):
        cache = DiskTTLCache(str(tmp_path / "d"), ttl_seconds=1, max_entries=10)
        try:
            await cache.set("k", "v")
            assert await cache.get("k") == "v"  # live
            await asyncio.sleep(1.2)  # past the 1s TTL
            assert await cache.get("k") is None  # expired
        finally:
            cache.close()


class TestBuildDiscoveryCache:
    def _cfg(self, **kw):
        base = dict(model_name="gemini-test", google_api_key="k")
        base.update(kw)
        return ExecutorConfig(**base)

    def test_memory_backend_is_ttlcache(self):
        cache = build_discovery_cache(self._cfg(cache_backend="memory"))
        assert isinstance(cache, TTLCache)

    def test_default_backend_is_memory_for_library_use(self):
        # ExecutorConfig's dataclass default keeps direct/library construction
        # (and unit tests) hermetic — no disk unless explicitly selected.
        cache = build_discovery_cache(self._cfg())
        assert isinstance(cache, TTLCache)

    def test_disk_backend_is_diskcache(self, tmp_path):
        cache = build_discovery_cache(
            self._cfg(cache_backend="disk", cache_dir=str(tmp_path / "d"))
        )
        try:
            assert isinstance(cache, DiskTTLCache)
        finally:
            cache.close()

    def test_disk_backend_without_dir_falls_back_to_memory(self):
        cache = build_discovery_cache(self._cfg(cache_backend="disk", cache_dir=None))
        assert isinstance(cache, TTLCache)

    def test_disk_backend_bad_dir_falls_back_to_memory(self):
        # A directory that can't be created (a path under a regular file) must
        # not break boot — the factory degrades to in-memory.
        cache = build_discovery_cache(
            self._cfg(cache_backend="disk", cache_dir="/dev/null/cannot/exist")
        )
        assert isinstance(cache, TTLCache)


# ai#70: place→book caches found and not-found answers for different lengths.


async def test_a_per_entry_ttl_overrides_the_default_on_disk(tmp_path):
    cache = DiskTTLCache(directory=str(tmp_path / "ttl"), ttl_seconds=3600, max_entries=10)
    await cache.set("short", "v", ttl_seconds=-1)  # already expired
    await cache.set("default", "v")
    assert await cache.get("short") is None
    assert await cache.get("default") == "v"
    cache.close()


async def test_a_per_entry_ttl_overrides_the_default_in_memory():
    now = [0.0]
    cache = TTLCache(ttl_seconds=100, max_entries=10, clock=lambda: now[0])
    await cache.set("short", "v", ttl_seconds=10)
    await cache.set("default", "v")
    now[0] = 50.0
    assert await cache.get("short") is None
    assert await cache.get("default") == "v"


def test_place_to_book_gets_its_own_store_under_cache_dir(tmp_path):
    config = ExecutorConfig(model_name="m", google_api_key="k", cache_backend="disk", cache_dir=str(tmp_path))
    cache = build_place_to_book_cache(config, ttl_seconds=60, max_entries=10)
    assert isinstance(cache, DiskTTLCache)
    assert cache.directory == str(tmp_path / "place2book")
    cache.close()


def test_place_to_book_stays_in_memory_on_the_memory_backend():
    config = ExecutorConfig(model_name="m", google_api_key="k", cache_backend="memory")
    cache = build_place_to_book_cache(config, ttl_seconds=60, max_entries=10)
    assert isinstance(cache, TTLCache)
