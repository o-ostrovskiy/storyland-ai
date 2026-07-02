"""Unit tests for the cache version fingerprint (core/cache_version.py)."""

from core.cache_version import compute_cache_version


class TestComputeCacheVersion:
    def test_deterministic_for_same_inputs(self):
        a = compute_cache_version("gemini-2.0-flash")
        b = compute_cache_version("gemini-2.0-flash")
        assert a == b

    def test_twelve_hex_chars(self):
        v = compute_cache_version("gemini-2.0-flash")
        assert len(v) == 12
        int(v, 16)  # parses as hex

    def test_model_name_changes_version(self):
        assert compute_cache_version("gemini-2.0-flash") != compute_cache_version(
            "gemini-3.1-flash-lite-preview"
        )

    def test_prompt_source_changes_version(self):
        # Two different sets of prompt modules produce different fingerprints,
        # proving the prompt source feeds the hash (a real prompt edit changes
        # the module source and therefore the version).
        base = compute_cache_version("m", prompt_modules=("core.prompts",))
        other = compute_cache_version(
            "m", prompt_modules=("core.prompts", "agents.prompts")
        )
        assert base != other

    def test_unreadable_module_degrades_gracefully(self):
        # An unknown module contributes "" instead of raising, so versioning
        # never breaks discovery — it just falls back to model-name-only keying.
        v = compute_cache_version("m", prompt_modules=("does.not.exist",))
        assert len(v) == 12
        # Same as hashing model + one empty part deterministically.
        assert v == compute_cache_version("m", prompt_modules=("does.not.exist",))

    def test_empty_model_name_is_stable(self):
        assert compute_cache_version("") == compute_cache_version("")
