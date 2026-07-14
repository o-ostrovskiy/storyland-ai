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


class TestFingerprintCoversTheMintLogicNotJustTheSchema:
    """MYS-460 review round 7 (Engineering Lead).

    The cached discovery payload's ``place_key`` values are MINTED by
    ``models.place_key.mint_checked_place_key`` and applied at the
    ``core.regions.enrich_region_analysis`` seam. A fingerprint that only
    hashes ``models.discovery`` (the payload's shape) is blind to a change in
    HOW a key is derived — so the next correction to the mint logic would be
    invisible on every cache hit, hardest on the popular repeated titles the
    combined readaway exists for. These modules must be load-bearing on the
    hash, proven the same way the schema module already is: by mutating the
    module's real source at import time and asserting the fingerprint moves.
    """

    def _mutated_source_version(self, module_name: str, model_name: str = "m"):
        """Compute a version where ``module_name``'s source is monkeypatched
        to a different (but still readable) string, isolating the effect of
        that one module's content on the hash."""
        import core.cache_version as cache_version_module

        original_module_source = cache_version_module._module_source

        def patched(name: str) -> str:
            if name == module_name:
                return original_module_source(name) + "\n# mutated for test\n"
            return original_module_source(name)

        cache_version_module._module_source = patched
        try:
            return compute_cache_version(
                model_name, prompt_modules=cache_version_module._PROMPT_MODULES
            )
        finally:
            cache_version_module._module_source = original_module_source

    def test_place_key_mint_logic_is_in_the_fingerprint(self):
        import core.cache_version as cache_version_module

        assert "models.place_key" in cache_version_module._PROMPT_MODULES

        baseline = compute_cache_version(
            "m", prompt_modules=cache_version_module._PROMPT_MODULES
        )
        mutated = self._mutated_source_version("models.place_key")
        assert baseline != mutated, (
            "a change to models/place_key.py (the module that MINTS every "
            "cached place_key) must invalidate the discovery cache"
        )

    def test_enrich_region_analysis_seam_is_in_the_fingerprint(self):
        import core.cache_version as cache_version_module

        assert "core.regions" in cache_version_module._PROMPT_MODULES

        baseline = compute_cache_version(
            "m", prompt_modules=cache_version_module._PROMPT_MODULES
        )
        mutated = self._mutated_source_version("core.regions")
        assert baseline != mutated, (
            "a change to core/regions.py (the seam that applies the minted "
            "place_key) must invalidate the discovery cache"
        )

    def test_default_prompt_modules_tuple_includes_both(self):
        # Guards against someone reverting to the pre-round-7 tuple.
        from core.cache_version import _PROMPT_MODULES

        assert "models.place_key" in _PROMPT_MODULES
        assert "core.regions" in _PROMPT_MODULES
