"""Unit tests for LangfusePlugin's `_pricing_for` model-rate lookup.

MYS-398 follow-up: `_GEMINI_PRICING` is matched by bare substring
("key in model_name.lower()"), and "gemini-2.5-flash" is itself a
substring of "gemini-2.5-flash-lite". Checking dict keys in plain
insertion order let the shorter, wrong entry match first, so every
flash-lite call was silently priced at the plain-flash rate -- a
pre-existing bug, unrelated to MYS-398's concurrency fix, first exposed
by that fix's own test asserting two flash variants' costs side by side.
"""

from plugins.langfuse_plugin import _GEMINI_PRICING, _pricing_for


class TestPricingFor:
    def test_flash_lite_gets_its_own_rate_not_plain_flash(self):
        assert _pricing_for("gemini-2.5-flash-lite") == _GEMINI_PRICING["gemini-2.5-flash-lite"]
        # PR 4 default: the 3.1 lite tier must resolve to its own rates (and
        # must never fall through to the default-pricing tuple).
        assert _pricing_for("gemini-3.1-flash-lite") == _GEMINI_PRICING["gemini-3.1-flash-lite"]
        assert _pricing_for("gemini-2.5-flash-lite") != _GEMINI_PRICING["gemini-2.5-flash"]

    def test_plain_flash_still_gets_its_own_rate(self):
        assert _pricing_for("gemini-2.5-flash") == _GEMINI_PRICING["gemini-2.5-flash"]

    def test_every_pricing_key_matches_its_own_full_name_exactly(self):
        # Regression guard: no entry in the table should resolve to a
        # DIFFERENT entry's rate just because one key is a substring of
        # another (the general form of the flash/flash-lite bug).
        for key, expected_rate in _GEMINI_PRICING.items():
            assert _pricing_for(key) == expected_rate, f"{key!r} resolved to the wrong rate"

    def test_unknown_model_falls_back_to_default(self):
        from plugins.langfuse_plugin import _GEMINI_DEFAULT_PRICING
        assert _pricing_for("some-future-model-nobody-priced-yet") == _GEMINI_DEFAULT_PRICING

    def test_empty_model_name_falls_back_to_default(self):
        from plugins.langfuse_plugin import _GEMINI_DEFAULT_PRICING
        assert _pricing_for("") == _GEMINI_DEFAULT_PRICING
        assert _pricing_for(None) == _GEMINI_DEFAULT_PRICING
