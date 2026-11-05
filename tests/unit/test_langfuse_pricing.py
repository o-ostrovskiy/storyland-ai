"""Unit tests for LangfusePlugin's `_pricing_for` model-rate lookup.

MYS-398 follow-up: `_GEMINI_PRICING` is matched by bare substring
("key in model_name.lower()"), and "gemini-2.5-flash" is itself a
substring of "gemini-2.5-flash-lite". Checking dict keys in plain
insertion order let the shorter, wrong entry match first, so every
flash-lite call was silently priced at the plain-flash rate -- a
pre-existing bug, unrelated to MYS-398's concurrency fix, first exposed
by that fix's own test asserting two flash variants' costs side by side.

ai-private#37: an unpriced model used to fall back to the 2.5-flash rate,
and a confidently-wrong number renders identically to a right one. It now
prices as unknown (None), and every model string the service can be
configured with must have its own row.
"""

import re
from pathlib import Path

from plugins.langfuse_plugin import _GEMINI_PRICING, _calculate_cost, _pricing_for


class TestPricingFor:
    def test_flash_lite_gets_its_own_rate_not_plain_flash(self):
        assert _pricing_for("gemini-2.5-flash-lite") == _GEMINI_PRICING["gemini-2.5-flash-lite"]
        # PR 4 default: the 3.1 lite tier must resolve to its own rates (and
        # must never fall through to an unpriced result).
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

    def test_unknown_model_is_unpriced_not_another_models_rate(self):
        assert _pricing_for("some-future-model-nobody-priced-yet") is None
        assert _calculate_cost(1_000, 1_000, "some-future-model-nobody-priced-yet") is None

    def test_empty_model_name_is_unpriced(self):
        assert _pricing_for("") is None
        assert _pricing_for(None) is None

    def test_production_preview_id_prices_at_its_stable_rate(self):
        # Live /health on 2026-10-22 reported this exact string. Google's
        # pricing page lists only the stable id, so the preview takes its rate.
        assert _pricing_for("gemini-3.1-flash-lite-preview") == (0.25, 1.50)


class TestEveryConfigurableModelIsPriced:
    """The class guard: the next model bump reds CI instead of mispricing."""

    def _configured_models(self) -> set[str]:
        from common.config import DEFAULT_MODEL_NAME

        models = {DEFAULT_MODEL_NAME, DEFAULT_MODEL_NAME + "-preview"}
        sources = [Path(".env.example"), *sorted(Path(".github/workflows").glob("*.yml"))]
        for path in sources:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                match = re.search(r"MODEL_NAME=([A-Za-z0-9._-]+)", stripped)
                if match:
                    models.add(match.group(1))
        return models

    def test_every_configured_model_has_a_pricing_row(self):
        models = self._configured_models()
        # Vacuity control: the default and its alias, plus at least one injection.
        assert len(models) >= 2, models
        unpriced = sorted(m for m in models if _pricing_for(m) is None)
        assert not unpriced, f"no pricing row for configured model(s): {unpriced}"
