"""Regression tests for LangfusePlugin._extract_token_usage (MYS: Codex finding on ADK 2 lift).

The defect: LlmResponse is pydantic, so ``usage_metadata`` is a *declared
field* and ``hasattr(response, 'usage_metadata')`` is always true. When the
value was None, ``getattr(None, 'prompt_token_count', 0)`` yielded 0 for
every count, producing a truthy ``TokenUsage(0, 0, 0)`` — the success branch
recorded a zero-token, zero-cost generation and the
``langfuse_token_usage_missing`` warning (added to catch exactly that) never
fired. These tests pin the fixed contract: absent, None, or count-less usage
is ``None`` (never a zero-token success), and the warning path owns it.
"""

from types import SimpleNamespace

from plugins.langfuse_plugin import LangfusePlugin


def _plugin() -> LangfusePlugin:
    # No credentials: plugin disables itself, but _extract_token_usage is
    # pure and independent of the enabled flag.
    return LangfusePlugin(secret_key=None, public_key=None, host=None)


def _meta(prompt=None, candidates=None, total=None):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=total,
    )


class TestExtractTokenUsage:
    def test_none_usage_metadata_returns_none(self):
        """The declared-field trap: metadata present as a field, value None."""
        response = SimpleNamespace(usage_metadata=None)
        assert _plugin()._extract_token_usage(response) is None

    def test_metadata_with_all_none_counts_returns_none(self):
        response = SimpleNamespace(usage_metadata=_meta())
        assert _plugin()._extract_token_usage(response) is None

    def test_metadata_with_all_zero_counts_returns_none(self):
        """A real generation can never be 0/0/0 — normalize to missing."""
        response = SimpleNamespace(usage_metadata=_meta(0, 0, 0))
        assert _plugin()._extract_token_usage(response) is None

    def test_real_counts_extracted(self):
        response = SimpleNamespace(usage_metadata=_meta(288, 555, 843))
        usage = _plugin()._extract_token_usage(response)
        assert usage is not None
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
            288,
            555,
            843,
        )

    def test_partial_counts_still_extracted(self):
        """Some SDK shapes omit total; any non-zero count is real usage."""
        response = SimpleNamespace(usage_metadata=_meta(prompt=10))
        usage = _plugin()._extract_token_usage(response)
        assert usage is not None
        assert usage.input_tokens == 10
        assert usage.output_tokens == 0

    def test_dict_shape_extracted(self):
        response = {
            "usage_metadata": {
                "prompt_token_count": 7,
                "candidates_token_count": 3,
                "total_token_count": 10,
            }
        }
        assert _plugin()._extract_token_usage(response).total_tokens == 10

    def test_dict_shape_with_none_metadata_returns_none(self):
        assert _plugin()._extract_token_usage({"usage_metadata": None}) is None

    def test_object_without_field_returns_none(self):
        assert _plugin()._extract_token_usage(object()) is None
