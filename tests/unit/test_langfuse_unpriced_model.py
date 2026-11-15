"""An unpriced model is costed as unknown end to end (ai-private#37).

Before, a model with no `_GEMINI_PRICING` row took the 2.5-flash rate, and
the generation, trace total and session stats all showed that borrowed
number as if it were measured. These rows assert the opposite at each
place a cost leaves the plugin: no cost on the generation, a warning per
call, and a `cost_unknown` flag on the session stats. Fakes follow
tests/unit/test_langfuse_search_grounding.py.
"""

from types import SimpleNamespace

from plugins.langfuse_plugin import LangfusePlugin


class FakeObservation:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        pass


def _ctx():
    return SimpleNamespace(
        agent_name="city_researcher",
        _invocation_context=SimpleNamespace(branch="b"),
    )


def _response():
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=1_000_000,
            candidates_token_count=1_000_000,
            total_token_count=2_000_000,
        ),
        grounding_metadata=None,
    )


def _plugin_with_model(model_name):
    plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
    plugin.enabled = True
    plugin.client = object()
    generation = FakeObservation()
    key = plugin._branch_key(_ctx())
    plugin._generations[key] = generation
    plugin._models[key] = model_name
    return plugin, generation


class TestUnpricedModel:
    async def test_generation_carries_no_cost_and_flags_unknown(self, capsys):
        plugin, generation = _plugin_with_model("gemini-9-made-up")

        await plugin.after_model_callback(callback_context=_ctx(), llm_response=_response())

        update = next(u for u in generation.updates if "usage_details" in u)
        assert "cost_details" not in update
        assert update["metadata"]["model_pricing"] is None
        assert update["metadata"]["cost_unknown"] is True
        assert update["usage_details"]["input"] == 1_000_000  # tokens still recorded

        out = capsys.readouterr().out
        assert "langfuse_model_unpriced" in out
        assert "gemini-9-made-up" in out

    async def test_session_stats_mark_cost_unknown_not_a_borrowed_rate(self):
        plugin, _ = _plugin_with_model("gemini-9-made-up")

        await plugin.after_model_callback(callback_context=_ctx(), llm_response=_response())

        stats = plugin.get_session_stats()
        assert stats["cost_usd"] == 0.0  # nothing priced, nothing borrowed
        assert stats["cost_unknown"] is True

    async def test_priced_model_keeps_its_cost_and_is_not_flagged(self):
        plugin, generation = _plugin_with_model("gemini-3.1-flash-lite-preview")

        await plugin.after_model_callback(callback_context=_ctx(), llm_response=_response())

        update = next(u for u in generation.updates if "usage_details" in u)
        assert update["cost_details"]["total_cost"] == 0.25 + 1.50
        assert update["metadata"]["cost_unknown"] is False
        assert plugin.get_session_stats()["cost_unknown"] is False

    async def test_reset_clears_the_unknown_flag(self):
        plugin, _ = _plugin_with_model("gemini-9-made-up")
        await plugin.after_model_callback(callback_context=_ctx(), llm_response=_response())

        plugin.reset_stats()

        assert plugin.get_session_stats()["cost_unknown"] is False
