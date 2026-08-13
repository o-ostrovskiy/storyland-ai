"""Search-grounding observability in LangfusePlugin.after_model_callback.

Two placement decisions are the whole point of these tests, and both are easy
to undo by accident during a refactor:

1. The log fires BEFORE the ``self.enabled / self.client / generation`` gate.
   "Did this researcher actually search?" is a question about product
   correctness, not tracing, so it must stay answerable on a deploy with no
   Langfuse credentials — where the plugin disables itself but ADK still
   invokes the hook.
2. Search receipts reach the generation independently of token usage. A
   response with no usage_metadata is exactly when the trace is emptiest, so
   folding the search metadata into the usage branch would lose it precisely
   when it is most wanted.

Fakes follow tests/unit/test_langfuse_plugin_concurrency.py: the plugin only
ever calls .update()/.end() on an observation, so a plain object suffices and
no network or credentials are involved.
"""

from types import SimpleNamespace

import pytest

from plugins.langfuse_plugin import LangfusePlugin


class FakeObservation:
    def __init__(self):
        self.updates = []
        self.ended = False

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True


def _ctx(agent_name="city_researcher", branch="b"):
    return SimpleNamespace(
        agent_name=agent_name,
        _invocation_context=SimpleNamespace(branch=branch),
    )


def _grounded_response(usage=None):
    return SimpleNamespace(
        usage_metadata=usage,
        grounding_metadata=SimpleNamespace(
            web_search_queries=["persuasion real locations"],
            grounding_chunks=[
                SimpleNamespace(
                    web=SimpleNamespace(
                        uri="https://example.com/bath", title="Bath", domain=None
                    )
                )
            ],
        ),
    )


def _server_side_response(usage=None):
    """What an agent with include_server_side_tool_invocations returns.

    grounding_metadata is None by design here — that flag moves the receipts
    onto tool_call parts instead. The trailing function_call part is the
    set_model_response call ADK injects for tools+output_schema agents.
    """
    return SimpleNamespace(
        usage_metadata=usage,
        grounding_metadata=None,
        content=SimpleNamespace(
            parts=[
                SimpleNamespace(
                    tool_call=SimpleNamespace(
                        tool_type="ToolType.GOOGLE_SEARCH_WEB",
                        args={"queries": ["piranesi real locations"]},
                    )
                ),
                SimpleNamespace(
                    function_call=SimpleNamespace(name="set_model_response")
                ),
            ]
        ),
    )


def _usage(prompt=10, candidates=5, total=15):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=total,
    )


def _enabled_plugin():
    plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
    plugin.enabled = True
    plugin.client = object()
    return plugin


class TestLogsWithoutLangfuse:
    """The gate must not swallow the search signal."""

    async def test_captured_logged_when_plugin_disabled(self, capsys):
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
        assert plugin.enabled is False  # no credentials -> self-disabled

        await plugin.after_model_callback(
            callback_context=_ctx(), llm_response=_grounded_response()
        )

        out = capsys.readouterr().out
        assert "search_grounding_captured" in out
        assert "city_researcher" in out

    async def test_absent_logged_when_plugin_disabled(self, capsys):
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)

        await plugin.after_model_callback(
            callback_context=_ctx(agent_name="city_formatter"),
            llm_response=SimpleNamespace(usage_metadata=None, grounding_metadata=None),
        )

        out = capsys.readouterr().out
        assert "search_grounding_absent" in out
        assert "city_formatter" in out

    async def test_query_strings_are_never_logged(self, capsys):
        """Queries embed the user's book title; only counts and hosts ship."""
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)

        await plugin.after_model_callback(
            callback_context=_ctx(), llm_response=_grounded_response()
        )

        out = capsys.readouterr().out
        assert "persuasion real locations" not in out
        assert "example.com" in out  # host is fine


class TestServerSideToolCallChannel:
    """An agent with tools+output_schema reports search via parts, not metadata.

    Pinned at the plugin layer, not just the extractor: the whole value of the
    fix is that the LOG comes out right, and the object the plugin receives is
    ADK's LlmResponse — which flattens the candidate, so the parts hang off
    ``.content`` and there is no ``.candidates`` list to walk.
    """

    async def test_captured_not_absent(self, capsys):
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)

        await plugin.after_model_callback(
            callback_context=_ctx(agent_name="city_pipeline"),
            llm_response=_server_side_response(),
        )

        out = capsys.readouterr().out
        assert "search_grounding_captured" in out
        assert "search_grounding_absent" not in out
        assert "city_pipeline" in out

    async def test_query_strings_still_never_logged(self, capsys):
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)

        await plugin.after_model_callback(
            callback_context=_ctx(agent_name="city_pipeline"),
            llm_response=_server_side_response(),
        )

        assert "piranesi real locations" not in capsys.readouterr().out

    async def test_recorded_on_the_generation(self):
        plugin = _enabled_plugin()
        generation = FakeObservation()
        plugin._generations[plugin._branch_key(_ctx())] = generation

        await plugin.after_model_callback(
            callback_context=_ctx(), llm_response=_server_side_response(usage=_usage())
        )

        metadata = next(u["metadata"] for u in generation.updates if "metadata" in u)
        assert metadata["search"]["queries"] == ["piranesi real locations"]
        assert metadata["search"]["sources"] == []


class TestGenerationMetadata:
    async def test_recorded_when_token_usage_is_missing(self):
        """The emptiest trace is where the receipts matter most."""
        plugin = _enabled_plugin()
        generation = FakeObservation()
        plugin._generations[plugin._branch_key(_ctx())] = generation

        await plugin.after_model_callback(
            callback_context=_ctx(), llm_response=_grounded_response(usage=None)
        )

        search = [u["metadata"]["search"] for u in generation.updates if "metadata" in u]
        assert search and search[0]["queries"] == ["persuasion real locations"]
        assert search[0]["sources"][0]["uri"] == "https://example.com/bath"

    async def test_recorded_alongside_pricing_when_usage_present(self):
        plugin = _enabled_plugin()
        generation = FakeObservation()
        plugin._generations[plugin._branch_key(_ctx())] = generation
        plugin._models[plugin._branch_key(_ctx())] = "gemini-3.1-flash-lite"

        await plugin.after_model_callback(
            callback_context=_ctx(), llm_response=_grounded_response(usage=_usage())
        )

        metadata = next(u["metadata"] for u in generation.updates if "metadata" in u)
        assert "model_pricing" in metadata  # existing contract intact
        assert metadata["search"]["queries"] == ["persuasion real locations"]

    async def test_ungrounded_response_records_search_none(self):
        """An explicit null distinguishes 'did not search' from 'not recorded'."""
        plugin = _enabled_plugin()
        generation = FakeObservation()
        plugin._generations[plugin._branch_key(_ctx())] = generation

        await plugin.after_model_callback(
            callback_context=_ctx(agent_name="city_formatter"),
            llm_response=SimpleNamespace(usage_metadata=_usage(), grounding_metadata=None),
        )

        metadata = next(u["metadata"] for u in generation.updates if "metadata" in u)
        assert metadata["search"] is None


class TestSearchedAgentsIsPositive:
    """``searched_agents`` must be a MEASUREMENT, not the complement of a gap.

    ``unsearched_agents`` is deliberately three-valued: an agent that never ran
    is not reported. That makes ``total - len(unsearched)`` wrong as a count of
    grounded researchers — an EMPTY ledger subtracts nothing and reads as
    everything-grounded, which is the exact inversion the fail-closed guard on
    this same plugin had to fix one level down.
    """

    CANDIDATES = ("city_researcher", "author_researcher", "book_context_researcher")

    async def test_empty_ledger_grounds_nobody(self):
        """The row the old derivation got backwards. Nothing observed = 0."""
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)

        assert plugin.searched_agents(self.CANDIDATES) == frozenset()
        # and the negative half is correctly silent, which is why the
        # subtraction lied: 3 - 0 == 3.
        assert plugin.unsearched_agents(self.CANDIDATES) == frozenset()

    async def test_only_the_researcher_that_searched_is_counted(self):
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)

        await plugin.after_model_callback(
            callback_context=_ctx(agent_name="city_researcher"),
            llm_response=_grounded_response(),
        )
        await plugin.after_model_callback(
            callback_context=_ctx(agent_name="author_researcher"),
            llm_response=SimpleNamespace(usage_metadata=None, grounding_metadata=None),
        )

        assert plugin.searched_agents(self.CANDIDATES) == frozenset({"city_researcher"})
        assert plugin.unsearched_agents(self.CANDIDATES) == frozenset(
            {"author_researcher"}
        )
        # book_context_researcher was never observed: in neither set. That
        # third state is the whole point — it is not grounded and it is not
        # a skip, it is a hole in the instrumentation.

    async def test_a_later_toolless_turn_does_not_retract_a_receipt(self):
        """Same rule the ledger already holds for ``unsearched_agents``."""
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)

        await plugin.after_model_callback(
            callback_context=_ctx(agent_name="city_researcher"),
            llm_response=_grounded_response(),
        )
        await plugin.after_model_callback(
            callback_context=_ctx(agent_name="city_researcher"),
            llm_response=SimpleNamespace(usage_metadata=None, grounding_metadata=None),
        )

        assert plugin.searched_agents(self.CANDIDATES) == frozenset({"city_researcher"})
        assert plugin.unsearched_agents(self.CANDIDATES) == frozenset()

    async def test_searched_is_a_subset_of_observed(self):
        """Pins the ordering ``_log_search_grounding`` relies on.

        ``_agents_searched`` is only ever written after ``_agents_seen``. If
        that ever inverts, an agent could be "grounded" without having been
        observed and the three states would stop partitioning the candidates.
        """
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)

        await plugin.after_model_callback(
            callback_context=_ctx(agent_name="city_researcher"),
            llm_response=_server_side_response(),
        )

        assert plugin._agents_searched <= plugin._agents_seen


class TestNeverBreaksTheRequest:
    async def test_malformed_response_does_not_raise(self, capsys):
        class Exploding:
            @property
            def grounding_metadata(self):
                raise RuntimeError("shape changed")

            usage_metadata = None

        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
        result = await plugin.after_model_callback(
            callback_context=_ctx(), llm_response=Exploding()
        )
        assert result is None
        assert "search_grounding_absent" in capsys.readouterr().out

    async def test_missing_agent_name_falls_back(self, capsys):
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
        await plugin.after_model_callback(
            callback_context=SimpleNamespace(
                _invocation_context=SimpleNamespace(branch="b")
            ),
            llm_response=_grounded_response(),
        )
        assert "unknown" in capsys.readouterr().out
