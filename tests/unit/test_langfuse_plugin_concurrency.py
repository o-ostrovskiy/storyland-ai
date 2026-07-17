"""Unit tests for LangfusePlugin's per-branch state isolation (MYS-398).

core/executor.py creates ONE LangfusePlugin instance per workflow run, and
agents/orchestrator.py's discovery_workflow runs three sub-agent pipelines
(city/landmark/author) concurrently under a single ParallelAgent
(parallel_discovery). Before this fix, before_model_callback/
after_model_callback/before_agent_callback/after_agent_callback tracked their
"current" generation, model name, and agent stack in shared scalars/a single
LIFO list on the plugin instance -- so an interleaved callback pair from one
branch could clobber state that belonged to a different, still-open branch.

These are true unit tests of the plugin's internal state machine: they build
minimal fakes for the pieces the plugin actually touches (a
callback_context whose only relevant attribute is
._invocation_context.branch -- the same field ADK's own ParallelAgent uses
to isolate concurrent sub-agent branches, see
google.adk.agents.parallel_agent._create_branch_ctx_for_sub_agent -- plus
fake llm_request/llm_response/agent objects and a fake Langfuse
"observation") rather than constructing real ADK Runner/agent graphs, which
need live Google credentials -- that end-to-end path is already covered by
tests/integration/test_langfuse_integration.py.
"""

from types import SimpleNamespace

import pytest

from plugins.langfuse_plugin import LangfusePlugin


class FakeObservation:
    """Stands in for a Langfuse generation/span observation."""

    def __init__(self, name):
        self.name = name
        self.ended = False
        self.updates = []

    def start_observation(self, **kwargs):
        return FakeObservation(kwargs.get("name"))

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True


def _make_plugin() -> LangfusePlugin:
    plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
    # Force-enable without a real Langfuse client: every code path this
    # plugin exercises only calls methods on self.client / self._current_trace,
    # both of which are fakeable directly without touching the network.
    plugin.enabled = True
    plugin.client = object()
    plugin._current_trace = FakeObservation("root")
    return plugin


def _ctx(branch):
    """Minimal stand-in for CallbackContext: LangfusePlugin._branch_key only
    reads ._invocation_context.branch off it."""
    return SimpleNamespace(_invocation_context=SimpleNamespace(branch=branch))


class TestBranchKey:
    def test_distinct_branches_get_distinct_keys(self):
        plugin = _make_plugin()
        a = plugin._branch_key(_ctx("parallel_discovery.city_pipeline"))
        b = plugin._branch_key(_ctx("parallel_discovery.landmark_pipeline"))
        assert a != b

    def test_missing_branch_falls_back_to_shared_root_key(self):
        plugin = _make_plugin()
        assert plugin._branch_key(_ctx(None)) == plugin._branch_key(_ctx(None)) == "_root"

    def test_missing_invocation_context_does_not_raise(self):
        plugin = _make_plugin()
        # A context object with no _invocation_context at all -- defensive
        # coverage in case a future ADK version renames the attribute.
        assert plugin._branch_key(SimpleNamespace()) == "_root"


class TestConcurrentBranchesDoNotCorruptEachOther:
    @pytest.mark.asyncio
    async def test_interleaved_model_calls_both_end_correctly(self):
        """The exact race MYS-398 describes: branch A's before_model, then
        branch B's before_model (which used to overwrite A's still-open
        generation via the old shared _current_generation scalar), then A's
        after_model, then B's after_model. Both generations must end, and
        neither must be costed using the other's model's pricing.
        """
        plugin = _make_plugin()
        ctx_a = _ctx("parallel_discovery.city_pipeline")
        ctx_b = _ctx("parallel_discovery.landmark_pipeline")

        req_a = SimpleNamespace(model="gemini-2.5-flash-lite")
        req_b = SimpleNamespace(model="gemini-2.5-flash")

        await plugin.before_model_callback(callback_context=ctx_a, llm_request=req_a)
        await plugin.before_model_callback(callback_context=ctx_b, llm_request=req_b)

        key_a = plugin._branch_key(ctx_a)
        key_b = plugin._branch_key(ctx_b)
        gen_a = plugin._generations[key_a]
        gen_b = plugin._generations[key_b]
        # Both branches have their own open generation -- the old shared
        # scalar could only ever hold one at a time.
        assert gen_a is not gen_b
        assert plugin._models[key_a] == "gemini-2.5-flash-lite"
        assert plugin._models[key_b] == "gemini-2.5-flash"

        resp_a = SimpleNamespace(usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20, total_token_count=120))
        resp_b = SimpleNamespace(usage_metadata=SimpleNamespace(
            prompt_token_count=50, candidates_token_count=10, total_token_count=60))

        await plugin.after_model_callback(callback_context=ctx_a, llm_response=resp_a)
        await plugin.after_model_callback(callback_context=ctx_b, llm_response=resp_b)

        assert gen_a.ended is True
        assert gen_b.ended is True
        # Neither branch's state lingers after it closes.
        assert key_a not in plugin._generations
        assert key_b not in plugin._generations

        # This is the actual bug made concrete: before the fix, gen_a would
        # have been costed using gemini-2.5-flash's rate (B's model),
        # because _current_model was a single shared scalar B had already
        # overwritten by the time A's after_model_callback ran.
        expected_cost_a = (100 / 1_000_000) * 0.10 + (20 / 1_000_000) * 0.40  # flash-lite rate
        expected_cost_b = (50 / 1_000_000) * 0.30 + (10 / 1_000_000) * 2.50  # flash rate
        assert gen_a.updates[0]["cost_details"]["total_cost"] == pytest.approx(expected_cost_a)
        assert gen_b.updates[0]["cost_details"]["total_cost"] == pytest.approx(expected_cost_b)

        # Session totals aggregate across both branches -- that's intended,
        # they're a whole-run total, not per-branch.
        assert plugin._token_usage.input_tokens == 150
        assert plugin._token_usage.output_tokens == 30

    @pytest.mark.asyncio
    async def test_agent_stack_pop_is_scoped_per_branch(self):
        """Regression for the LIFO half of the bug: two branches' agent
        start/end calls used to push/pop a single shared list, so a branch
        could end up ending the WRONG agent's span whenever completion order
        didn't match push order -- which asyncio gives no guarantee of.
        """
        plugin = _make_plugin()
        ctx_a = _ctx("parallel_discovery.city_pipeline")
        ctx_b = _ctx("parallel_discovery.landmark_pipeline")

        agent_a = SimpleNamespace(name="city_research_agent")
        agent_b = SimpleNamespace(name="landmark_research_agent")

        await plugin.before_agent_callback(agent=agent_a, callback_context=ctx_a)
        await plugin.before_agent_callback(agent=agent_b, callback_context=ctx_b)

        span_a = plugin._agent_stacks[plugin._branch_key(ctx_a)][-1][1]
        span_b = plugin._agent_stacks[plugin._branch_key(ctx_b)][-1][1]
        assert span_a is not span_b

        # B finishes first -- completion order need not match push order
        # under real concurrency.
        await plugin.after_agent_callback(agent=agent_b, callback_context=ctx_b)
        assert span_b.ended is True
        assert span_a.ended is False  # would have been True pre-fix (wrong pop)

        await plugin.after_agent_callback(agent=agent_a, callback_context=ctx_a)
        assert span_a.ended is True

        assert plugin._agent_stacks == {}

    @pytest.mark.asyncio
    async def test_sequential_only_path_shares_one_key_and_still_works(self):
        """No ParallelAgent in play (branch is None throughout -- e.g. the
        single-agent local_atmosphere/trip_composer workflows) must keep
        behaving exactly like the pre-fix single-scalar version, since
        there's never more than one in-flight generation there.
        """
        plugin = _make_plugin()
        ctx = _ctx(None)
        req = SimpleNamespace(model="gemini-2.5-flash")

        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        assert plugin._branch_key(ctx) == "_root"
        assert "_root" in plugin._generations

        resp = SimpleNamespace(usage_metadata=SimpleNamespace(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15))
        await plugin.after_model_callback(callback_context=ctx, llm_response=resp)
        assert "_root" not in plugin._generations
