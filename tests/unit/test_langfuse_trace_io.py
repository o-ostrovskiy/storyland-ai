"""Unit tests pinning the run's overall I/O on the root observation (MYS-788).

Langfuse v4 is observations-first: trace-level input/output is gone and the SDK
setter for it (`set_trace_io`) is removed in Python SDK v5. The plugin therefore
writes the user message and the final response to its root observation only.

Same fake-based approach as test_langfuse_plugin_concurrency.py: the plugin
only ever calls methods on self.client / self._current_trace, so neither a
real Langfuse client nor live credentials are needed. `FakeRootSpan` has no
`set_trace_io`, so a reintroduced call raises and the run loses its output.
"""

from types import SimpleNamespace

import pytest

from plugins.langfuse_plugin import LangfusePlugin


class FakeRootSpan:
    """Root observation that records update/end."""

    def __init__(self):
        self.updates = []        # update() kwargs (root I/O)
        self.ended = False
        self.trace_id = "trace-abc"

    def start_observation(self, **kwargs):
        return FakeRootSpan()

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True


class FakeClient:
    def __init__(self, root):
        self._root = root
        self.started = []

    def start_observation(self, **kwargs):
        self.started.append(kwargs)
        return self._root


def _make_plugin(root=None):
    plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
    plugin.enabled = True
    root = root or FakeRootSpan()
    plugin.client = FakeClient(root)
    plugin._current_trace = root
    return plugin, root


def _invocation_context():
    # agent=None mirrors every Runner(node=...) root on ADK 2.5, which is why
    # the plugin falls back to the injected root_name -- see _resolve_root_name.
    return SimpleNamespace(
        invocation_id="inv-1",
        user_id="anon:test",
        session=SimpleNamespace(id="sess-1"),
        agent=None,
    )


def _event(text, final=True):
    """Minimal ADK Event stand-in: on_event_callback reads is_final_response()
    and content.parts[*].text, nothing else."""
    parts = [SimpleNamespace(text=t) for t in ([text] if isinstance(text, str) else text)]
    return SimpleNamespace(
        is_final_response=lambda: final,
        content=SimpleNamespace(parts=parts),
    )


class TestRootInput:
    async def test_user_message_lands_on_the_root_observation(self, monkeypatch):
        plugin, root = _make_plugin()
        monkeypatch.setattr(
            "plugins.langfuse_plugin.propagate_attributes",
            lambda **kwargs: SimpleNamespace(
                __enter__=lambda *a: None, __exit__=lambda *a: None
            ),
        )
        plugin.root_name = "book_to_place_discovery"

        await plugin.on_user_message_callback(
            invocation_context=_invocation_context(), user_message="Where is 1984 set?"
        )

        assert "1984" in str(plugin.client.started[0]["input"])


class TestRootOutput:
    async def test_final_response_becomes_the_root_output(self):
        plugin, root = _make_plugin()

        await plugin.on_event_callback(
            invocation_context=_invocation_context(), event=_event("London, Airstrip One.")
        )
        await plugin.after_run_callback(invocation_context=_invocation_context())

        outputs = [u["output"] for u in root.updates if "output" in u]
        assert outputs == ["London, Airstrip One."]

    async def test_multipart_final_response_is_joined(self):
        plugin, root = _make_plugin()

        await plugin.on_event_callback(
            invocation_context=_invocation_context(),
            event=_event(["London, ", "Airstrip One."]),
        )
        await plugin.after_run_callback(invocation_context=_invocation_context())

        assert root.updates[-1]["output"] == "London, Airstrip One."

    async def test_last_final_response_wins(self):
        plugin, root = _make_plugin()

        for text in ("first pass", "second pass"):
            await plugin.on_event_callback(
                invocation_context=_invocation_context(), event=_event(text)
            )
        await plugin.after_run_callback(invocation_context=_invocation_context())

        assert root.updates[-1]["output"] == "second pass"

    async def test_non_final_events_are_ignored(self):
        plugin, root = _make_plugin()

        await plugin.on_event_callback(
            invocation_context=_invocation_context(),
            event=_event("intermediate chatter", final=False),
        )
        await plugin.after_run_callback(invocation_context=_invocation_context())

        # No final response -> status blob, never a blank root.
        assert root.updates[-1]["output"] == {"status": "completed"}

    async def test_error_path_with_no_events_still_sets_an_output(self):
        plugin, root = _make_plugin()

        await plugin.after_run_callback(invocation_context=_invocation_context())

        assert root.updates[-1]["output"] == {"status": "completed"}
        assert root.ended is True


class TestStateHygiene:
    async def test_final_response_does_not_leak_into_the_next_run(self):
        plugin, root = _make_plugin()

        await plugin.on_event_callback(
            invocation_context=_invocation_context(), event=_event("run one answer")
        )
        await plugin.after_run_callback(invocation_context=_invocation_context())

        # Second invocation on the same plugin instance produces no final
        # response: it must NOT re-report run one's answer.
        plugin._current_trace = root
        await plugin.after_run_callback(invocation_context=_invocation_context())

        assert root.updates[-1]["output"] == {"status": "completed"}

    async def test_reset_stats_clears_the_captured_response(self):
        plugin, _ = _make_plugin()

        await plugin.on_event_callback(
            invocation_context=_invocation_context(), event=_event("answer")
        )
        plugin.reset_stats()

        assert plugin._final_response_text is None

    async def test_malformed_event_does_not_raise(self):
        plugin, root = _make_plugin()

        # content=None is what ADK emits for tool-only / control events.
        bad = SimpleNamespace(is_final_response=lambda: True, content=None)
        await plugin.on_event_callback(invocation_context=_invocation_context(), event=bad)

        assert plugin._final_response_text is None

    async def test_disabled_plugin_sets_nothing(self):
        plugin, root = _make_plugin()
        plugin.enabled = False

        await plugin.on_event_callback(
            invocation_context=_invocation_context(), event=_event("answer")
        )
        await plugin.after_run_callback(invocation_context=_invocation_context())

        assert root.updates == []


class FakePropagation:
    """propagate_attributes() stand-in that records enter/exit."""

    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False


class SequencedClient:
    """Hands out a fresh root span per invocation, like the real client."""

    def __init__(self):
        self.roots = []

    def start_observation(self, **kwargs):
        root = FakeRootSpan()
        self.roots.append(root)
        return root


class TestRunErrorFinalizes:
    """ADK runs after_run_callback only when the run succeeded; on an unhandled
    error it calls on_run_error_callback instead (runners._notify_run_error).
    Without this hook the failed invocation's root span was never ended and its
    propagate_attributes context never exited -- and the composer truncation
    retry (core/executor.py compose()) reuses the same plugin, so attempt 2's
    on_user_message_callback overwrote both (PR #287, review r3)."""

    @pytest.fixture
    def plugin(self, monkeypatch):
        cms = []

        def _propagate(**kwargs):
            cm = FakePropagation()
            cms.append(cm)
            return cm

        monkeypatch.setattr("plugins.langfuse_plugin.propagate_attributes", _propagate)
        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
        plugin.enabled = True
        plugin.client = SequencedClient()
        plugin.root_name = "book_to_place_discovery"
        plugin.cms = cms
        return plugin

    async def test_retry_after_error_does_not_leak_the_first_attempt(self, plugin):
        ctx = _invocation_context()
        await plugin.on_user_message_callback(invocation_context=ctx, user_message="attempt 1")
        await plugin.on_run_error_callback(invocation_context=ctx, error=RuntimeError("boom"))
        await plugin.on_user_message_callback(invocation_context=ctx, user_message="attempt 2")

        first_root, second_root = plugin.client.roots
        first_cm, second_cm = plugin.cms
        assert first_root.ended is True
        assert first_cm.exited is True
        assert plugin._current_trace is second_root
        assert plugin._propagation_cm is second_cm
        assert second_cm.exited is False

    async def test_error_marks_the_span_without_message_text(self, plugin):
        ctx = _invocation_context()
        await plugin.on_user_message_callback(invocation_context=ctx, user_message="q")
        await plugin.on_run_error_callback(
            invocation_context=ctx, error=ValueError("user text must not leak")
        )

        (root,) = plugin.client.roots
        assert root.updates[-1]["level"] == "ERROR"
        assert root.updates[-1]["status_message"] == "ValueError"
        assert "user text" not in str(root.updates)
        assert root.updates[-1]["output"] == {"status": "error", "error_type": "ValueError"}
        assert plugin._current_trace is None
        assert plugin._propagation_cm is None
        assert plugin._final_response_text is None

    async def test_no_open_trace_is_a_no_op(self, plugin):
        await plugin.on_run_error_callback(
            invocation_context=_invocation_context(), error=RuntimeError("x")
        )

        assert plugin.client.roots == []
