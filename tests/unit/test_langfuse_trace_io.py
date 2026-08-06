"""Unit tests pinning trace-level input/output on LangfusePlugin (MYS-788).

Langfuse's session view and trace list render the TRACE's input/output, not
the root span's. The plugin set both on the root span only, so every
production trace displayed as "This trace has no input or output" even though
the span tree underneath was complete -- see the session that surfaced it:
us.cloud.langfuse.com/project/cml1naeqj01adad07nx6cwg0u/sessions/2492a1a4-26b5-45f0-bcde-afb9a76295fc

Python SDK v4 moved trace attributes to propagate_attributes(), but that
context manager carries only correlating attributes (user_id/session_id/
tags/metadata) -- set_trace_io() remains the only way to populate the two
fields those views actually read, which is why the plugin calls a method the
SDK marks deprecated.

Same fake-based approach as test_langfuse_plugin_concurrency.py: the plugin
only ever calls methods on self.client / self._current_trace, so neither a
real Langfuse client nor live credentials are needed.
"""

from types import SimpleNamespace

import pytest

from plugins.langfuse_plugin import LangfusePlugin


class FakeRootSpan:
    """Root observation that records set_trace_io/update/end separately."""

    def __init__(self):
        self.trace_io = []       # set_trace_io kwargs, in call order
        self.updates = []        # update() kwargs (root-span I/O)
        self.ended = False
        self.trace_id = "trace-abc"

    def set_trace_io(self, **kwargs):
        # Mirror the SDK: None-valued fields are dropped, so a later
        # output-only call can never clear an earlier input.
        self.trace_io.append({k: v for k, v in kwargs.items() if v is not None})
        return self

    def start_observation(self, **kwargs):
        return FakeRootSpan()

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True


class FakeClient:
    def __init__(self, root):
        self._root = root

    def start_observation(self, **kwargs):
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


class TestTraceInput:
    async def test_user_message_lands_on_the_trace_not_only_the_span(self, monkeypatch):
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

        assert root.trace_io, "trace-level input was never set -- session view renders blank"
        assert "1984" in str(root.trace_io[0]["input"])


class TestTraceOutput:
    async def test_final_response_becomes_the_trace_output(self):
        plugin, root = _make_plugin()

        await plugin.on_event_callback(
            invocation_context=_invocation_context(), event=_event("London, Airstrip One.")
        )
        await plugin.after_run_callback(invocation_context=_invocation_context())

        outputs = [io["output"] for io in root.trace_io if "output" in io]
        assert outputs == ["London, Airstrip One."]

    async def test_multipart_final_response_is_joined(self):
        plugin, root = _make_plugin()

        await plugin.on_event_callback(
            invocation_context=_invocation_context(),
            event=_event(["London, ", "Airstrip One."]),
        )
        await plugin.after_run_callback(invocation_context=_invocation_context())

        assert root.trace_io[-1]["output"] == "London, Airstrip One."

    async def test_last_final_response_wins(self):
        plugin, root = _make_plugin()

        for text in ("first pass", "second pass"):
            await plugin.on_event_callback(
                invocation_context=_invocation_context(), event=_event(text)
            )
        await plugin.after_run_callback(invocation_context=_invocation_context())

        assert root.trace_io[-1]["output"] == "second pass"

    async def test_non_final_events_are_ignored(self):
        plugin, root = _make_plugin()

        await plugin.on_event_callback(
            invocation_context=_invocation_context(),
            event=_event("intermediate chatter", final=False),
        )
        await plugin.after_run_callback(invocation_context=_invocation_context())

        # No final response -> status blob, never a blank trace.
        assert root.trace_io[-1]["output"] == {"status": "completed"}

    async def test_error_path_with_no_events_still_sets_an_output(self):
        plugin, root = _make_plugin()

        await plugin.after_run_callback(invocation_context=_invocation_context())

        assert root.trace_io[-1]["output"] == {"status": "completed"}
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

        assert root.trace_io[-1]["output"] == {"status": "completed"}

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

        assert root.trace_io == []
