"""MCP request source -> Langfuse tags/metadata (storyland-services#266).

The gateway's MCP server sends X-Request-Source/X-MCP-* on discover and
compose; the job keeps them in session state and the Langfuse plugin turns them
into trace tags and metadata. A website call sends none and its trace is
unchanged.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Depends
from httpx import ASGITransport, AsyncClient

from api.dependencies import get_request_source
from core.events import JobStarted, ProgressEvent
from common.request_source import RequestSource
from core.session_state import SessionStateKeys
from core.types import ExecutorConfig
from plugins.langfuse_plugin import LangfusePlugin

MCP_HEADERS = {
    "X-Request-Source": "mcp",
    "X-MCP-Tool": "plan_trip",
    "X-MCP-Client": "claude-ai",
    "X-MCP-Client-Version": "0.1.0",
    "X-MCP-Session": "22522e04",
}
SOURCE = RequestSource.mcp("plan_trip", "claude-ai", "0.1.0", "22522e04")


# --- the model -----------------------------------------------------------------


class TestRequestSource:
    def test_tags_and_metadata(self):
        assert SOURCE.langfuse_tags() == ["mcp", "tool:plan_trip", "client:claude-ai"]
        assert SOURCE.langfuse_metadata() == {
            "source": "mcp",
            "mcp_tool": "plan_trip",
            "mcp_client": "claude-ai",
            "mcp_client_version": "0.1.0",
            "mcp_session": "22522e04",
        }

    def test_missing_values_read_unknown_and_none(self):
        source = RequestSource.mcp(None, "", "  ", None)
        assert source.langfuse_tags() == ["mcp", "tool:unknown", "client:unknown"]
        assert source.session == "none"

    @pytest.mark.parametrize(
        ("raw", "clean"),
        [("Claude Desktop", "Claude-Desktop"), ("Клод", "unknown"), ("x" * 300, "x" * 64)],
    )
    def test_values_are_short_ascii_tags(self, raw, clean):
        assert RequestSource.mcp("t", raw, None, None).client == clean

    def test_state_round_trip(self):
        assert RequestSource.from_state(SOURCE.to_state()) == SOURCE

    @pytest.mark.parametrize("state", [None, {}, "mcp", {"source": "web", "tool": "x"}])
    def test_anything_else_in_state_is_no_source(self, state):
        assert RequestSource.from_state(state) is None


# --- the header dependency --------------------------------------------------------


def _echo_app():
    app = FastAPI()

    @app.get("/probe")
    def probe(source=Depends(get_request_source)):
        return source.to_state() if source else None

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestGetRequestSource:
    @pytest.mark.asyncio
    async def test_mcp_headers_become_a_source(self):
        async with _echo_app() as client:
            body = (await client.get("/probe", headers=MCP_HEADERS)).json()
        assert body == SOURCE.to_state()

    @pytest.mark.asyncio
    async def test_a_website_call_has_no_source(self):
        async with _echo_app() as client:
            assert (await client.get("/probe")).json() is None

    @pytest.mark.asyncio
    async def test_other_source_values_and_stray_mcp_headers_are_ignored(self):
        async with _echo_app() as client:
            headers = {**MCP_HEADERS, "X-Request-Source": "web"}
            assert (await client.get("/probe", headers=headers)).json() is None
            stray = {k: v for k, v in MCP_HEADERS.items() if k != "X-Request-Source"}
            assert (await client.get("/probe", headers=stray)).json() is None


# --- the executor keeps it on the job -----------------------------------------------


def _executor():
    from core.executor import WorkflowExecutor

    cfg = ExecutorConfig(model_name="gemini-test", google_api_key="k", cache_enabled=False, use_database=False)
    return WorkflowExecutor(cfg, model=MagicMock())


async def _state(executor, user_id, job_id):
    from core.executor import APP_NAME

    session = await executor.session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=job_id)
    return session.state


class TestExecutorStoresSource:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("source", [SOURCE, None])
    async def test_discover_writes_the_source_into_the_new_session(self, source):
        executor = _executor()
        stream = executor.discover("1984", "George Orwell", user_id="mcp:s", request_source=source)
        started = await anext(stream)
        await stream.aclose()
        assert isinstance(started, JobStarted)

        state = await _state(executor, "mcp:s", started.job_id)
        if source is None:
            assert SessionStateKeys.REQUEST_SOURCE not in state
        else:
            assert state[SessionStateKeys.REQUEST_SOURCE] == SOURCE.to_state()

    @pytest.mark.asyncio
    async def test_compose_records_its_own_source_and_keeps_discovers_otherwise(self):
        from core.executor import APP_NAME

        executor = _executor()
        regions = {"regions": [{"region_id": 1, "region_name": "Paris", "cities": [{"name": "Paris"}]}]}
        discover_source = RequestSource.mcp("get_literary_route", "codex", "1", "s")
        for job_id, compose_source, expected in (
            ("j-keep", None, discover_source),
            ("j-new", SOURCE, SOURCE),
        ):
            await executor.session_service.create_session(
                app_name=APP_NAME,
                user_id="mcp:s",
                session_id=job_id,
                state={
                    SessionStateKeys.BOOK_TITLE: "1984",
                    SessionStateKeys.AUTHOR: "George Orwell",
                    SessionStateKeys.REGION_ANALYSIS: regions,
                    SessionStateKeys.REQUEST_SOURCE: discover_source.to_state(),
                },
            )
            stream = executor.compose(job_id, [1], user_id="mcp:s", request_source=compose_source)
            first = await anext(stream)
            await stream.aclose()
            assert isinstance(first, ProgressEvent)
            state = await _state(executor, "mcp:s", job_id)
            assert RequestSource.from_state(state[SessionStateKeys.REQUEST_SOURCE]) == expected


# --- the Langfuse plugin tags the trace ------------------------------------------------


class _Root:
    trace_id = "t"

    def start_observation(self, **kwargs):
        return _Root()

    def update(self, **kwargs):
        pass

    def end(self):
        pass


def _plugin(monkeypatch):
    calls = []

    def propagate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(__enter__=lambda *a: None, __exit__=lambda *a: None)

    monkeypatch.setattr("plugins.langfuse_plugin.propagate_attributes", propagate)
    plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
    plugin.enabled = True
    plugin.client = SimpleNamespace(start_observation=lambda **kw: _Root())
    plugin.root_name = "book_to_place_composition"
    return plugin, calls


def _ctx(state):
    return SimpleNamespace(
        invocation_id="inv-1",
        user_id="mcp:22522e04",
        session=SimpleNamespace(id="job-1", state=state),
        agent=None,
    )


class TestPluginTags:
    @pytest.mark.asyncio
    async def test_an_mcp_job_is_tagged_by_source_tool_and_client(self, monkeypatch):
        plugin, calls = _plugin(monkeypatch)
        await plugin.on_user_message_callback(
            invocation_context=_ctx({SessionStateKeys.REQUEST_SOURCE: SOURCE.to_state()}), user_message="go"
        )
        (attrs,) = calls
        assert attrs["tags"] == ["mcp", "tool:plan_trip", "client:claude-ai"]
        assert attrs["metadata"]["source"] == "mcp"
        assert attrs["metadata"]["mcp_tool"] == "plan_trip"
        assert attrs["metadata"]["mcp_client"] == "claude-ai"
        assert attrs["metadata"]["mcp_client_version"] == "0.1.0"
        assert attrs["metadata"]["mcp_session"] == "22522e04"
        assert {"prompt_version", "prompt_hash"} <= attrs["metadata"].keys()
        assert attrs["user_id"] == "mcp:22522e04"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", [{}, None, {SessionStateKeys.BOOK_TITLE: "1984"}])
    async def test_a_website_job_is_unchanged(self, monkeypatch, state):
        plugin, calls = _plugin(monkeypatch)
        await plugin.on_user_message_callback(invocation_context=_ctx(state), user_message="go")
        (attrs,) = calls
        assert attrs["tags"] is None
        assert set(attrs["metadata"]) == {"prompt_version", "prompt_hash"}
