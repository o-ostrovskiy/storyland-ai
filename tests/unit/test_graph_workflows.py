"""Semantics pins for ADK 2 graph workflows (PR 3 of the ADK 2 migration).

These tests run REAL `google.adk.workflow.Workflow` graphs through a REAL
Runner with a scripted model (Gemini.generate_content_async patched at class
level, the same seam test_cache_real_api meters). They pin the four
behaviors the orchestrator rewrite depends on — if an ADK upgrade changes
any of them, these fail before the eval gate has to catch it:

1. History: a downstream LlmAgent node sees the upstream node's response in
   its llm_request contents (the researcher→formatter anti-hallucination
   contract, ADR #2).
2. output_key: a formatter's structured output lands in session.state under
   its output_key (the state-handoff contract, ADR #5).
3. Authorship: events carry the AGENT's name as author (the progress-mapping
   contract the harness pump and golden-stream rely on).
4. Fan-out/fan-in: a tuple chain element fans out, a JoinNode waits for ALL
   branches (the parallel-discovery contract, ADR #3).
"""

import json

import pytest
from pydantic import BaseModel

import google.adk.models.google_llm as google_llm
from google.adk.agents import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import JoinNode, START, Workflow
from google.genai import types

from core.events import Phase

APP = "storyland"
USER = "graph-test"


class _FactOut(BaseModel):
    fact: str


def _model():
    return google_llm.Gemini(
        model="gemini-2.5-flash-lite", client_kwargs={"api_key": "dummy"}
    )


def _canned(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15
        ),
    )


def _install_scripted_model(monkeypatch, script_for_request):
    """Patch Gemini.generate_content_async with a scripted responder.

    ``script_for_request(llm_request) -> str`` chooses the canned text; every
    captured llm_request is recorded for assertions.
    """
    captured = []

    async def scripted(self, *args, **kwargs):
        llm_request = args[0] if args else kwargs.get("llm_request")
        captured.append(llm_request)
        yield _canned(script_for_request(llm_request))

    monkeypatch.setattr(google_llm.Gemini, "generate_content_async", scripted)
    return captured


def _request_text(llm_request) -> str:
    """All text visible to the model in one request (system + contents)."""
    chunks = []
    config = getattr(llm_request, "config", None)
    system = getattr(config, "system_instruction", None) if config else None
    if system:
        chunks.append(str(system))
    for content in getattr(llm_request, "contents", None) or []:
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


async def _run(workflow, prompt="run it"):
    service = InMemorySessionService()
    session_id = "graph-session"
    await service.create_session(
        app_name=APP, user_id=USER, session_id=session_id, state={}
    )
    runner = Runner(node=workflow, app_name=APP, session_service=service)
    events = []
    async with runner:
        async for event in runner.run_async(
            user_id=USER,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            events.append(event)
    session = await service.get_session(
        app_name=APP, user_id=USER, session_id=session_id
    )
    return events, session


class TestGraphSemantics:
    async def test_downstream_node_sees_upstream_history_and_output_key(
        self, monkeypatch
    ):
        researcher = LlmAgent(
            name="g_researcher",
            model=_model(),
            instruction="RESEARCH-MARKER: find facts.",
        )
        formatter = LlmAgent(
            name="g_formatter",
            model=_model(),
            instruction="FORMAT-MARKER: structure the researcher's findings.",
            output_schema=_FactOut,
            output_key="g_out",
        )

        def script(llm_request):
            text = _request_text(llm_request)
            if "RESEARCH-MARKER" in text:
                return "GROUNDED FACT ALPHA discovered in the archives."
            return json.dumps({"fact": "GROUNDED FACT ALPHA"})

        captured = _install_scripted_model(monkeypatch, script)

        workflow = Workflow(
            name="g_wf", edges=[(START, researcher, formatter)]
        )
        events, session = await _run(workflow)

        # Pin 1 — history: the formatter's request must contain the
        # researcher's response text (ADR #2 depends on it).
        formatter_requests = [
            r for r in captured if "FORMAT-MARKER" in _request_text(r)
        ]
        assert len(formatter_requests) == 1
        assert "GROUNDED FACT ALPHA discovered in the archives." in _request_text(
            formatter_requests[0]
        ), (
            "formatter did not see the researcher's output in its request — "
            "the researcher→formatter contract is broken under graph workflows"
        )

        # Pin 2 — output_key: structured output landed in session state.
        raw = session.state.get("g_out")
        assert raw is not None, "output_key wrote nothing to session state"
        parsed = raw if isinstance(raw, dict) else json.loads(raw)
        assert parsed["fact"] == "GROUNDED FACT ALPHA"

        # Pin 3 — authorship: agent-authored events keep the agent name.
        authors = {e.author for e in events if getattr(e, "author", None)}
        assert "g_researcher" in authors
        assert "g_formatter" in authors

    async def test_fanout_join_runs_all_branches_before_join_successor(
        self, monkeypatch
    ):
        branch_agents = [
            LlmAgent(
                name=f"branch_{key}",
                model=_model(),
                instruction=f"BRANCH-{key.upper()}-MARKER: research {key}.",
                output_key=f"out_{key}",
            )
            for key in ("city", "landmark", "author")
        ]
        analyzer = LlmAgent(
            name="g_analyzer",
            model=_model(),
            instruction="ANALYZE-MARKER: combine all branch findings.",
            output_key="analysis",
        )

        def script(llm_request):
            text = _request_text(llm_request)
            for key in ("city", "landmark", "author"):
                if f"BRANCH-{key.upper()}-MARKER" in text:
                    return f"finding-for-{key}"
            return "combined-analysis"

        captured = _install_scripted_model(monkeypatch, script)

        join = JoinNode(name="g_join")
        workflow = Workflow(
            name="g_parallel_wf",
            edges=[
                (START, tuple(branch_agents)),
                (branch_agents[0], join),
                (branch_agents[1], join),
                (branch_agents[2], join),
                (join, analyzer),
            ],
        )
        events, session = await _run(workflow)

        # All three branches ran, and each wrote its output_key.
        for key in ("city", "landmark", "author"):
            assert session.state.get(f"out_{key}") == f"finding-for-{key}"

        # Pin 4 — the analyzer ran exactly once, and only after every branch:
        # its request is the LAST model call, and by then all three branch
        # responses exist.
        analyzer_requests = [
            r for r in captured if "ANALYZE-MARKER" in _request_text(r)
        ]
        assert len(analyzer_requests) == 1, (
            "JoinNode must gate the analyzer to exactly one run after ALL "
            "predecessors — got %d runs" % len(analyzer_requests)
        )
        assert captured.index(analyzer_requests[0]) == len(captured) - 1
        assert session.state.get("analysis") == "combined-analysis"

        # Branch events keep their agent authorship (progress mapping).
        authors = {e.author for e in events if getattr(e, "author", None)}
        assert {"branch_city", "branch_landmark", "branch_author"} <= authors


class TestHarnessPumpAgainstRealGraph:
    """Drive the REAL harness pump against a REAL graph run (Codex P2 check).

    The concern: Workflow sets ctx.event_author = workflow name, so child
    events might be workflow-authored, silently breaking pump_events'
    per-agent progress mapping and capture_authors researcher-text capture.
    This runs the actual pump_events (not a fake) over an actual
    Runner(node=Workflow) stream and asserts both mechanisms work — the
    LLM-content events are agent-authored even though workflow-level events
    exist alongside them.
    """

    async def test_pump_progress_and_capture_work_under_graphs(self, monkeypatch):
        from core.run_harness import RunCapture, pump_events

        researcher = LlmAgent(
            name="p_researcher",
            model=_model(),
            instruction="P-RESEARCH-MARKER: find things.",
        )
        formatter = LlmAgent(
            name="p_formatter",
            model=_model(),
            instruction="P-FORMAT-MARKER: structure things.",
        )

        def script(llm_request):
            if "P-RESEARCH-MARKER" in _request_text(llm_request):
                return "captured-researcher-evidence"
            return "formatted"

        _install_scripted_model(monkeypatch, script)

        service = InMemorySessionService()
        await service.create_session(
            app_name=APP, user_id=USER, session_id="pump", state={}
        )
        workflow = Workflow(name="p_wf", edges=[(START, researcher, formatter)])
        runner = Runner(node=workflow, app_name=APP, session_service=service)

        capture = RunCapture()
        progress = [
            ev
            async for ev in pump_events(
                runner,
                user_id=USER,
                session_id="pump",
                message=types.Content(role="user", parts=[types.Part(text="go")]),
                phase=Phase.DISCOVERY,
                agent_steps={
                    "p_researcher": "Researching",
                    "p_formatter": "Formatting",
                },
                capture=capture,
                capture_authors=("p_researcher",),
            )
        ]

        assert [p.step for p in progress] == ["Researching", "Formatting"], (
            "per-agent progress mapping broke under the graph runtime — "
            "child events are no longer agent-authored"
        )
        assert capture.text_for("p_researcher") == "captured-researcher-evidence", (
            "capture_authors recorded nothing — the grounding filter's "
            "evidence path is dead under graphs"
        )


class TestGraphHistoryScoping:
    """Pin the graph runtime's history model so nobody re-assumes 1.x behavior.

    Under graphs, an agent's conversation is scoped to its TRIGGER CHAIN:
    a direct successor sees its predecessor's output (pinned above), but a
    second invocation on the same session sees NOTHING of the first — not
    even its user message. This is why the composer's grounded inputs are
    passed explicitly via build_composition_prompt (state → prompt), not via
    implicit conversation history. If an ADK upgrade ever changes this to
    full-history again, this test fails and the explicit wiring (plus its
    token cost) should be re-evaluated.
    """

    async def test_second_invocation_sees_nothing_of_the_first(self, monkeypatch):
        first = LlmAgent(
            name="first_agent",
            model=_model(),
            instruction="FIRST-MARKER: produce facts.",
        )
        second = LlmAgent(
            name="second_agent",
            model=_model(),
            instruction="SECOND-MARKER: compose from whatever you can see.",
        )

        def script(llm_request):
            text = _request_text(llm_request)
            if "FIRST-MARKER" in text:
                return "FIRST-INVOCATION-OUTPUT"
            return "second-output"

        captured = _install_scripted_model(monkeypatch, script)

        service = InMemorySessionService()
        await service.create_session(
            app_name=APP, user_id=USER, session_id="cross", state={}
        )

        async def drive(workflow, prompt):
            runner = Runner(node=workflow, app_name=APP, session_service=service)
            async with runner:
                async for _ in runner.run_async(
                    user_id=USER,
                    session_id="cross",
                    new_message=types.Content(
                        role="user", parts=[types.Part(text=prompt)]
                    ),
                ):
                    pass

        await drive(Workflow(name="wf_a", edges=[(START, first)]), "prompt-one")
        await drive(Workflow(name="wf_b", edges=[(START, second)]), "prompt-two")

        second_requests = [
            r for r in captured if "SECOND-MARKER" in _request_text(r)
        ]
        assert len(second_requests) == 1
        text = _request_text(second_requests[0])
        assert "FIRST-INVOCATION-OUTPUT" not in text
        assert "prompt-one" not in text
