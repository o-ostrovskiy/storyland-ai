"""ai-private#33 Defect 1: a truncated composer envelope fails cleanly, never half-recovered.

The composer intermittently emitted an envelope whose JSON stopped mid-string.
Nothing read the finish reason, and the text fallback's ``{ … rfind('}')`` slicer
could parse a smaller, different object out of the cut-off text. These tests pin:

* the detector (the composer's ``after_model_callback``) raises on a length
  finish or an invalid envelope, and passes a valid one through;
* through a REAL ADK Runner, the raise happens before ``output_key`` is written;
* ``compose()`` retries exactly once: truncated-then-good ships the good
  itinerary, and truncated twice is the client-safe transient error. Neither
  path ever calls ``extract_json_from_text``;
* the eval report counts the truncation as a named defect occurrence.
"""

import json

import pytest

import google.adk.models.google_llm as google_llm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import core.executor as executor_module
import core.extraction as extraction_module
from agents.composer_truncation import (
    COMPOSER_ATTEMPTS,
    TRUNCATION_EVENT,
    ComposerEnvelopeTruncated,
    detect_composer_truncation,
    find_composer_truncation,
)
from agents.orchestrator import create_book_to_place_composition_workflow
from agents.trip_composer_agent import create_trip_composer_agent
from common.config import COMPOSER_MAX_OUTPUT_TOKENS
from core.events import ItineraryReady, WorkflowError
from core.executor import APP_NAME, WorkflowExecutor
from core.session_state import SessionStateKeys
from core.types import ExecutorConfig
from evaluation.tools.run_scheduled_eval import case_defect, count_defects
from services.session_service import create_session_service

VALID_ENVELOPE = json.dumps(
    {
        "itinerary": {
            "cities": [],
            "summary_text": "A short trip through Bath.",
        },
        "suggestions": [],
    }
)
# Cut mid-string, the shape observed in the eval runs.
TRUNCATED_ENVELOPE = VALID_ENVELOPE[: len(VALID_ENVELOPE) // 2]


def _response(text, finish=types.FinishReason.STOP, partial=False):
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        finish_reason=finish,
        partial=partial,
    )


class TestDetector:
    def test_valid_envelope_passes_through(self):
        assert detect_composer_truncation(None, _response(VALID_ENVELOPE)) is None

    def test_length_finish_is_truncation_even_if_the_text_parses(self):
        with pytest.raises(ComposerEnvelopeTruncated) as info:
            detect_composer_truncation(
                None, _response(VALID_ENVELOPE, types.FinishReason.MAX_TOKENS)
            )
        assert info.value.finish_reason == "MAX_TOKENS"
        assert info.value.output_length == len(VALID_ENVELOPE)

    def test_cut_off_envelope_with_a_normal_finish_is_truncation(self):
        with pytest.raises(ComposerEnvelopeTruncated):
            detect_composer_truncation(None, _response(TRUNCATED_ENVELOPE))

    def test_partial_stream_chunks_are_not_judged(self):
        assert detect_composer_truncation(
            None, _response(TRUNCATED_ENVELOPE, partial=True)
        ) is None

    def test_empty_text_is_left_to_the_extraction_error_path(self):
        assert detect_composer_truncation(None, _response("")) is None

    def test_found_inside_a_taskgroup_exception_group(self):
        inner = ComposerEnvelopeTruncated("MAX_TOKENS", 10)
        group = ExceptionGroup("unhandled errors in a TaskGroup", [ValueError(), inner])
        assert find_composer_truncation(group) is inner
        assert find_composer_truncation(RuntimeError("other")) is None

    def test_plain_exception_without_group_builtins(self, monkeypatch):
        # Python 3.10 (still in requires-python) has no BaseExceptionGroup builtin.
        monkeypatch.setattr("core.discovery_errors._BASE_EXCEPTION_GROUP", ())
        assert find_composer_truncation(ValueError()) is None
        inner = ComposerEnvelopeTruncated("MAX_TOKENS", 10)
        assert find_composer_truncation(inner) is inner


def test_composer_agent_carries_the_explicit_ceiling_and_the_detector():
    agent = create_trip_composer_agent(model="gemini-test")
    assert agent.generate_content_config.max_output_tokens == COMPOSER_MAX_OUTPUT_TOKENS
    assert agent.after_model_callback is detect_composer_truncation


def _gemini():
    return google_llm.Gemini(model="gemini-2.5-flash-lite", client_kwargs={"api_key": "dummy"})


def _script_model(monkeypatch, responses):
    """Each composer call takes the next canned response; returns the call log."""
    calls = []

    async def scripted(self, *args, **kwargs):
        calls.append(1)
        yield responses[min(len(calls), len(responses)) - 1]

    monkeypatch.setattr(google_llm.Gemini, "generate_content_async", scripted)
    return calls


async def test_real_runner_raises_before_output_key_is_written(monkeypatch):
    _script_model(
        monkeypatch, [_response(TRUNCATED_ENVELOPE, types.FinishReason.MAX_TOKENS)]
    )
    service = InMemorySessionService()
    await service.create_session(app_name=APP_NAME, user_id="u", session_id="s", state={})
    runner = Runner(
        node=create_book_to_place_composition_workflow(_gemini()),
        app_name=APP_NAME,
        session_service=service,
    )
    with pytest.raises(BaseException) as info:
        async with runner:
            async for _ in runner.run_async(
                user_id="u",
                session_id="s",
                new_message=types.Content(role="user", parts=[types.Part(text="go")]),
            ):
                pass
    assert find_composer_truncation(info.value) is not None
    session = await service.get_session(app_name=APP_NAME, user_id="u", session_id="s")
    assert session.state.get(SessionStateKeys.COMPOSER_ENVELOPE) is None


async def _run_compose(monkeypatch, responses):
    calls = _script_model(monkeypatch, responses)

    def slicer_must_not_run(text):  # pragma: no cover - failing is the assertion
        raise AssertionError("a truncated envelope reached the text-fallback slicer")

    monkeypatch.setattr(extraction_module, "extract_json_from_text", slicer_must_not_run)
    executor = WorkflowExecutor(
        config=ExecutorConfig(model_name="gemini-2.5-flash-lite", google_api_key="test-key"),
        session_service=create_session_service(use_database=False),
        model=_gemini(),
    )
    job_id = "job-truncation"
    await executor._session_service.create_session(
        app_name=APP_NAME,
        user_id="default",
        session_id=job_id,
        state={
            SessionStateKeys.REGION_ANALYSIS: {
                "regions": [{"region_id": 1, "name": "Bath, England"}]
            }
        },
    )
    events = [e async for e in executor.compose(job_id=job_id, region_ids=[1])]
    return calls, events


async def test_compose_truncated_then_good_ships_the_good_itinerary(monkeypatch):
    calls, events = await _run_compose(
        monkeypatch,
        [_response(TRUNCATED_ENVELOPE, types.FinishReason.MAX_TOKENS), _response(VALID_ENVELOPE)],
    )
    assert len(calls) == 2, "exactly one retry"
    ready = [e for e in events if isinstance(e, ItineraryReady)]
    assert len(ready) == 1
    assert ready[0].itinerary["summary_text"] == "A short trip through Bath."
    assert not [e for e in events if isinstance(e, WorkflowError)]


async def test_compose_truncated_twice_is_the_clean_transient_error(monkeypatch):
    calls, events = await _run_compose(
        monkeypatch, [_response(TRUNCATED_ENVELOPE, types.FinishReason.MAX_TOKENS)]
    )
    assert len(calls) == COMPOSER_ATTEMPTS, "one retry, then stop"
    assert not [e for e in events if isinstance(e, ItineraryReady)], "no partial itinerary"
    errors = [e for e in events if isinstance(e, WorkflowError)]
    assert len(errors) == 1
    assert errors[0].error_type == "DiscoveryComposeError"
    assert errors[0].reason == "transient"
    assert TRUNCATION_EVENT not in errors[0].message


def test_eval_counts_truncation_as_a_named_defect():
    assert case_defect(ComposerEnvelopeTruncated("MAX_TOKENS", 10)) == TRUNCATION_EVENT
    assert case_defect(RuntimeError("judge 404")) is None
    cases = [
        {"status": "failed", "defect": TRUNCATION_EVENT},
        {"status": "failed", "defect": TRUNCATION_EVENT},
        {"status": "failed", "error": "other"},
        {"status": "evaluated", "scores": {"x": 1}},
    ]
    assert count_defects(cases) == {TRUNCATION_EVENT: 2}
