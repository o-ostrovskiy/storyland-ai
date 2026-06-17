"""Real-API integration test for the Discovery result cache.

Unlike ``tests/unit/test_cache.py`` (which mocks the discovery chain), this
test exercises the cache against the *live* Gemini API:

1. First ``discover()`` call is a cache MISS -> runs the real discovery chain,
   makes one or more Gemini calls and consumes real tokens.
2. Second ``discover()`` call with an identical book/author/preferences is a
   cache HIT -> must make ZERO new Gemini calls and consume ZERO new Gemini
   tokens, while returning the same previously-validated regions.

"Gemini tokens" are measured at ground truth by metering
``Gemini.generate_content_async`` directly (call count + summed
``usage_metadata.total_token_count``), so the assertion does not depend on
Langfuse being configured.

Run with: ``make test-integration-real`` or
``pytest tests/integration/test_cache_real_api.py -v -m real_api``.
Requires ``GOOGLE_API_KEY``; skips otherwise.
"""

import os

import pytest
import google.adk.models.google_llm as google_llm

from core.events import RegionsReady, WorkflowComplete, WorkflowError
from core.executor import WorkflowExecutor
from core.types import ExecutorConfig
from services.session_service import create_session_service


class _GeminiMeter:
    """Counts live Gemini generation calls and the tokens they consume."""

    def __init__(self):
        self.calls = 0
        self.total_tokens = 0


def _install_gemini_meter(monkeypatch) -> _GeminiMeter:
    """Wrap Gemini.generate_content_async to record real API usage.

    Every discovery agent shares the Gemini class, so patching at the class
    level captures all generation traffic the workflow produces.
    """
    meter = _GeminiMeter()
    real_generate = google_llm.Gemini.generate_content_async

    async def metered_generate(self, *args, **kwargs):
        meter.calls += 1
        async for response in real_generate(self, *args, **kwargs):
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                meter.total_tokens += getattr(usage, "total_token_count", 0) or 0
            yield response

    monkeypatch.setattr(
        google_llm.Gemini, "generate_content_async", metered_generate
    )
    return meter


def _first(events, event_type):
    return next((e for e in events if isinstance(e, event_type)), None)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.real_api
async def test_second_discover_is_cache_hit_with_zero_new_gemini_tokens(
    real_api_key, monkeypatch
):
    """A repeated discover() must short-circuit with zero new Gemini tokens."""
    meter = _install_gemini_meter(monkeypatch)

    config = ExecutorConfig(
        model_name=os.getenv("MODEL_NAME", "gemini-2.5-flash"),
        google_api_key=real_api_key,
        use_database=False,
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    executor = WorkflowExecutor(
        config=config,
        session_service=create_session_service(use_database=False),
    )
    assert executor._discovery_cache is not None, "cache must be enabled"

    request = {"book_title": "The Great Gatsby", "author": "F. Scott Fitzgerald"}

    # --- Call 1: cache MISS, real discovery against live Gemini ---------------
    first_events = [event async for event in executor.discover(**request)]

    assert _first(first_events, WorkflowError) is None, "first discover errored"
    first_regions = _first(first_events, RegionsReady)
    assert first_regions is not None and first_regions.regions, (
        "first discover must produce regions to populate the cache"
    )
    calls_after_first = meter.calls
    tokens_after_first = meter.total_tokens
    assert calls_after_first > 0, "miss should invoke Gemini at least once"
    assert tokens_after_first > 0, "miss should consume real Gemini tokens"

    # --- Call 2: identical request, expected cache HIT ------------------------
    second_events = [event async for event in executor.discover(**request)]

    assert _first(second_events, WorkflowError) is None, "second discover errored"
    second_regions = _first(second_events, RegionsReady)
    assert second_regions is not None, "second discover must return regions"

    new_calls = meter.calls - calls_after_first
    new_tokens = meter.total_tokens - tokens_after_first
    assert new_calls == 0, (
        f"cache hit must make zero Gemini calls, made {new_calls}"
    )
    assert new_tokens == 0, (
        f"cache hit must consume zero new Gemini tokens, consumed {new_tokens}"
    )

    # The hit must serve the same previously-validated regions...
    assert second_regions.regions == first_regions.regions
    # ...and the cached fast-path reports no token usage at completion.
    second_complete = _first(second_events, WorkflowComplete)
    assert second_complete is not None
    assert second_complete.token_usage is None

    print(
        f"\n✅ cache hit verified: call 1 = {calls_after_first} Gemini call(s) / "
        f"{tokens_after_first} tokens; call 2 = +{new_calls} calls / "
        f"+{new_tokens} tokens"
    )
