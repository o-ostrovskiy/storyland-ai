"""ADR #40: the compose cache serves the reader's FIRST compose of a discovered book.

These tests pin, offline (the composer model is scripted, nothing calls Gemini):

* a second reader of the same book, composing regions[0], gets the journey with
  ZERO composer calls and a session that matches a fresh compose (only the
  per-call chip ids differ, as they do between two fresh composes);
* the hit replays the stored RAW envelope through the same extraction and
  guards, so a guard fix applies to hits at once;
* what is NOT cached: any other selection, a different grounding, an empty
  journey, a disabled cache, a session that predates the discovery key;
* a cache read or write error never breaks compose, and an unusable cached
  envelope falls back to a fresh run;
* the namespace: Discovery inputs + composer modules + the output ceiling.
"""

import copy
import json
import uuid

import pytest

import google.adk.models.google_llm as google_llm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

import core.executor as executor_module
from core.cache import TTLCache, build_compose_cache
from core.cache_version import (
    _COMPOSE_MODULES,
    _PROMPT_MODULES,
    compute_cache_version,
    compute_compose_cache_version,
)
from core.events import ItineraryReady, ProgressEvent, WorkflowComplete, WorkflowError
from core.executor import APP_NAME, WorkflowExecutor
from core.session_state import SessionStateKeys
from core.types import ExecutorConfig
from services.session_service import create_session_service

TITLE = "A Gentleman in Moscow"
AUTHOR = "Amor Towles"

ENVELOPE = {
    "itinerary": {
        "cities": [
            {
                "name": "Moscow",
                "country": "Russia",
                "days_suggested": 2,
                "overview": "The Count's city.",
                "stops": [
                    {
                        "name": "Hotel Metropol",
                        "type": "hotel",
                        "reason": "The Count lives here under house arrest.",
                        "time_of_day": "morning",
                    }
                ],
            }
        ],
        "summary_text": "Two days around the Metropol.",
    },
    "suggestions": [
        {
            "id": "model-id",
            "label": "More hotels",
            "action_prompt": "Find more grand hotels in Moscow",
        }
    ],
}
EMPTY_ENVELOPE = {
    "itinerary": {"cities": [], "summary_text": "Nothing found."},
    "suggestions": [],
}

REGIONS = {
    "regions": [
        {
            "region_id": 1,
            "name": "Moscow, Russia",
            "country_code": "RU",
            "primary_locality": "Moscow",
            "cities": [{"name": "Moscow", "country": "Russia"}],
        },
        {
            "region_id": 2,
            "name": "Paris, France",
            "country_code": "FR",
            "primary_locality": "Paris",
            "cities": [{"name": "Paris", "country": "France"}],
        },
    ],
    "analysis_note": "",
}


def _gemini():
    return google_llm.Gemini(
        model="gemini-2.5-flash-lite", client_kwargs={"api_key": "dummy"}
    )


def _script_model(monkeypatch, envelope=ENVELOPE):
    """Every composer call answers ``envelope``; returns the call log."""
    calls = []

    async def scripted(self, *args, **kwargs):
        calls.append(1)
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text=json.dumps(envelope))]
            ),
            finish_reason=types.FinishReason.STOP,
        )

    monkeypatch.setattr(google_llm.Gemini, "generate_content_async", scripted)
    return calls


def _executor(**config):
    return WorkflowExecutor(
        config=ExecutorConfig(
            model_name="gemini-2.5-flash-lite", google_api_key="test-key", **config
        ),
        session_service=create_session_service(use_database=False),
        model=_gemini(),
    )


async def _prime_discovery(executor, bundle=None):
    key = executor._versioned_key(
        WorkflowExecutor._discovery_cache_key(TITLE, AUTHOR, None)
    )
    await executor._discovery_cache.set(
        key,
        bundle
        or {
            "region_analysis": copy.deepcopy(REGIONS),
            "book_context": {"themes": ["house arrest"]},
            "city_discovery": {"cities": [{"name": "Moscow"}]},
            "unverified_discovery": [],
        },
    )


async def _discover(executor):
    events = [e async for e in executor.discover(book_title=TITLE, author=AUTHOR)]
    return events[0].job_id


async def _compose(executor, job_id, region_ids=(1,)):
    return [e async for e in executor.compose(job_id=job_id, region_ids=list(region_ids))]


async def _state(executor, job_id):
    session = await executor._session_service.get_session(
        app_name=APP_NAME, user_id="default", session_id=job_id
    )
    return dict(session.state)


def _without_ids(state):
    """A session's state with the per-call server ids blanked out."""
    state = copy.deepcopy(state)
    for chip in state.get(SessionStateKeys.LAST_SUGGESTIONS) or []:
        chip["id"] = "<id>"
    chip = state.get(SessionStateKeys.BOOK_RECOMMENDATION_CHIP)
    if chip:
        chip["id"] = "<id>"
    if state.get(SessionStateKeys.BOOK_RECOMMENDATION_CHIP_ID):
        state[SessionStateKeys.BOOK_RECOMMENDATION_CHIP_ID] = "<id>"
    return state


async def test_second_reader_gets_the_journey_without_a_composer_call(monkeypatch):
    calls = _script_model(monkeypatch)
    executor = _executor()
    await _prime_discovery(executor)

    first_job = await _discover(executor)
    fresh = await _compose(executor, first_job)
    assert len(calls) == 1
    assert len(executor._compose_cache) == 1

    second_job = await _discover(executor)
    hit = await _compose(executor, second_job)
    assert len(calls) == 1, "a hit makes zero composer calls"

    fresh_ready = next(e for e in fresh if isinstance(e, ItineraryReady))
    hit_ready = next(e for e in hit if isinstance(e, ItineraryReady))
    assert hit_ready.itinerary == fresh_ready.itinerary
    assert [c["label"] for c in hit_ready.suggestions] == ["More hotels"]
    # Like a discover hit, a compose hit carries no progress and no usage.
    assert any(isinstance(e, ProgressEvent) for e in fresh)
    assert not any(isinstance(e, ProgressEvent) for e in hit)
    complete = [e for e in hit if isinstance(e, WorkflowComplete)]
    assert len(complete) == 1 and complete[0].token_usage is None
    assert not any(isinstance(e, WorkflowError) for e in hit)

    # The session a hit leaves is the session a fresh compose leaves.
    fresh_state = await _state(executor, first_job)
    hit_state = await _state(executor, second_job)
    assert _without_ids(hit_state) == _without_ids(fresh_state)

    # Chip ids are minted per call: two readers never share one, and the ids
    # the hit emitted are the ones expand() will accept for that job.
    fresh_ids = {c["id"] for c in fresh_state[SessionStateKeys.LAST_SUGGESTIONS]}
    hit_ids = {c["id"] for c in hit_state[SessionStateKeys.LAST_SUGGESTIONS]}
    assert fresh_ids.isdisjoint(hit_ids)
    assert hit_ids == {c["id"] for c in hit_ready.suggestions}
    assert "model-id" not in hit_ids
    assert (
        hit_state[SessionStateKeys.BOOK_RECOMMENDATION_CHIP_ID]
        == hit_ready.book_recommendation_chip["id"]
        != fresh_state[SessionStateKeys.BOOK_RECOMMENDATION_CHIP_ID]
    )


async def test_expand_accepts_a_chip_from_a_cache_hit(monkeypatch):
    _script_model(monkeypatch)
    executor = _executor()
    await _prime_discovery(executor)
    await _compose(executor, await _discover(executor))
    job = await _discover(executor)
    hit = await _compose(executor, job)
    chip = next(e for e in hit if isinstance(e, ItineraryReady)).suggestions[0]

    # Stop expand() at the model: admission (job ready, chip id valid) is
    # what a hit must satisfy; the expansion itself is not under test.
    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(executor_module, "create_expansion_workflow", stop, raising=False)
    monkeypatch.setattr(executor, "_build_runner", stop)
    events = [
        e
        async for e in executor.expand(
            job_id=job,
            action_id=chip["id"],
            action_label=chip["label"],
            action_prompt="ignored",
        )
    ]
    errors = [e for e in events if isinstance(e, WorkflowError)]
    assert errors, "the run got past admission and stopped at the model"
    assert all(
        e.error_type not in {"InvalidActionId", "ItineraryNotReady", "JobFailed"}
        for e in errors
    )


async def test_a_hit_reruns_extraction_against_the_new_session(monkeypatch):
    _script_model(monkeypatch)
    executor = _executor()
    await _prime_discovery(executor)
    first_job = await _discover(executor)
    await _compose(executor, first_job)
    # ADK stores the schema-validated envelope (defaults filled in); that dict,
    # not the model's text, is what the cache holds and replays.
    stored = (await _state(executor, first_job))[SessionStateKeys.COMPOSER_ENVELOPE]

    seen = []
    real = executor_module.extract_itinerary_from_response

    def spy(final_response, state_accessor):
        seen.append(state_accessor.composer_envelope)
        return real(final_response, state_accessor)

    monkeypatch.setattr(executor_module, "extract_itinerary_from_response", spy)
    await _compose(executor, await _discover(executor))
    assert seen == [stored], "the raw envelope goes through extraction on a hit"


async def test_only_the_first_region_is_cached(monkeypatch):
    calls = _script_model(monkeypatch)
    executor = _executor()
    await _prime_discovery(executor)
    for region_ids in ((2,), (1, 2), (2,)):
        await _compose(executor, await _discover(executor), region_ids)
    assert len(calls) == 3
    assert len(executor._compose_cache) == 0


async def test_different_grounding_is_a_miss(monkeypatch):
    calls = _script_model(monkeypatch)
    executor = _executor()
    await _prime_discovery(executor)
    await _compose(executor, await _discover(executor))

    # The discovery entry is recomputed with other grounding (evicted, then a
    # fresh run): same discover key and region, so only the prompt differs.
    await _prime_discovery(
        executor,
        {
            "region_analysis": copy.deepcopy(REGIONS),
            "book_context": {"themes": ["a different reading"]},
            "unverified_discovery": [],
        },
    )
    await _compose(executor, await _discover(executor))
    assert len(calls) == 2
    assert len(executor._compose_cache) == 2


async def test_an_empty_journey_is_not_stored(monkeypatch):
    calls = _script_model(monkeypatch, EMPTY_ENVELOPE)
    executor = _executor()
    await _prime_discovery(executor)
    await _compose(executor, await _discover(executor))
    await _compose(executor, await _discover(executor))
    assert len(calls) == 2
    assert len(executor._compose_cache) == 0


async def test_disabled_cache_never_stores(monkeypatch):
    calls = _script_model(monkeypatch)
    executor = _executor(cache_enabled=False)
    for _ in range(2):
        job = await _discover_without_cache(executor)
        await _compose(executor, job)
    assert len(calls) == 2
    assert len(executor._compose_cache) == 0


async def _discover_without_cache(executor):
    """A job in the state discover() leaves, without running discovery."""
    job_id = str(uuid.uuid4())
    await executor._session_service.create_session(
        app_name=APP_NAME,
        user_id="default",
        session_id=job_id,
        state={
            SessionStateKeys.BOOK_TITLE: TITLE,
            SessionStateKeys.AUTHOR: AUTHOR,
            SessionStateKeys.DISCOVERY_CACHE_KEY: WorkflowExecutor._discovery_cache_key(
                TITLE, AUTHOR, None
            ),
            SessionStateKeys.REGION_ANALYSIS: copy.deepcopy(REGIONS),
        },
    )
    return job_id


async def test_a_session_without_the_discovery_key_composes_fresh(monkeypatch):
    calls = _script_model(monkeypatch)
    executor = _executor()
    for n in range(2):
        job_id = f"legacy-{n}"
        await executor._session_service.create_session(
            app_name=APP_NAME,
            user_id="default",
            session_id=job_id,
            state={SessionStateKeys.REGION_ANALYSIS: copy.deepcopy(REGIONS)},
        )
        await _compose(executor, job_id)
    assert len(calls) == 2
    assert len(executor._compose_cache) == 0


async def test_discover_writes_the_discovery_key(monkeypatch):
    executor = _executor()
    await _prime_discovery(executor)
    state = await _state(executor, await _discover(executor))
    assert state[SessionStateKeys.DISCOVERY_CACHE_KEY] == (
        WorkflowExecutor._discovery_cache_key(TITLE, AUTHOR, None)
    )


class _BrokenCache:
    async def get(self, key):
        raise OSError("disk gone")

    async def set(self, key, value, ttl_seconds=None):
        raise OSError("disk gone")

    def __len__(self):
        return 0


async def test_cache_errors_never_break_compose(monkeypatch):
    calls = _script_model(monkeypatch)
    executor = _executor()
    executor._compose_cache = _BrokenCache()
    await _prime_discovery(executor)
    events = await _compose(executor, await _discover(executor))
    assert len(calls) == 1
    assert any(isinstance(e, ItineraryReady) for e in events)
    assert not any(isinstance(e, WorkflowError) for e in events)


@pytest.mark.parametrize("stored", ["not a dict", {"itinerary": "broken"}])
async def test_an_unusable_entry_falls_back_to_a_fresh_run(monkeypatch, stored):
    calls = _script_model(monkeypatch)
    executor = _executor()
    await _prime_discovery(executor)
    first_job = await _discover(executor)
    await _compose(executor, first_job)
    good = (await _state(executor, first_job))[SessionStateKeys.COMPOSER_ENVELOPE]
    (key,) = list(executor._compose_cache._store.keys())
    await executor._compose_cache.set(key, stored)

    job = await _discover(executor)
    events = await _compose(executor, job)
    assert len(calls) == 2
    ready = next(e for e in events if isinstance(e, ItineraryReady))
    assert ready.itinerary["cities"][0]["name"] == "Moscow"
    state = await _state(executor, job)
    assert state[SessionStateKeys.COMPOSER_ENVELOPE] == good
    # The fresh run re-stored a good envelope over the unusable one.
    assert await executor._compose_cache.get(key) == good


class TestNamespace:
    def test_compose_modules_extend_the_discovery_inputs(self):
        assert set(_COMPOSE_MODULES).isdisjoint(_PROMPT_MODULES)
        assert "models.itinerary" in _COMPOSE_MODULES
        assert "agents.trip_composer_agent" in _COMPOSE_MODULES

    def test_differs_from_discovery_and_tracks_its_inputs(self):
        base = compute_compose_cache_version("m", 8192)
        assert base == compute_compose_cache_version("m", 8192)
        assert base != compute_cache_version("m")
        assert base != compute_compose_cache_version("m", 4096)
        assert base != compute_compose_cache_version("other", 8192)

    def test_a_composer_module_edit_flips_it(self, monkeypatch):
        import core.cache_version as cv

        base = compute_compose_cache_version("m", 8192)
        discovery = compute_cache_version("m")
        real = cv._module_source
        monkeypatch.setattr(
            cv,
            "_module_source",
            lambda name: real(name) + ("#edit" if name == "models.itinerary" else ""),
        )
        assert compute_compose_cache_version("m", 8192) != base
        # ...and leaves the Discovery namespace alone: no discover is re-paid.
        assert compute_cache_version("m") == discovery


class TestBuildComposeCache:
    def test_memory_backend(self):
        cache = build_compose_cache(
            ExecutorConfig(model_name="m", google_api_key="k")
        )
        assert isinstance(cache, TTLCache)

    def test_disk_backend_has_its_own_store(self, tmp_path):
        cache = build_compose_cache(
            ExecutorConfig(
                model_name="m",
                google_api_key="k",
                cache_backend="disk",
                cache_dir=str(tmp_path),
                cache_ttl_seconds=2592000,
                cache_max_entries=500,
            )
        )
        try:
            assert cache.directory == str(tmp_path / "compose")
            assert cache._ttl == 2592000
        finally:
            cache.close()
