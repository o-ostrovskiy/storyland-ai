"""Golden-stream snapshot: the SSE event contract for discover -> compose.

This is the migration-safety net (ADK 1.x -> 2.x, template -> graph runtime):
whatever changes inside the executor/harness/agents, the ordered sequence of
SSE event types the frontend sees for a successful discover -> compose run —
and for the standard rejection path — must not change. Uses fake Runners (no
Gemini), the real executor, the real harness, and the real
``domain_event_to_sse`` mapping, so it exercises the full core -> api seam.

If this test reds on a migration PR, the wire contract changed: either fix
the regression or (deliberately, with a frontend ticket) update the snapshot.
"""

from google.adk.events import Event
from google.adk.events.event_actions import EventActions

from api.streaming import domain_event_to_sse
from core.events import JobStarted
from core.session_state import SessionStateKeys
from core.types import ExecutorConfig

# Minimal valid ComposerEnvelope payload (validated by extraction).
_ENVELOPE = {
    "itinerary": {
        "cities": [
            {
                "name": "Bath",
                "country": "England",
                "days_suggested": 2,
                "overview": "Austen's Bath",
                "stops": [
                    {
                        "name": "The Pump Room",
                        "type": "landmark",
                        "reason": "Featured in Persuasion",
                        "time_of_day": "morning",
                    }
                ],
            }
        ],
        "summary_text": "A Persuasion journey through Bath.",
    },
    "suggestions": [
        {"label": "Add cafes nearby", "action_prompt": "Find literary cafes in Bath"}
    ],
}

_REGIONS = [{"region_id": 1, "name": "Bath, England"}]


def _state_writing_runner(state_delta_fn):
    """A fake Runner that writes a state delta via append_event, like a real run."""

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self._session_service = kwargs.get("session_service")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run_async(self, user_id, session_id, new_message):
            from core.executor import APP_NAME

            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id
            )
            ev = Event(
                invocation_id="system",
                author="system",
                actions=EventActions(state_delta=state_delta_fn()),
            )
            await self._session_service.append_event(session, ev)
            if False:  # async generator with no yielded events
                yield None

    return _FakeRunner


def _make_executor(monkeypatch):
    import core.executor as ex
    from core.executor import WorkflowExecutor
    from services.session_service import create_session_service

    monkeypatch.setattr(ex, "create_discovery_workflow", lambda *a, **k: object())
    monkeypatch.setattr(ex, "create_composition_workflow", lambda *a, **k: object())

    config = ExecutorConfig(
        model_name="gemini-2.0-flash",
        google_api_key="test-key",
        cache_enabled=False,  # deterministic: no fast-path replay between tests
    )
    return WorkflowExecutor(
        config=config,
        session_service=create_session_service(use_database=False),
        model=object(),
    ), ex


async def _collect_sse(agen):
    domain_types, sse_names = [], []
    async for ev in agen:
        domain_types.append(type(ev).__name__)
        sse_names.append(domain_event_to_sse(ev)["event"])
    return domain_types, sse_names


class TestGoldenStream:
    async def test_discover_then_compose_happy_path(self, monkeypatch):
        executor, ex = _make_executor(monkeypatch)

        monkeypatch.setattr(
            ex,
            "Runner",
            _state_writing_runner(
                lambda: {
                    SessionStateKeys.REGION_ANALYSIS: {
                        "regions": _REGIONS,
                        "analysis_note": "note",
                    }
                }
            ),
        )

        job_id = None
        discover_types, discover_sse = [], []
        async for ev in executor.discover(book_title="Persuasion", author="Jane Austen"):
            if isinstance(ev, JobStarted):
                job_id = ev.job_id
            discover_types.append(type(ev).__name__)
            discover_sse.append(domain_event_to_sse(ev)["event"])

        # GOLDEN: the exact discover stream the frontend contract relies on.
        assert discover_types == [
            "JobStarted",
            "MetadataReady",
            "ProgressEvent",
            "RegionsReady",
            "WorkflowComplete",
        ]
        assert discover_sse == ["started", "metadata", "progress", "regions", "done"]
        assert job_id is not None

        monkeypatch.setattr(
            ex,
            "Runner",
            _state_writing_runner(
                lambda: {SessionStateKeys.COMPOSER_ENVELOPE: dict(_ENVELOPE)}
            ),
        )

        compose_types, compose_sse = await _collect_sse(
            executor.compose(job_id=job_id, region_ids=[1])
        )

        # GOLDEN: the exact compose stream.
        assert compose_types == [
            "ProgressEvent",
            "ItineraryReady",
            "WorkflowComplete",
        ]
        assert compose_sse == ["progress", "itinerary", "done"]

    async def test_compose_rejection_stream(self, monkeypatch):
        """The standard rejection path is always exactly error -> done."""
        executor, ex = _make_executor(monkeypatch)
        monkeypatch.setattr(
            ex,
            "Runner",
            _state_writing_runner(
                lambda: {
                    SessionStateKeys.REGION_ANALYSIS: {
                        "regions": _REGIONS,
                        "analysis_note": "note",
                    }
                }
            ),
        )

        job_id = None
        async for ev in executor.discover(book_title="Persuasion", author="Jane Austen"):
            if isinstance(ev, JobStarted):
                job_id = ev.job_id

        types_, sse = await _collect_sse(
            executor.compose(job_id=job_id, region_ids=[999])  # invalid region id
        )
        assert types_ == ["WorkflowError", "WorkflowComplete"]
        assert sse == ["error", "done"]

    async def test_unknown_job_rejection_stream(self, monkeypatch):
        executor, _ = _make_executor(monkeypatch)
        types_, sse = await _collect_sse(
            executor.compose(job_id="nonexistent-job", region_ids=[1])
        )
        assert types_ == ["WorkflowError", "WorkflowComplete"]
        assert sse == ["error", "done"]
