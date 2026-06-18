"""Unit tests for the empty-discovery guard in WorkflowExecutor.discover().

When the discovery chain returns zero regions (obscure book, failed/empty
google_search, schema-extraction miss), discover() must emit a clean
WorkflowError(error_type="NoRegions") + WorkflowComplete instead of a silent,
empty-but-"successful" RegionsReady that dead-ends the user at the activation
moment (and is recorded as a success by the funnel). Mirrors the existing
compose() NoRegions guard. The guard is always on; rollback = revert the commit.
"""

from google.adk.events import Event
from google.adk.events.event_actions import EventActions

from core.events import RegionsReady, WorkflowError, WorkflowComplete
from core.types import ExecutorConfig
from core.session_state import SessionStateKeys


def _make_executor(monkeypatch, regions):
    """Build a WorkflowExecutor whose discovery chain is stubbed out.

    The Gemini workflow + Runner are replaced so no real model is invoked.
    ``regions`` is the region list the (fake) discovery run writes into
    session state via append_event: pass [] to simulate an empty discovery,
    or a non-empty list to simulate a normal run.
    """
    import core.executor as ex
    from core.executor import WorkflowExecutor, APP_NAME
    from services.session_service import create_session_service

    # Stub the workflow builder: never construct/run a real Gemini chain.
    monkeypatch.setattr(ex, "create_discovery_workflow", lambda *a, **k: object())

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self._session_service = kwargs.get("session_service")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run_async(self, user_id, session_id, new_message):
            # Simulate the discovery chain writing (or not writing) regions
            # into session state, exactly as the real run does.
            if regions is not None:
                session = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=session_id
                )
                ev = Event(
                    invocation_id="system",
                    author="system",
                    actions=EventActions(
                        state_delta={
                            SessionStateKeys.REGION_ANALYSIS: {
                                "regions": regions,
                                "analysis_note": "note",
                            }
                        }
                    ),
                )
                await self._session_service.append_event(session, ev)
            if False:  # make this an async generator that yields no events
                yield None

    monkeypatch.setattr(ex, "Runner", _FakeRunner)

    config = ExecutorConfig(
        model_name="gemini-2.0-flash",
        google_api_key="test-key",
    )
    return WorkflowExecutor(
        config=config,
        session_service=create_session_service(use_database=False),
        model=object(),  # never used; the chain is stubbed
    )


class TestEmptyDiscoveryGuard:
    async def test_empty_regions_emit_no_regions_error(self, monkeypatch):
        """Zero regions -> clean NoRegions error, never an empty RegionsReady."""
        executor = _make_executor(monkeypatch, regions=[])
        events = [
            e
            async for e in executor.discover(
                book_title="Asdkfj Nonexistent Title", author="Nobody"
            )
        ]

        assert not any(isinstance(e, RegionsReady) for e in events), (
            "must not emit an empty RegionsReady"
        )
        errors = [e for e in events if isinstance(e, WorkflowError)]
        assert len(errors) == 1
        assert errors[0].error_type == "NoRegions"
        assert any(isinstance(e, WorkflowComplete) for e in events)

    async def test_nonempty_regions_unaffected(self, monkeypatch):
        """Happy path: a real region set still yields RegionsReady, no error."""
        regions = [{"region_id": 1, "name": "Bath, England"}]
        executor = _make_executor(monkeypatch, regions=regions)

        events = [
            e
            async for e in executor.discover(
                book_title="Persuasion", author="Jane Austen"
            )
        ]

        regions_events = [e for e in events if isinstance(e, RegionsReady)]
        assert len(regions_events) == 1
        assert regions_events[0].regions == regions
        assert not any(isinstance(e, WorkflowError) for e in events)
        assert any(isinstance(e, WorkflowComplete) for e in events)
