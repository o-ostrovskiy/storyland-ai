"""MYS-172: state.book_metadata = ... was a silent no-op against ADK's
persisted session state -- in-place mutation of `session.state` does not
survive a fresh `get_session()` call; only `session_service.append_event()`
does. `core/executor.py`'s `discover()` and `local_atmosphere()` both wrote
metadata the old way. These tests drive the REAL code paths (not a mock of
SessionStateAccessor) and re-fetch state via a NEW `get_session()` call --
per the Eng Lead's tech plan, a test that reads back from the same in-memory
object the code just wrote to would pass today, defect and all, because the
bug is specifically about what a SECOND, independent read sees.

Each test fails the underlying agent workflow deliberately (by monkeypatching
the workflow factory to raise) immediately AFTER the metadata write and
BEFORE any real Gemini/Runner call -- so this never needs live model access,
while still exercising the exact write this ticket fixes.
"""

import pytest

from core.events import JobStarted
from core.types import ExecutorConfig
from services.session_service import create_session_service


class TestBookMetadataPersistsAcrossGetSession:
    async def test_discover_book_metadata_survives_a_fresh_get_session(
        self, monkeypatch
    ):
        import core.executor as ex

        def _boom(*args, **kwargs):
            raise AssertionError(
                "discovery workflow invoked -- metadata write must happen "
                "before this, and this test must not need a real Gemini call"
            )

        monkeypatch.setattr(ex, "create_book_to_place_discovery_workflow", _boom)
        monkeypatch.setattr(ex, "Runner", _boom)

        session_service = create_session_service(use_database=False)
        config = ExecutorConfig(
            model_name="gemini-2.0-flash", google_api_key="test-key"
        )
        executor = ex.WorkflowExecutor(
            config=config,
            session_service=session_service,
            model=object(),  # never reached; the workflow factory raises first
        )

        job_id = None
        async for event in executor.discover(
            book_title="1984", author="George Orwell"
        ):
            if isinstance(event, JobStarted):
                job_id = event.job_id
        assert job_id is not None, "discover() never emitted JobStarted"

        # The regression check: a SECOND, independent get_session() call,
        # not a read of the same object discover() already holds.
        refetched = await session_service.get_session(
            app_name=ex.APP_NAME, user_id="default", session_id=job_id
        )
        assert refetched.state["book_metadata"]["book_title"] == "1984"
        assert refetched.state["book_metadata"]["author"] == "George Orwell"
        # The exact truthiness api/routes.py::_derive_job_status checks to
        # report JobStatus.DISCOVERING instead of SEARCHING.
        assert bool(refetched.state.get("book_metadata"))

    async def test_local_atmosphere_book_metadata_survives_a_fresh_get_session(
        self, monkeypatch
    ):
        import core.executor as ex

        def _boom(*args, **kwargs):
            raise AssertionError(
                "local-atmosphere workflow invoked -- metadata write must "
                "happen before this, and this test must not need a real "
                "Gemini call"
            )

        monkeypatch.setattr(ex, "create_local_atmosphere_workflow", _boom)
        monkeypatch.setattr(ex, "Runner", _boom)

        session_service = create_session_service(use_database=False)
        config = ExecutorConfig(
            model_name="gemini-2.0-flash", google_api_key="test-key"
        )
        executor = ex.WorkflowExecutor(
            config=config,
            session_service=session_service,
            model=object(),
        )

        job_id = None
        async for event in executor.local_atmosphere(
            book_title="1984",
            author="George Orwell",
            location_label="New York, NY",
            lat=40.7128,
            lng=-74.0060,
        ):
            if isinstance(event, JobStarted):
                job_id = event.job_id
        assert job_id is not None, "local_atmosphere() never emitted JobStarted"

        refetched = await session_service.get_session(
            app_name=ex.APP_NAME, user_id="default", session_id=job_id
        )
        assert refetched.state["book_metadata"]["book_title"] == "1984"
        assert refetched.state["book_metadata"]["author"] == "George Orwell"
        assert bool(refetched.state.get("book_metadata"))
