"""Unit tests for server-side funnel telemetry (core/telemetry.py)."""

import pytest

from core.events import RegionsReady, ItineraryReady, WorkflowError
from core.telemetry import (
    FunnelStage,
    FunnelTelemetry,
    SearchEntry,
    get_funnel_telemetry,
    track_funnel,
)


class _Recorder(FunnelTelemetry):
    """Telemetry that records emitted stages instead of logging."""

    def __init__(self):
        super().__init__(enabled=True)
        self.events = []

    def emit(self, stage, *, entry, user_id, latency_ms=None, **fields):
        self.events.append((stage, entry, user_id, latency_ms))


async def _gen(events):
    for e in events:
        yield e


@pytest.mark.asyncio
async def test_disabled_telemetry_is_noop():
    t = FunnelTelemetry(enabled=False)
    # Should not raise and should record nothing observable.
    t.emit(FunnelStage.SEARCH_SUBMITTED, entry=SearchEntry.BOOK, user_id="u")
    assert t.enabled is False


def test_get_funnel_telemetry_reads_config():
    class Cfg:
        analytics_enabled = True
        environment = "prod"

    t = get_funnel_telemetry(Cfg())
    assert t.enabled is True


def test_get_funnel_telemetry_defaults_when_missing():
    t = get_funnel_telemetry(object())
    assert t.enabled is False


@pytest.mark.asyncio
async def test_result_shown_on_non_empty_regions():
    rec = _Recorder()
    regions = RegionsReady(job_id="j", regions=[{"id": 1}], analysis_note="n")
    out = [
        e
        async for e in track_funnel(
            _gen([regions]),
            telemetry=rec,
            entry=SearchEntry.BOOK,
            user_id="u",
            result_types=(RegionsReady,),
            is_empty=lambda e: not getattr(e, "regions", None),
        )
    ]
    assert out == [regions]  # passthrough preserved
    stages = [e[0] for e in rec.events]
    assert stages == [FunnelStage.SEARCH_SUBMITTED, FunnelStage.RESULT_SHOWN]


@pytest.mark.asyncio
async def test_search_empty_on_empty_regions():
    rec = _Recorder()
    empty = RegionsReady(job_id="j", regions=[], analysis_note="")
    _ = [
        e
        async for e in track_funnel(
            _gen([empty]),
            telemetry=rec,
            entry=SearchEntry.BOOK,
            user_id="u",
            result_types=(RegionsReady,),
            is_empty=lambda e: not getattr(e, "regions", None),
        )
    ]
    assert rec.events[-1][0] == FunnelStage.SEARCH_EMPTY


@pytest.mark.asyncio
async def test_search_failed_on_workflow_error():
    rec = _Recorder()
    err = WorkflowError(message="boom", error_type="X", phase=None)
    _ = [
        e
        async for e in track_funnel(
            _gen([err]),
            telemetry=rec,
            entry=SearchEntry.PLACE,
            user_id="u",
            result_types=(ItineraryReady,),
            is_empty=lambda e: not getattr(e, "itinerary", None),
        )
    ]
    assert rec.events[-1][0] == FunnelStage.SEARCH_FAILED
    assert rec.events[-1][1] == SearchEntry.PLACE


@pytest.mark.asyncio
async def test_search_failed_on_exception_propagates():
    rec = _Recorder()

    async def _boom():
        if False:
            yield  # make it an async generator
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError):
        _ = [
            e
            async for e in track_funnel(
                _boom(),
                telemetry=rec,
                entry=SearchEntry.BOOK,
                user_id="u",
                result_types=(RegionsReady,),
                is_empty=lambda e: False,
            )
        ]
    assert rec.events[-1][0] == FunnelStage.SEARCH_FAILED
