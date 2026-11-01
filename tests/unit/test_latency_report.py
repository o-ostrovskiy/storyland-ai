"""scripts/latency_report.py (storyland-ai-private#65): the two reader waits from
recorded /v2/metrics answers, and the three failure shapes (an API error, an
empty 200, a malformed body)."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from langfuse.api.core import ApiError

from scripts import latency_report as lr

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "langfuse_latency_metrics.json").read_text())
TODAY = date(2026, 10, 31)  # the fixture was recorded for the 7 UTC days ending here


class FakeAPI:
    """Answers each query from `answers[name]["daily" | "pooled"]` and records the queries."""

    def __init__(self, answers=None, error=None):
        self.answers, self.error, self.queries = answers, error, []
        self.metrics = SimpleNamespace(metrics=self._metrics)

    def _metrics(self, *, query):
        q = json.loads(query)
        self.queries.append(q)
        if self.error:
            raise self.error
        name = next(f["value"] for f in q["filters"] if f["column"] == "name")
        return SimpleNamespace(data=self.answers[name]["daily" if "timeDimension" in q else "pooled"])


def run(api, capsys, days=7):
    code = lr.main(["--days", str(days)], api=api, today=TODAY)
    return code, capsys.readouterr().out.splitlines()


def test_recorded_week_reports_every_day_and_says_too_few(capsys):
    code, out = run(FakeAPI(FIXTURE), capsys)
    assert code == 0
    assert out[0].startswith("Reader waits, production, 2026-10-25 … 2026-10-31")
    region = out[out.index("first region (book_to_place_discovery_invocation):") + 1 :][:8]
    assert region[0] == "  2026-10-25  too few (n=1)"
    assert region[3] == "  2026-10-28  too few (n=4)"
    # The recorded 10-31 row is count 0: an empty day is "no data", never 0 s.
    assert region[6] == "  2026-10-31  no data"
    assert region[7] == "  7 days   too few (n=12)"
    assert out[-1] == "  7 days   too few (n=10)"


def test_enough_samples_prints_p50_and_p95_in_seconds(capsys):
    answers = json.loads(json.dumps(FIXTURE))
    for name in lr.WAITS.values():
        answers[name]["pooled"] = [{"count_count": 25, "p50_latency": 15044, "p95_latency": 16621.8}]
    _, out = run(FakeAPI(answers), capsys)
    assert out[-1] == "  7 days   p50 15.0 s · p95 16.6 s (n=25)"


def test_queries_are_root_only_production_and_v2_shaped(capsys):
    api = FakeAPI(FIXTURE)
    run(api, capsys)
    assert len(api.queries) == 4  # daily + pooled, per wait
    for q in api.queries:
        assert q["view"] == "observations"
        assert {"column": "isRootObservation", "operator": "=", "value": True, "type": "boolean"} in q["filters"]
        assert {"column": "environment", "operator": "=", "value": "production", "type": "string"} in q["filters"]
        assert q["fromTimestamp"] == "2026-10-25T00:00:00+00:00"
        assert q["toTimestamp"] == "2026-11-01T00:00:00+00:00"
    names = {f["value"] for q in api.queries for f in q["filters"] if f["column"] == "name"}
    assert names == set(lr.WAITS.values())  # never the `invoke_workflow` duplicate root


def test_empty_200_is_no_data_never_zero_seconds(capsys):
    empty = {name: {"daily": [], "pooled": []} for name in lr.WAITS.values()}
    code, out = run(FakeAPI(empty), capsys)
    assert code == 0
    assert "  2026-10-27  no data" in out
    assert out[-1] == "  7 days   no data"
    assert not any(" 0.0 s" in line for line in out)


def test_api_error_is_unverified(capsys):
    code, out = run(FakeAPI(error=ApiError(status_code=503, body="unavailable")), capsys)
    assert code == 2
    assert out == ["latency unverified: /v2/metrics failed: ApiError: " + str(ApiError(status_code=503, body="unavailable"))[:160]]


@pytest.mark.parametrize(
    "rows",
    [
        [{"time_dimension": "2026-10-28", "p50_latency": 1}],  # no count
        [{"time_dimension": "2026-10-28", "count_count": "many", "p50_latency": 1, "p95_latency": 2}],
        ["not a row"],
    ],
    ids=["no-count", "non-numeric", "not-a-dict"],
)
def test_malformed_body_is_unverified(capsys, rows):
    answers = {name: {"daily": rows, "pooled": []} for name in lr.WAITS.values()}
    code, out = run(FakeAPI(answers), capsys)
    assert code == 2
    assert out[0].startswith("latency unverified: malformed metrics row")


def test_no_data_list_is_unverified(capsys):
    api = FakeAPI(FIXTURE)
    api.metrics = SimpleNamespace(metrics=lambda *, query: SimpleNamespace(data=None))
    assert run(api, capsys)[0] == 2


def test_enough_runs_without_latency_is_unverified(capsys):
    answers = json.loads(json.dumps(FIXTURE))
    for name in lr.WAITS.values():
        answers[name]["pooled"] = [{"count_count": 30, "p50_latency": None, "p95_latency": None}]
    assert run(FakeAPI(answers), capsys)[0] == 2


def test_missing_keys_is_unverified_without_a_call(monkeypatch, capsys):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert lr.main([], today=TODAY) == 2
    assert capsys.readouterr().out.startswith("latency unverified: LANGFUSE_PUBLIC_KEY")
