"""The two waits a reader feels, as p50/p95 per day from Langfuse (storyland-ai-private#65).

    python scripts/latency_report.py [--days 7] [--env production]

- first region: the ``book_to_place_discovery_invocation`` root. ``RegionsReady``
  is yielded when the discovery workflow ends (core/executor.py), so the root's
  latency is the wait before the reader sees regions.
- full journey: the ``book_to_place_composition_invocation`` root, compose → done.

Each trace has a second root, ``invoke_workflow <name>``; only the ``*_invocation``
root is counted. A discovery-cache hit runs no workflow and writes no trace, so
the numbers cover fresh runs only.

Read-only: one ``GET /api/public/v2/metrics`` per wait and window (the v2 API, not
the endpoints Langfuse retires on 2026-11-16), through the SDK's bare REST client,
so no tracer starts and nothing is written. Keys come from the usual
``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST``. No Gemini calls.

Fewer than ``MIN_SAMPLES`` runs in a row prints "too few", a day with no runs
prints "no data" (an empty answer is never 0 s). An API error or a malformed
body exits 2 with ``latency unverified``.
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, time, timedelta, timezone

WAITS = {
    "first region": "book_to_place_discovery_invocation",
    "full journey": "book_to_place_composition_invocation",
}
MIN_SAMPLES = 20


class Unverified(Exception):
    pass


def metrics_query(name: str, environment: str, start: datetime, end: datetime, daily: bool) -> dict:
    query = {
        "view": "observations",
        "metrics": [
            {"measure": "count", "aggregation": "count"},
            {"measure": "latency", "aggregation": "p50"},
            {"measure": "latency", "aggregation": "p95"},
        ],
        "dimensions": [],
        "filters": [
            {"column": "isRootObservation", "operator": "=", "value": True, "type": "boolean"},
            {"column": "name", "operator": "=", "value": name, "type": "string"},
            {"column": "environment", "operator": "=", "value": environment, "type": "string"},
        ],
        "fromTimestamp": start.isoformat(),
        "toTimestamp": end.isoformat(),
        "config": {"row_limit": 1000},
    }
    if daily:
        query["timeDimension"] = {"granularity": "day"}
    return query


def parse_row(row: object) -> tuple[int, float | None, float | None]:
    """(sample size, p50 ms, p95 ms) from one metrics row; Unverified if it isn't one."""
    if not isinstance(row, dict) or "count_count" not in row:
        raise Unverified(f"malformed metrics row: {str(row)[:120]}")
    try:
        n = int(row["count_count"] or 0)
        p50 = row.get("p50_latency")
        p95 = row.get("p95_latency")
        return n, None if p50 is None else float(p50), None if p95 is None else float(p95)
    except (TypeError, ValueError) as exc:
        raise Unverified(f"malformed metrics row: {str(row)[:120]}") from exc


def cell(n: int, p50: float | None, p95: float | None) -> str:
    if n == 0:
        return "no data"
    if n < MIN_SAMPLES:
        return f"too few (n={n})"
    if p50 is None or p95 is None:
        raise Unverified(f"n={n} but no latency in the row")
    return f"p50 {p50 / 1000:.1f} s · p95 {p95 / 1000:.1f} s (n={n})"


def fetch(api, query: dict) -> list:
    try:
        response = api.metrics.metrics(query=json.dumps(query))
    except Exception as exc:  # ApiError, a timeout, or a body the SDK could not parse
        raise Unverified(f"/v2/metrics failed: {type(exc).__name__}: {str(exc)[:160]}") from exc
    rows = getattr(response, "data", None)
    if not isinstance(rows, list):
        raise Unverified("/v2/metrics answered without a data list")
    return rows


def report(api, days: int, environment: str, today: date) -> list[str]:
    """The report's lines, oldest day first, the pooled window last."""
    end = datetime.combine(today + timedelta(days=1), time(), timezone.utc)
    start = end - timedelta(days=days)
    day_keys = [(start + timedelta(days=i)).date().isoformat() for i in range(days)]
    lines = [f"Reader waits, {environment}, {day_keys[0]} … {day_keys[-1]} (UTC days; fresh runs only, cache hits make no trace)"]
    for wait, name in WAITS.items():
        by_day: dict[str, tuple[int, float | None, float | None]] = {}
        for row in fetch(api, metrics_query(name, environment, start, end, daily=True)):
            parsed = parse_row(row)
            day = str(row.get("time_dimension") or "")[:10]
            if day in day_keys and parsed[0]:
                by_day[day] = parsed
        lines.append(f"{wait} ({name}):")
        lines.extend(f"  {day}  {cell(*by_day.get(day, (0, None, None)))}" for day in day_keys)
        pooled = fetch(api, metrics_query(name, environment, start, end, daily=False))
        if len(pooled) > 1:
            raise Unverified(f"pooled query returned {len(pooled)} rows, expected 1")
        lines.append(f"  {days} days   {cell(*(parse_row(pooled[0]) if pooled else (0, None, None)))}")
    return lines


def make_api():
    from langfuse.api.client import LangfuseAPI

    public, secret = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
    if not (public and secret):
        raise Unverified("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
    host = (os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")
    return LangfuseAPI(base_url=host, username=public, password=secret, timeout=60)


def main(argv: list[str] | None = None, api=None, today: date | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--env", default="production")
    args = parser.parse_args(argv)
    try:
        lines = report(api or make_api(), args.days, args.env, today or datetime.now(timezone.utc).date())
    except Unverified as exc:
        print(f"latency unverified: {exc}")
        return 2
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
