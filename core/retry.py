"""Bounded HTTP retry configuration for the shared Gemini model.

Centralizes retry-backoff construction so ``core.executor`` and the
evaluation runner stay in parity (a single source of truth for the retry
schedule).

Background: the shared Gemini model previously used ``exp_base=7`` with
``attempts=5``, which produces a worst-case backoff schedule of roughly
1s, 7s, 49s, 343s -> ~400s. Because every workflow phase runs inside
``workflow_timeout`` (300s by default), a 429/500/503 burst would sit in
multi-minute blind backoff until the timeout wall fired, surfacing a
"timed out" error to the user precisely when the API was throttling.

Using ``exp_base=2`` with an explicit ``max_delay`` keeps the worst-case
cumulative backoff far under ``workflow_timeout``, converting silent
timeouts into either a fast success or a clean, fast error.
"""

from __future__ import annotations

from google.genai import types

# Transient server / rate-limit status codes we retry on (unchanged).
RETRY_STATUS_CODES = [429, 500, 503, 504]


def worst_case_backoff_seconds(
    attempts: int,
    exp_base: float,
    initial_delay: float,
    max_delay: float,
) -> float:
    """Return the worst-case cumulative delay spent waiting between retries.

    Standard exponential backoff: the delay before retry ``i`` (0-indexed)
    is ``initial_delay * exp_base ** i``, capped at ``max_delay``. A run of
    ``attempts`` total attempts performs at most ``attempts - 1`` retries.
    """
    total = 0.0
    for i in range(max(attempts - 1, 0)):
        total += min(initial_delay * (exp_base ** i), max_delay)
    return total


def build_retry_options(
    *,
    attempts: int,
    exp_base: float,
    initial_delay: float,
    max_delay: float,
) -> "types.HttpRetryOptions":
    """Build a genai ``HttpRetryOptions`` from bounded backoff parameters.

    Single construction point shared by the executor and the eval runner so
    their retry behaviour can never silently diverge.
    """
    return types.HttpRetryOptions(
        attempts=attempts,
        exp_base=exp_base,
        initial_delay=initial_delay,
        max_delay=max_delay,
        http_status_codes=list(RETRY_STATUS_CODES),
    )
