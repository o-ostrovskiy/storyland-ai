"""Unit tests for the bounded Gemini retry backoff (core/retry.py).

Guards the fix for the pathological ``exp_base=7`` schedule: the shared model
is used by every workflow phase, each wrapped in ``workflow_timeout`` (300s),
so the worst-case retry schedule must finish well inside that budget.
"""

import os

import pytest

from common.config import _env_float, load_config
from core.retry import (
    RETRY_STATUS_CODES,
    build_retry_options,
    worst_case_backoff_seconds,
)
from core.types import ExecutorConfig

WORKFLOW_TIMEOUT_DEFAULT = 300


class TestWorstCaseBackoff:
    def test_defaults_sum_well_under_workflow_timeout(self):
        # Default bounded schedule: attempts=4, exp_base=2, initial=1, max=12.
        total = worst_case_backoff_seconds(
            attempts=4, exp_base=2.0, initial_delay=1.0, max_delay=12.0
        )
        # 1 + 2 + 4 = 7s over the 3 retries.
        assert total == pytest.approx(7.0)
        assert total < WORKFLOW_TIMEOUT_DEFAULT

    def test_executor_config_defaults_are_bounded(self):
        cfg = ExecutorConfig(model_name="m", google_api_key="k")
        total = worst_case_backoff_seconds(
            attempts=cfg.retry_attempts,
            exp_base=cfg.retry_exp_base,
            initial_delay=1.0,
            max_delay=cfg.retry_max_delay,
        )
        # Far under the interactive budget (and the per-agent timeout of 60s).
        assert total < cfg.workflow_timeout
        assert total < 60

    def test_max_delay_caps_the_schedule(self):
        # Without a cap the 4th-ish delay explodes; the cap holds each term.
        capped = worst_case_backoff_seconds(
            attempts=6, exp_base=2.0, initial_delay=1.0, max_delay=12.0
        )
        # 1 + 2 + 4 + 8 + 12(capped from 16) = 27, not 1+2+4+8+16 = 31.
        assert capped == pytest.approx(27.0)

    def test_pathological_legacy_config_would_blow_the_budget(self):
        # Regression documentation: the OLD config (exp_base=7, attempts=5,
        # effectively no cap) exceeds workflow_timeout — which is the bug.
        legacy = worst_case_backoff_seconds(
            attempts=5, exp_base=7.0, initial_delay=1.0, max_delay=float("inf")
        )
        # 1 + 7 + 49 + 343 = 400s > 300s wall.
        assert legacy == pytest.approx(400.0)
        assert legacy > WORKFLOW_TIMEOUT_DEFAULT


class TestBuildRetryOptions:
    def test_builds_with_expected_fields(self):
        opts = build_retry_options(
            attempts=4, exp_base=2.0, initial_delay=1.0, max_delay=12.0
        )
        assert opts.attempts == 4
        assert opts.exp_base == 2.0
        assert opts.max_delay == 12.0
        assert opts.initial_delay == 1.0
        assert list(opts.http_status_codes) == [429, 500, 503, 504]

    def test_status_codes_unchanged_from_legacy(self):
        assert RETRY_STATUS_CODES == [429, 500, 503, 504]

    def test_executor_and_eval_runner_use_identical_config(self):
        """Parity: both call sites build options from the same defaults."""
        cfg = ExecutorConfig(model_name="m", google_api_key="k")
        executor_opts = build_retry_options(
            attempts=cfg.retry_attempts,
            exp_base=cfg.retry_exp_base,
            initial_delay=1.0,
            max_delay=cfg.retry_max_delay,
        )
        # The eval runner reads the same three values from common.config.Config,
        # whose defaults match ExecutorConfig's.
        eval_opts = build_retry_options(
            attempts=cfg.retry_attempts,
            exp_base=cfg.retry_exp_base,
            initial_delay=1.0,
            max_delay=cfg.retry_max_delay,
        )
        assert (executor_opts.attempts, executor_opts.exp_base, executor_opts.max_delay) == (
            eval_opts.attempts,
            eval_opts.exp_base,
            eval_opts.max_delay,
        )


class TestEnvDrivenConfig:
    def test_env_float_default_and_override(self, monkeypatch):
        monkeypatch.delenv("RETRY_EXP_BASE", raising=False)
        assert _env_float("RETRY_EXP_BASE", 2.0) == 2.0
        monkeypatch.setenv("RETRY_EXP_BASE", "3.5")
        assert _env_float("RETRY_EXP_BASE", 2.0) == 3.5

    def test_load_config_reads_retry_overrides(self, monkeypatch):
        base_env = {
            "GOOGLE_API_KEY": "k",
            "USE_DATABASE": "false",
            "SESSION_MAX_EVENTS": "100",
            "MAX_CONTEXT_TOKENS": "1000",
            "MODEL_NAME": "gemini-2.5-flash-lite",
            "WORKFLOW_TIMEOUT": "300",
            "AGENT_TIMEOUT": "60",
            "LOG_LEVEL": "INFO",
            "ENABLE_ADK_DEBUG": "false",
        }
        for k, v in base_env.items():
            monkeypatch.setenv(k, v)

        # Defaults when unset.
        for k in ("RETRY_EXP_BASE", "RETRY_MAX_DELAY", "RETRY_ATTEMPTS"):
            monkeypatch.delenv(k, raising=False)
        cfg = load_config()
        assert cfg.retry_exp_base == 2.0
        assert cfg.retry_max_delay == 12.0
        assert cfg.retry_attempts == 4

        # Overrides honored.
        monkeypatch.setenv("RETRY_EXP_BASE", "2")
        monkeypatch.setenv("RETRY_MAX_DELAY", "8")
        monkeypatch.setenv("RETRY_ATTEMPTS", "3")
        cfg2 = load_config()
        assert cfg2.retry_exp_base == 2.0
        assert cfg2.retry_max_delay == 8.0
        assert cfg2.retry_attempts == 3
        # Even the override schedule stays bounded.
        assert worst_case_backoff_seconds(
            attempts=cfg2.retry_attempts,
            exp_base=cfg2.retry_exp_base,
            initial_delay=1.0,
            max_delay=cfg2.retry_max_delay,
        ) < cfg2.workflow_timeout


class TestExecutorConfigPlumbing:
    def test_from_config_carries_retry_fields(self):
        class _Cfg:
            model_name = "m"
            google_api_key = "k"
            workflow_timeout = 300
            database_url = None
            use_database = False
            langfuse_secret_key = None
            langfuse_public_key = None
            langfuse_host = None
            environment = "local"
            cache_ttl_seconds = 86400
            cache_max_entries = 500
            retry_exp_base = 2.0
            retry_max_delay = 12.0
            retry_attempts = 4

        ec = ExecutorConfig.from_config(_Cfg())
        assert ec.retry_exp_base == 2.0
        assert ec.retry_max_delay == 12.0
        assert ec.retry_attempts == 4

    def test_from_config_falls_back_when_fields_absent(self):
        class _OldCfg:
            model_name = "m"
            google_api_key = "k"
            workflow_timeout = 300
            database_url = None
            use_database = False
            langfuse_secret_key = None
            langfuse_public_key = None
            langfuse_host = None
            environment = "local"
            cache_ttl_seconds = 86400
            cache_max_entries = 500

        ec = ExecutorConfig.from_config(_OldCfg())
        assert ec.retry_exp_base == 2.0
        assert ec.retry_attempts == 4
