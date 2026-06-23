"""
Configuration management for StoryLand AI.

All configuration loaded from environment variables - no defaults.
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Application configuration - all values from environment."""

    google_api_key: str
    database_url: Optional[str]
    use_database: bool
    session_max_events: int
    max_context_tokens: int
    model_name: str
    workflow_timeout: int
    agent_timeout: int
    log_level: str
    enable_adk_debug: bool
    langfuse_secret_key: Optional[str]
    langfuse_public_key: Optional[str]
    langfuse_host: Optional[str]
    environment: str
    internal_api_secret: str
    # Result cache for the Discovery chain (always on; validated in prod 2026-06-17).
    cache_ttl_seconds: int
    cache_max_entries: int
    # Bounded Gemini retry backoff (keeps worst-case schedule < workflow_timeout).
    retry_exp_base: float
    retry_max_delay: float
    retry_attempts: int
    # Book-recommendation floor: minimum recommendations the formatter may
    # return (default 3) so a thin researcher result never forces an invented
    # 5th book. Tunable via REC_MIN_RESULTS.
    rec_min_results: int
    # Load-shedding guards for the expensive discovery chain (api/ratelimit.py).
    # Both DISABLED by default (value <= 0) so prod behaviour is unchanged until
    # an operator opts in; additive and reversible.
    rate_limit_requests: int
    rate_limit_window_seconds: int
    max_inflight_requests: int


def _require_env(key: str) -> str:
    """Get required environment variable or raise error."""
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"{key} environment variable is required.")
    return value


def _require_env_int(key: str) -> int:
    """Get required integer environment variable."""
    return int(_require_env(key))


def _require_env_bool(key: str) -> bool:
    """Get required boolean environment variable."""
    return _require_env(key).lower() == "true"


def _env_int(key: str, default: int) -> int:
    """Get optional integer environment variable with a default."""
    value = os.getenv(key)
    if value is None:
        return default
    return int(value)


def _env_float(key: str, default: float) -> float:
    """Get optional float environment variable with a default."""
    value = os.getenv(key)
    if value is None:
        return default
    return float(value)


def load_config() -> Config:
    """
    Load configuration from environment variables.

    All variables are required - set them in .env file.

    Required environment variables:
        - GOOGLE_API_KEY: Google API key
        - USE_DATABASE: "true" or "false"
        - SESSION_MAX_EVENTS: Max events in session
        - MAX_CONTEXT_TOKENS: Max tokens for context
        - MODEL_NAME: Model to use
        - WORKFLOW_TIMEOUT: Max seconds for workflow
        - AGENT_TIMEOUT: Max seconds per agent
        - LOG_LEVEL: Logging level
        - ENABLE_ADK_DEBUG: Enable DEBUG for ADK loggers

    Optional:
        - DATABASE_URL: Database connection string
        - LANGFUSE_SECRET_KEY: Langfuse secret key
        - LANGFUSE_PUBLIC_KEY: Langfuse public key
        - LANGFUSE_HOST: Langfuse host URL
        - ENVIRONMENT: deployment environment tag (default: "local")
        - RATE_LIMIT_REQUESTS: max requests per window per user/IP on the
          expensive endpoints; 0 disables (default: 0)
        - RATE_LIMIT_WINDOW_SECONDS: rate-limit window length (default: 60)
        - MAX_INFLIGHT_REQUESTS: max concurrent in-flight heavy requests;
          0 disables (default: 0)

    Returns:
        Config object

    Raises:
        ValueError: If any required variable is not set
    """
    return Config(
        google_api_key=_require_env("GOOGLE_API_KEY"),
        database_url=os.getenv("DATABASE_URL"),
        use_database=_require_env_bool("USE_DATABASE"),
        session_max_events=_require_env_int("SESSION_MAX_EVENTS"),
        max_context_tokens=_require_env_int("MAX_CONTEXT_TOKENS"),
        model_name=_require_env("MODEL_NAME"),
        workflow_timeout=_require_env_int("WORKFLOW_TIMEOUT"),
        agent_timeout=_require_env_int("AGENT_TIMEOUT"),
        log_level=_require_env("LOG_LEVEL").upper(),
        enable_adk_debug=_require_env_bool("ENABLE_ADK_DEBUG"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_host=os.getenv("LANGFUSE_HOST"),
        environment=os.getenv("ENVIRONMENT", "local"),
        internal_api_secret=os.getenv("INTERNAL_API_SECRET", ""),
        cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", 86400),
        cache_max_entries=_env_int("CACHE_MAX_ENTRIES", 500),
        retry_exp_base=_env_float("RETRY_EXP_BASE", 2.0),
        retry_max_delay=_env_float("RETRY_MAX_DELAY", 12.0),
        retry_attempts=_env_int("RETRY_ATTEMPTS", 4),
        rec_min_results=_env_int("REC_MIN_RESULTS", 3),
        rate_limit_requests=_env_int("RATE_LIMIT_REQUESTS", 0),
        rate_limit_window_seconds=_env_int("RATE_LIMIT_WINDOW_SECONDS", 60),
        max_inflight_requests=_env_int("MAX_INFLIGHT_REQUESTS", 0),
    )


def get_config() -> Config:
    """Get the application configuration."""
    return load_config()
