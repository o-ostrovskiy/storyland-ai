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
    )


def get_config() -> Config:
    """Get the application configuration."""
    return load_config()
