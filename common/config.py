"""
Configuration management for StoryLand AI.

Loads configuration from environment variables and provides centralized
access to application settings.
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Application configuration."""

    # API Keys
    google_api_key: str

    # Database
    database_url: Optional[str] = None
    use_database: bool = False

    # Session Management
    session_max_events: int = 20

    # Context Management
    max_context_tokens: int = 30000

    # Model Configuration
    model_name: str = "gemini-2.0-flash-lite"

    # Logging
    log_level: str = "INFO"
    enable_adk_debug: bool = False  # Enable DEBUG for ADK internal loggers


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Environment variables:
        - GOOGLE_API_KEY (required): Google API key
        - DATABASE_URL (optional): Database connection string
        - USE_DATABASE (optional): "true" or "false" (default: false)
        - SESSION_MAX_EVENTS (optional): Max events in session (default: 20)
        - MAX_CONTEXT_TOKENS (optional): Max tokens for context (default: 30000)
        - MODEL_NAME (optional): Model to use (default: gemini-2.0-flash-lite)
        - LOG_LEVEL (optional): Logging level (default: INFO)
        - ENABLE_ADK_DEBUG (optional): Enable DEBUG for ADK loggers (default: false)

    Returns:
        Config object

    Raises:
        ValueError: If GOOGLE_API_KEY is not set
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is required. "
            "Please set it in your .env file or environment."
        )

    return Config(
        google_api_key=google_api_key,
        database_url=os.getenv("DATABASE_URL"),
        use_database=os.getenv("USE_DATABASE", "false").lower() == "true",
        session_max_events=int(os.getenv("SESSION_MAX_EVENTS", "20")),
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "30000")),
        model_name=os.getenv("MODEL_NAME", "gemini-2.0-flash-lite"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        enable_adk_debug=os.getenv("ENABLE_ADK_DEBUG", "false").lower() == "true",
    )


def get_config() -> Config:
    """Get the application configuration."""
    return load_config()
