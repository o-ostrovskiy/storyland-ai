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


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Environment variables:
        - GOOGLE_API_KEY (required): Google API key
        - DATABASE_URL (optional): Database connection string (e.g., sqlite:///sessions.db)
        - USE_DATABASE (optional): "true" or "false" (default: false)
        - SESSION_MAX_EVENTS (optional): Max events to keep in session (default: 20)
        - MAX_CONTEXT_TOKENS (optional): Max tokens for context (default: 30000)
        - MODEL_NAME (optional): Model to use (default: gemini-2.0-flash-lite)
        - LOG_LEVEL (optional): Logging level (default: INFO)

    Returns:
        Config object

    Raises:
        ValueError: If GOOGLE_API_KEY is not set

    Example:
        >>> config = load_config()
        >>> print(config.google_api_key)
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
    )


def get_config() -> Config:
    """
    Get the application configuration.

    This is a convenience function that calls load_config().

    Returns:
        Config object
    """
    return load_config()
