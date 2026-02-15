"""
Application state and dependency injection.

Holds shared resources (config, model, session_service, langfuse_plugin)
initialized at startup and available to all request handlers via get_app_state().

Pattern mirrors main.py lines 234-298 initialization sequence.
"""

from dataclasses import dataclass
from typing import Optional

from google.genai import types
from google.adk.models.google_llm import Gemini

from common.config import Config, load_config
from common.logging import configure_logging, get_logger
from common.langfuse_init import initialize_langfuse
from services.session_service import create_session_service
from plugins.langfuse_plugin import LangfusePlugin


@dataclass
class AppState:
    """Shared application state initialized at startup."""

    config: Config
    model: Gemini
    session_service: object  # InMemorySessionService or DatabaseSessionService
    langfuse_plugin: LangfusePlugin


_app_state: Optional[AppState] = None


async def initialize() -> AppState:
    """
    Initialize application state.

    Creates config, model with retry options, session service, and Langfuse plugin.
    Mirrors the setup in main.py create_itinerary() lines 234-298.

    Returns:
        Initialized AppState
    """
    global _app_state

    config = load_config()
    configure_logging(
        level=config.log_level, enable_adk_debug=config.enable_adk_debug
    )
    logger = get_logger("storyland.api")

    # Initialize Langfuse tracing
    initialize_langfuse(
        secret_key=config.langfuse_secret_key,
        public_key=config.langfuse_public_key,
        host=config.langfuse_host,
    )

    # Configure model with retry (mirrors main.py lines 273-281)
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )
    model = Gemini(
        model=config.model_name,
        api_key=config.google_api_key,
        retry_options=retry_config,
    )

    # Session service
    session_service = create_session_service(
        connection_string=config.database_url,
        use_database=config.use_database,
    )

    # Langfuse plugin
    langfuse_plugin = LangfusePlugin(
        secret_key=config.langfuse_secret_key,
        public_key=config.langfuse_public_key,
        host=config.langfuse_host,
    )

    _app_state = AppState(
        config=config,
        model=model,
        session_service=session_service,
        langfuse_plugin=langfuse_plugin,
    )

    logger.info("api_initialized", model=config.model_name)
    return _app_state


async def shutdown() -> None:
    """Cleanup on application shutdown."""
    global _app_state
    if _app_state and _app_state.langfuse_plugin:
        await _app_state.langfuse_plugin.close()
    _app_state = None


def get_app_state() -> AppState:
    """
    Get the initialized application state.

    Raises:
        RuntimeError: If called before initialize()
    """
    if _app_state is None:
        raise RuntimeError("Application not initialized. Call initialize() first.")
    return _app_state
