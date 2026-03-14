"""
Application state and dependency injection.

Holds the WorkflowExecutor (core SDK) initialized at startup and
available to all request handlers via get_app_state().
"""

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request

from common.config import Config, load_config
from common.logging import configure_logging, get_logger
from common.langfuse_init import initialize_langfuse
from core.executor import WorkflowExecutor
from core.types import ExecutorConfig


@dataclass
class AppState:
    """Shared application state initialized at startup."""

    config: Config
    executor: WorkflowExecutor


_app_state: Optional[AppState] = None


async def initialize() -> AppState:
    """Initialize application state with WorkflowExecutor."""
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

    # Create executor with config
    executor_config = ExecutorConfig.from_config(config)
    executor = WorkflowExecutor(executor_config)

    _app_state = AppState(config=config, executor=executor)

    logger.info("api_initialized", model=config.model_name)
    return _app_state


async def shutdown() -> None:
    """Cleanup on application shutdown."""
    global _app_state
    if _app_state:
        await _app_state.executor.close()
    _app_state = None


def get_app_state() -> AppState:
    """Get the initialized application state."""
    if _app_state is None:
        raise RuntimeError("Application not initialized. Call initialize() first.")
    return _app_state


def verify_gateway_secret(request: Request) -> None:
    """Require X-Internal-Secret header when INTERNAL_API_SECRET is configured."""
    secret = get_app_state().config.internal_api_secret
    if secret and request.headers.get("X-Internal-Secret") != secret:
        raise HTTPException(status_code=403, detail="Forbidden")
