"""
Application state and dependency injection.

Holds the WorkflowExecutor (core SDK) initialized at startup and
available to all request handlers via get_app_state().
"""

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request

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


def _user_from_jwt(authorization: str | None) -> str | None:
    """Extract email/sub from a Bearer JWT payload without verifying the signature."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        import base64
        import json
        payload_b64 = authorization.split(" ", 1)[1].split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("email") or payload.get("sub") or None
    except Exception:
        return None


def get_gateway_user_id(
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
    authorization: str | None = Header(default=None),
) -> str:
    """
    Extract the authenticated user_id.

    Priority:
    1. X-User-ID header (set by the backend proxy after JWT validation)
    2. JWT payload from Authorization header (when frontend calls AI directly)
    3. 'dev_user' fallback for local development

    Raises HTTP 403 if INTERNAL_API_SECRET is configured but X-User-ID is absent.
    """
    secret = get_app_state().config.internal_api_secret
    if not secret:
        return x_user_id or _user_from_jwt(authorization) or "dev_user"
    if not x_user_id:
        raise HTTPException(status_code=403, detail="X-User-ID header is required")
    return x_user_id
