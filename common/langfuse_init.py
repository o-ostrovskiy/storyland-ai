"""
Langfuse initialization for Google ADK tracing.

Configures OpenTelemetry instrumentation to capture all ADK agent interactions,
tool calls, and model completions and send them to Langfuse for observability.
"""

import os
from typing import Optional
from common.logging import get_logger

logger = get_logger("storyland.langfuse")

_instrumentation_enabled = False


def initialize_langfuse(
    secret_key: Optional[str] = None,
    public_key: Optional[str] = None,
    host: Optional[str] = None,
) -> bool:
    """
    Initialize Langfuse tracing for Google ADK.

    This function sets up OpenTelemetry instrumentation to capture all ADK
    agent interactions and send them to Langfuse for observability, debugging,
    and evaluation.

    Args:
        secret_key: Langfuse secret key (or set LANGFUSE_SECRET_KEY env var)
        public_key: Langfuse public key (or set LANGFUSE_PUBLIC_KEY env var)
        host: Langfuse host URL (or set LANGFUSE_HOST env var)

    Returns:
        True if Langfuse was successfully initialized, False otherwise

    Example:
        >>> from common.langfuse_init import initialize_langfuse
        >>> initialize_langfuse()  # Uses env vars from .env
    """
    global _instrumentation_enabled

    if _instrumentation_enabled:
        logger.debug("langfuse_already_initialized")
        return True

    # Get credentials from args or environment
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    host = host or os.getenv("LANGFUSE_HOST")

    # Check if all credentials are provided
    if not all([secret_key, public_key, host]):
        logger.info(
            "langfuse_not_configured",
            message="Langfuse credentials not found. Skipping tracing initialization. "
            "Set LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, and LANGFUSE_HOST to enable.",
        )
        return False

    try:
        # Import Langfuse client
        from langfuse import get_client

        # Initialize Langfuse client
        langfuse = get_client()

        # Verify authentication
        if not langfuse.auth_check():
            logger.error(
                "langfuse_auth_failed",
                message="Langfuse authentication failed. Check your credentials.",
            )
            return False

        logger.info("langfuse_authenticated", host=host)

        # Import and configure OpenTelemetry instrumentation
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor

        # Instrument Google ADK
        GoogleADKInstrumentor().instrument()

        _instrumentation_enabled = True
        logger.info(
            "langfuse_instrumentation_enabled",
            message="Langfuse tracing is now active for all ADK agents",
        )
        return True

    except ImportError as e:
        logger.error(
            "langfuse_import_error",
            error=str(e),
            message="Failed to import Langfuse. Install with: pip install langfuse openinference-instrumentation-google-adk",
        )
        return False
    except Exception as e:
        logger.error(
            "langfuse_initialization_error",
            error=str(e),
            message="Failed to initialize Langfuse tracing",
        )
        return False


def is_enabled() -> bool:
    """Check if Langfuse instrumentation is enabled."""
    return _instrumentation_enabled
