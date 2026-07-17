"""
Logging utilities for StoryLand AI.

Logging Levels:
    - DEBUG: Full LLM prompts, detailed troubleshooting
    - INFO: Agent lifecycle, tool execution
    - WARNING: Potential issues
    - ERROR: Critical failures
"""

import logging
import sys

import structlog


def _sentry_error_processor(logger, method_name, event_dict):
    """
    Forward error-level structlog events to Sentry.

    structlog here uses PrintLoggerFactory (straight to stdout), so the Sentry
    SDK's stdlib logging integration never sees these events — without this
    processor, handled workflow failures (caught + logger.error + streamed as
    a WorkflowError) would stay docker-log-only. When called inside an except
    block the live exception is captured with its traceback; otherwise the
    event message is captured. Every capture_* call is a no-op unless
    init_sentry() actually initialized the SDK (no SENTRY_DSN → no client),
    so local dev and CI still never report.
    """
    if method_name in ("error", "critical", "exception"):
        import sentry_sdk

        exc = sys.exc_info()[1]
        extras = {k: v for k, v in event_dict.items() if k != "event"}
        with sentry_sdk.new_scope() as scope:
            scope.set_context("structlog", extras)
            if exc is not None:
                sentry_sdk.capture_exception(exc)
            else:
                level = "critical" if method_name == "critical" else "error"
                sentry_sdk.capture_message(str(event_dict.get("event")), level=level)
    return event_dict


# ADK module loggers for fine-grained control
ADK_LOGGERS = [
    "google.adk",
    "google.adk.agents",
    "google.adk.runners",
    "google.adk.tools",
]


def configure_logging(level: str = "INFO", enable_adk_debug: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        enable_adk_debug: Enable DEBUG for ADK internal loggers
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Configure ADK loggers
    adk_level = logging.DEBUG if enable_adk_debug else log_level
    for logger_name in ADK_LOGGERS:
        logging.getLogger(logger_name).setLevel(adk_level)

    # Configure structlog with colored console output
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            _sentry_error_processor,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a standard Python logger (backwards compatible).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
