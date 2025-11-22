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
