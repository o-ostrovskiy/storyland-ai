"""
Logging and tracing utilities for StoryLand AI.
"""

import logging
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


class Tracer:
    """Simple tracer for tracking execution flow."""

    def __init__(self):
        self.logger = setup_logger("tracer")

    @contextmanager
    def trace(self, name: str, inputs: Optional[Dict[str, Any]] = None):
        """Trace an operation."""
        start_time = time.time()
        context = TraceContext(name)
        try:
            yield context
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Agent {name} completed in {duration_ms:.0f}ms")


class TraceContext:
    """Context for a traced operation."""

    def __init__(self, name: str):
        self.name = name
        self.metadata: Dict[str, Any] = {}
        self.output: Optional[Any] = None

    def add_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def set_output(self, output: Any):
        self.output = output


_tracer = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer
