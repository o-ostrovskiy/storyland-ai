"""Composer envelope truncation: detect it at the model boundary, never recover half of it.

The composer intermittently emitted an envelope whose JSON stopped mid-string
(issue ai-private#33, Defect 1). Two things went wrong with that response:

* Nothing read the finish reason, so a length-capped answer looked like any
  other answer until something tried to parse it.
* The text fallback (``core.extraction.extract_json_from_text``) slices
  ``{ … rfind('}')`` and can parse a *different*, smaller object out of a cut-off
  envelope, which is the "silently half-recovered" itinerary the card bans.

``detect_composer_truncation`` is the composer's ``after_model_callback``. It
runs before ADK writes ``output_key``, so a truncated envelope raises a typed
``ComposerEnvelopeTruncated`` before it can reach session state or the slicer.
The executor retries the composer exactly once on it (``COMPOSER_ATTEMPTS``).
A second failure propagates to ``run_guarded``, which maps it to the existing
client-safe ``DiscoveryComposeError(kind="transient")``.
"""

from __future__ import annotations

from typing import Optional

from google.genai import types
from pydantic import ValidationError

from common.logging import get_logger
from models.itinerary import ComposerEnvelope

# Deliberately no `core` import: core/__init__ imports the executor, which
# imports agents.orchestrator, which imports the composer agent that imports
# this module. The two helpers below are the only things it needed from core.

logger = get_logger("storyland.agents.composer_truncation")

# Stable event name. The eval report and log searches key on this exact string.
TRUNCATION_EVENT = "composer_envelope_truncated"

# One try plus exactly one retry. A second truncation is a clean error.
COMPOSER_ATTEMPTS = 2


class ComposerEnvelopeTruncated(Exception):
    """The composer's envelope was cut off (length finish) or is not a valid envelope."""

    def __init__(self, finish_reason: Optional[str], output_length: int) -> None:
        self.finish_reason = finish_reason
        self.output_length = output_length
        super().__init__(
            f"{TRUNCATION_EVENT}: finish_reason={finish_reason} "
            f"output_length={output_length}"
        )


def _is_valid_envelope(text: str) -> bool:
    """Same contract as core.extraction.validate_composer_envelope on a JSON string."""
    try:
        ComposerEnvelope.model_validate_json(text)
    except ValidationError:
        return False
    return True


def _response_text(llm_response) -> str:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(
        p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)
    )


def detect_composer_truncation(callback_context, llm_response):
    """``after_model_callback``: raise on a truncated envelope, else pass through.

    Returns ``None`` (keep the response as-is) for a valid envelope, a streaming
    partial chunk, or a response with no text. Empty text is left to the
    existing ``ExtractionError`` path: there is nothing half-written to recover.
    """
    if getattr(llm_response, "partial", False):
        return None
    text = _response_text(llm_response)
    finish = getattr(llm_response, "finish_reason", None)
    length_capped = finish == types.FinishReason.MAX_TOKENS
    if not text.strip() and not length_capped:
        return None
    if not length_capped and _is_valid_envelope(text):
        return None
    finish_name = getattr(finish, "name", None) if finish is not None else None
    logger.warning(
        TRUNCATION_EVENT,
        finish_reason=finish_name,
        output_length=len(text),
        agent=getattr(callback_context, "agent_name", None),
    )
    raise ComposerEnvelopeTruncated(finish_name, len(text))


def find_composer_truncation(exc: BaseException) -> Optional[ComposerEnvelopeTruncated]:
    """The truncation inside ``exc``, which may arrive wrapped in a TaskGroup group."""
    # Local import: ``core/__init__`` pulls in the executor, which imports this module.
    from core.discovery_errors import _iter_leaves

    for leaf in _iter_leaves(exc):
        if isinstance(leaf, ComposerEnvelopeTruncated):
            return leaf
    return None
