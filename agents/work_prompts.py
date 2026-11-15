"""
Prompt loading by kind of work (ADR #31, Movies epic story 1).

``for_kind(kind)`` is the seam the film and series prompts plug into (story 4).
Today every kind returns the book prompt set unchanged: this story adds the
seam, not film behaviour.

Why a separate module and not ``agents/prompts.py``: ``core/cache_version.py``
fingerprints the *source* of ``agents.prompts``, so an edit there flips the
discovery cache namespace and cold-starts every cached book. This seam changes
no prompt text, so it must not cost that. When story 4 makes this module
return different prompts, it adds ``agents.work_prompts`` to
``_PROMPT_MODULES`` in the same PR (one deliberate cold-cache window then,
when outputs really change).
"""

from typing import get_args

from agents.prompts import CURRENT_PROMPT_VERSION, AgentPrompts, load_prompts
from models.work import WorkKind

_KNOWN_KINDS = frozenset(get_args(WorkKind))


def for_kind(kind: str, version: str = CURRENT_PROMPT_VERSION) -> AgentPrompts:
    """Return the agent prompt set for a kind of work.

    ``"book"`` returns ``load_prompts(version)`` unchanged. ``"film"`` and
    ``"series"`` return the same set until their prompts land (story 4).
    An unknown kind is a programming error and raises ``ValueError``.
    """
    if kind not in _KNOWN_KINDS:
        raise ValueError(f"unknown work kind: {kind!r}")
    return load_prompts(version)
