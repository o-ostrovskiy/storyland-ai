"""
Prompt loading by kind of work (ADR #31, Movies epic stories 1 and 4).

``for_kind("book")`` returns ``load_prompts(version)`` unchanged: the book
path never reads the screen overlay, so its instructions are byte-identical
(pinned in ``tests/unit/test_work_kind.py``).

``for_kind("film")`` / ``for_kind("series")`` return the book set with the
screen overlay ``agents/prompts/screen_v1.json`` applied (ai-private#67 Part 2):
``replace`` swaps an agent's whole instruction (the context researcher and
formatter, the city, landmark and creator researchers), ``append`` adds a
section to the book text (the trip composer's filmed_here / set_here rules).
The screen path adds no agents, only different instructions.

Why the overlay is not in ``_PROMPT_MODULES`` (``core/cache_version.py``): that
fingerprint is global, so hashing the overlay there would cold-start every
cached BOOK for a change that only touches films. Instead a screen journey's
discovery cache key carries ``screen_prompt_hash()`` next to its kind
(``core/executor.py``), so an overlay edit re-keys films only.
"""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import get_args

from agents.prompts import CURRENT_PROMPT_VERSION, AgentPrompts, load_prompts
from models.work import DEFAULT_WORK_KIND, WorkKind

_KNOWN_KINDS = frozenset(get_args(WorkKind))
SCREEN_PROMPT_VERSION = "screen_v1"
_SCREEN_OVERLAY = Path(__file__).parent / "prompts" / f"{SCREEN_PROMPT_VERSION}.json"
_cache: dict[str, AgentPrompts] = {}


def _overlay() -> dict:
    with open(_SCREEN_OVERLAY, encoding="utf-8") as f:
        data = json.load(f)
    fields = {f.name for f in dataclasses.fields(AgentPrompts)}
    unknown = (set(data.get("replace", {})) | set(data.get("append", {}))) - fields
    if unknown:
        raise ValueError(f"screen overlay names unknown agents: {sorted(unknown)}")
    return data


def screen_prompt_hash() -> str:
    """Short hash of the screen overlay's content; '' if unreadable, never raises."""
    try:
        return hashlib.sha1(_SCREEN_OVERLAY.read_bytes()).hexdigest()[:12]
    except OSError:
        return ""


def for_kind(kind: str, version: str = CURRENT_PROMPT_VERSION) -> AgentPrompts:
    """Return the agent prompt set for a kind of work.

    ``"book"`` returns ``load_prompts(version)`` unchanged. ``"film"`` and
    ``"series"`` return it with the screen overlay applied.
    An unknown kind is a programming error and raises ``ValueError``.
    """
    if kind not in _KNOWN_KINDS:
        raise ValueError(f"unknown work kind: {kind!r}")
    base = load_prompts(version)
    if kind == DEFAULT_WORK_KIND:
        return base
    if version not in _cache:
        data = _overlay()
        changes = dict(data.get("replace", {}))
        for name, extra in data.get("append", {}).items():
            changes[name] = getattr(base, name) + extra
        _cache[version] = dataclasses.replace(base, **changes)
    return _cache[version]


def is_screen_kind(kind: str) -> bool:
    return kind in _KNOWN_KINDS and kind != DEFAULT_WORK_KIND
