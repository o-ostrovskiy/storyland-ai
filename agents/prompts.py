"""
Agent prompt versioning.

Loads agent instruction strings from versioned JSON files in agents/prompts/.
Use load_prompts(version) to get a full prompt set, then pass it to workflow
factory functions to control which prompts are used at runtime.

Usage:
    from agents.prompts import load_prompts

    prompts = load_prompts("v3")          # explicit version
    prompts = load_prompts()              # default (v3)
"""

import json
from dataclasses import dataclass
from pathlib import Path

# v3 (MYS-460): region_analyzer additionally emits country_code + primary_locality,
# the grounded fields the canonical place_key is minted from. Bumped here rather
# than edited into v2.json in place: a published version is immutable, and — see
# MYS-462 — an in-place JSON edit would NOT change the discovery cache
# fingerprint, so the flush would silently not happen. THIS constant's source text
# is hashed (core/cache_version.py), so the bump is part of what invalidates the
# stale, keyless entries.
CURRENT_PROMPT_VERSION = "v3"

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_cache: dict[str, "AgentPrompts"] = {}


@dataclass(frozen=True)
class AgentPrompts:
    trip_composer: str
    city_researcher: str
    city_formatter: str
    landmark_researcher: str
    landmark_formatter: str
    author_researcher: str
    author_formatter: str
    book_context_researcher: str  # template with {book_ref} and {search_hint} placeholders
    book_context_formatter: str
    reader_profile: str
    region_analyzer: str
    local_atmosphere_researcher: str  # template with {location_label} and {radius_km} placeholders
    local_atmosphere_formatter: str  # template with {location_label} and {radius_km} placeholders
    expansion_researcher: str  # template with {book_title}, {author}, {parent_city}, {action_prompt}, {existing_places}
    expansion_formatter: str  # template with {parent_city}, {action_prompt}
    book_recommendation_researcher: str  # template with {book_title}, {author}, {destinations}, {themes}
    book_recommendation_formatter: str   # template with {book_title}, {author}, {destinations}, {themes}
    place_to_book_researcher: str  # template with {place}
    place_to_book_formatter: str  # template with {place}


def book_facts_block(
    book_title: str,
    author: str,
    vibe: str | None = None,
    taste_context: dict | None = None,
) -> str:
    """Explicit book-facts header prepended to graph-scoped researcher instructions.

    Under the ADK 2 graph runtime (ADR #24) a node's conversation is scoped to
    its trigger chain: the city/landmark/author researchers receive only the
    BookContext emitted by their direct predecessor — which carries locations,
    period, and themes but NOT the exact title, author, or the reader's
    vibe/taste biasing that ride the discovery user prompt. On 1.x templates
    they saw all of it implicitly. This block restores those facts explicitly.

    Lives in THIS module deliberately: core/cache_version.py fingerprints
    ``agents.prompts`` source, so any edit here flips the discovery cache
    namespace. Baking the block in the factory modules instead would change
    effective instructions WITHOUT invalidating cached discovery results —
    the MYS-222/MYS-462 class of error.
    """
    lines = [
        "BOOK FACTS (explicit — the conversation may not contain them):",
        f'Title: "{book_title}"',
        f"Author: {author}",
    ]
    if vibe:
        lines.append(f"Reader's requested vibe: {vibe}")
    if taste_context:
        titles = ", ".join(taste_context.get("titles") or [])
        moods = ", ".join(taste_context.get("moods") or [])
        if titles:
            lines.append(f"Reader also loved: {titles}")
        if moods:
            lines.append(f"Reader's preferred moods: {moods}")
    return "\n".join(lines) + "\n\n"


def load_prompts(version: str = CURRENT_PROMPT_VERSION) -> AgentPrompts:
    """
    Load agent prompts for a given version from agents/prompts/{version}.json.

    Results are cached in-process so repeated calls are free.

    Args:
        version: Prompt version identifier, e.g. "v1", "v2", "v3". Defaults to
                 CURRENT_PROMPT_VERSION ("v3").

    Returns:
        AgentPrompts dataclass with all agent instruction strings.

    Raises:
        FileNotFoundError: If agents/prompts/{version}.json does not exist.
    """
    if version not in _cache:
        path = _PROMPTS_DIR / f"{version}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt version '{version}' not found. "
                f"Expected file: {path}. "
                f"Available versions: {[p.stem for p in _PROMPTS_DIR.glob('*.json')]}"
            )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        agents_data = _inject_tone_guardrail(data["agents"])
        _cache[version] = AgentPrompts(**agents_data)
    return _cache[version]


# Agents that emit reader-facing explanation text ("why it fits" / reason /
# overview / summary). The rec-explanation tone guardrail clause is appended to
# each so the model never writes reader-directed judgement. Researcher agents
# (which only gather candidates) and pure formatters that don't author the
# explanation copy are intentionally excluded.
_EXPLANATION_AGENTS = (
    "trip_composer",
    "local_atmosphere_formatter",
    "expansion_formatter",
    "book_recommendation_formatter",
    "place_to_book_formatter",
)


def _inject_tone_guardrail(agents_data: dict) -> dict:
    """Append the reader-safety tone guardrail to explanation-producing prompts.

    Returns a shallow copy with the guardrail appended to each explanation
    agent's instruction. Idempotent: skips an instruction that already carries
    the clause (the cache also makes load_prompts a no-op after the first call).

    The guardrail constant is imported lazily here (not at module load) so the
    ``agents`` package never imports the heavy ``core`` package at import time,
    avoiding an agents<->core import cycle. By the time load_prompts() first
    runs, all modules are fully initialized.
    """
    from core.guardrails import READER_TONE_GUARDRAIL

    out = dict(agents_data)
    for name in _EXPLANATION_AGENTS:
        instruction = out.get(name)
        if isinstance(instruction, str) and READER_TONE_GUARDRAIL not in instruction:
            out[name] = instruction + READER_TONE_GUARDRAIL
    return out
