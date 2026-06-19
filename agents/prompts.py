"""
Agent prompt versioning.

Loads agent instruction strings from versioned JSON files in agents/prompts/.
Use load_prompts(version) to get a full prompt set, then pass it to workflow
factory functions to control which prompts are used at runtime.

Usage:
    from agents.prompts import load_prompts

    prompts = load_prompts("v2")          # explicit version
    prompts = load_prompts()              # default (v2)
"""

import json
from dataclasses import dataclass
from pathlib import Path

CURRENT_PROMPT_VERSION = "v2"

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


def load_prompts(version: str = CURRENT_PROMPT_VERSION) -> AgentPrompts:
    """
    Load agent prompts for a given version from agents/prompts/{version}.json.

    Results are cached in-process so repeated calls are free.

    Args:
        version: Prompt version identifier, e.g. "v1", "v2". Defaults to
                 CURRENT_PROMPT_VERSION ("v2").

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
        _cache[version] = AgentPrompts(**data["agents"])
    return _cache[version]
