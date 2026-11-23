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

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# v3 (MYS-460): region_analyzer additionally emits country_code + primary_locality,
# the grounded fields the canonical place_key is minted from. Bumped here rather
# than edited into v2.json in place: a published version is immutable. (The
# discovery cache fingerprint hashes both this constant's source text and the
# active JSON's content — see prompt_content_hash() and MYS-462 — so either a
# bump or an in-place edit invalidates stale entries.)
# v4 (ai-private#71): v3 + the place_to_book researcher/formatter revised (literal
# only inside the place typed, invented places come back empty, each book once,
# the real author, queries for regions and countries). Also moves the
# place2book:v1:<version> cache key, so no v3 answer is served under v4.
# v5 (Luga, 2026-11-21): v4 + the place_to_book researcher/formatter revised so
# a small real town is never an early exit: an empty answer only after book
# searches (plus a region/country/history search for small towns), and "5" is a
# target for famous places, not a pass mark. Moves place2book:v1:<version>, so
# the cached Luga not-found (and its kind) are not served under v5.
CURRENT_PROMPT_VERSION = "v5"

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


def preferences_block(preferences: dict | None) -> str:
    """Explicit reader-preferences block for graph-scoped agent instructions.

    Appending preferences to a flow's initial user prompt only reaches the
    FIRST node of the graph (trigger-chain scoping, ADR #24); any agent
    deeper in the chain needs them baked into its instruction. Empty when no
    preferences were supplied — the instruction is byte-identical to the
    no-preferences build. Lives in this cache-fingerprinted module for the
    same reason as book_facts_block.
    """
    if not preferences:
        return ""
    import json

    return (
        "\n\nREADER PREFERENCES — honor these when choosing places and "
        "pacing:\n" + json.dumps(preferences)
    )


def prompt_content_hash(version: str = CURRENT_PROMPT_VERSION) -> str:
    """Short hash of a prompt version's JSON *content* (MYS-462).

    The version label alone can't distinguish an in-place edit of vN.json from
    the published vN, so this is what both the discovery cache fingerprint
    (core/cache_version.py) and the Langfuse trace metadata carry. Best-effort:
    returns '' if the file can't be read, never raises.
    """
    try:
        data = (_PROMPTS_DIR / f"{version}.json").read_bytes()
    except OSError:
        return ""
    return hashlib.sha1(data).hexdigest()[:12]


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
        agents_data = _inject_clarity_rules(_inject_tone_guardrail(data["agents"]))
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


# Agents that write the one-line "connection" / "relevance" text on a city or
# landmark ("why this place is linked to the book"). They get the clarity rules
# below but not the tone guardrail, which was scoped to the explanation agents.
_CONNECTION_AGENTS = (
    "city_formatter",
    "landmark_formatter",
)

# Clarity rules for every reader-facing explanation (2026-11-22). Readers found
# some "why this place is connected" lines hard to follow: packed noun strings,
# unclear "it"/"this", the link to the book implied rather than stated. Adapted
# from ksoomro's "write for human understanding" rules (TLDR community thread,
# distilled from arXiv 2610.29162) for a literary-travel reader.
#
# Lives in THIS module, not core/guardrails, on purpose: core/cache_version.py
# fingerprints agents.prompts source, so editing this text flushes cached
# discovery results instead of serving old wording (the MYS-462 class of error).
# No braces in the text: several instructions are str.format templates.
READER_CLARITY_RULES = (
    "\n\n## WRITE FOR THE READER'S UNDERSTANDING (explanation text)\n"
    "Every reader-facing sentence you write (connection, relevance, reason, "
    "why it fits, overview, summary, description) must be understood on one "
    "read by a curious traveler who may not know the book well. A few extra "
    "words are worth it when they make the link clearer.\n"
    "- Say the link first and say it plainly: what happens in the book at this "
    "place, who is there, or why the author wrote about it. Then, if there is "
    "room, why it is worth seeing today.\n"
    "- Name the actors. Use the character's, author's or book's name instead of "
    "an unclear \"it\", \"this\" or \"they\".\n"
    "- Keep the subject next to its verb. Split a sentence that carries more "
    "than one idea instead of nesting clauses inside it.\n"
    "- Unpack noun stacks (\"Victorian dockside smuggling-scene backdrop\") "
    "into a phrase that shows the relationship (\"the docks where the smugglers "
    "meet in chapter 3\"). Keep small connecting words such as \"that\", "
    "\"where\" and \"because\".\n"
    "- Prefer familiar, precise words over unusual word pairs, invented verbs "
    "or places that \"whisper\" and \"breathe\". A metaphor is fine only when it "
    "makes the point clearer.\n"
    "- Explain a needed literary or historical term in a few words the first "
    "time it appears.\n"
    "- Keep every fact, qualification and hedge from the research (\"inspired "
    "by\", \"likely\", \"the author never named it\"). Add nothing the research "
    "does not support. If the link is loose, say so honestly rather than "
    "dressing it up.\n"
    "- Respect every length limit in these instructions (for example \"one "
    "sentence\"), but never compress so far that the link becomes ambiguous.\n"
    "- Plain prose inside text fields: no markdown, bullets or headings there. "
    "American spelling.\n"
    "Before returning, reread each explanation as that traveler: could they "
    "tell, without rereading or guessing, how this place is tied to the book?"
)


def _inject_clarity_rules(agents_data: dict) -> dict:
    """Append READER_CLARITY_RULES to every agent that writes reader-facing text.

    Covers the explanation agents (which also get the tone guardrail) and the
    city/landmark formatters that write the "connection" line. Idempotent, and
    returns a shallow copy like _inject_tone_guardrail.
    """
    out = dict(agents_data)
    for name in _EXPLANATION_AGENTS + _CONNECTION_AGENTS:
        instruction = out.get(name)
        if isinstance(instruction, str) and READER_CLARITY_RULES not in instruction:
            out[name] = instruction + READER_CLARITY_RULES
    return out


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
