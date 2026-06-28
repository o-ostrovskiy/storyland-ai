"""Rec-explanation tone guardrail.

Storyland's recommendations are AI-generated and *explained* ("why this place
fits this book"). This guardrail keeps every such explanation focused on the
book<->place **fit** and forbids the model from characterizing, psychoanalyzing,
grading, moralizing about, or inferring the identity of the **reader** or their
taste. It is purely downside-protective (cf. the Dec-2024 Fable AI blow-up where
an "explained" AI graded users' reading taste with bigoted commentary).

Two layers, one source of truth (this module):

1. ``READER_TONE_GUARDRAIL`` — a system-instruction clause + prohibited-pattern
   list appended to every explanation-producing agent prompt (see
   ``agents.prompts.load_prompts``). This is the primary control: the model is
   told not to write reader-directed judgements.

2. ``flag_reader_directed`` / ``sanitize_explanation`` — a deterministic,
   regex/string-rule checker (ZERO Gemini calls, safe to run on every request)
   that scans finalized explanation text and drops any reader-directed clause
   that slipped through, falling back to the fit-only explanation. Wired into the
   ``core.extraction`` finalization points (itinerary, expansion, book recs).

Conservative & fail-open by design — it must never make a result worse:
  * It only removes clauses that judge the *reader*; legitimate fit text about
    scenes, themes, history, mood, or geography (incl. benign second person like
    "you'll walk the same misty streets") is never touched.
  * If sanitizing would empty a required field, a neutral fit-only fallback is
    used instead of surfacing an empty or reader-grading line.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from common.logging import get_logger

logger = get_logger("storyland.core.guardrails.tone")


# --------------------------------------------------------------------------- #
# System-instruction clause (primary control). Appended to every
# explanation-producing agent prompt so the model never writes reader-directed
# judgement in the first place. Same module feeds the deterministic checker
# below, so the prompt and the check stay in lock-step.
# --------------------------------------------------------------------------- #
READER_TONE_GUARDRAIL = (
    "\n\n## READER-SAFETY GUARDRAIL (explanation tone)\n"
    "Every explanation you write — the \"why it fits\" / reason / overview / "
    "summary text — must describe ONLY why the place or book fits: its scenes, "
    "themes, history, mood, geography, or atmosphere. You must NEVER "
    "characterize, psychoanalyze, grade, rank, moralize about, or infer the "
    "identity, demographics, intelligence, politics, or worth of the reader or "
    "their taste.\n"
    "Prohibited — never write anything like these:\n"
    "- Verdicts on the reader's taste or character (\"your taste is…\", "
    "\"this says a lot about you\", \"you seem like…\").\n"
    "- Labels or inferences about who the reader is (\"as a sophisticated "
    "reader…\", \"readers like you…\", \"you're the kind of person who…\").\n"
    "- Any praise, criticism, or moral judgement aimed at the reader rather than "
    "the book<->place fit.\n"
    "Keep the focus on the book and the place. Second person is fine ONLY to "
    "describe what the reader will see or experience at the place — never to "
    "judge the reader."
)


# --------------------------------------------------------------------------- #
# Deterministic checker (secondary, zero-spend safety net).
#
# Each pattern targets a reader-DIRECTED judgement (a verdict/label/inference
# about the reader or their taste/identity). Patterns are deliberately narrow
# and high-precision to avoid stripping legitimate fit explanations that happen
# to use second person to invite the reader into the *place* ("you'll wander the
# foggy quays", "you can stand where the duel was fought").
# --------------------------------------------------------------------------- #
_READER_NOUNS = r"(?:taste|tastes|personality|character|soul|mind|intellect|reading habits?|reading|choices?|preferences?|picks?|selections?)"

_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    # "your taste is…", "your reading choices reveal…"
    (
        "reader_taste_verdict",
        re.compile(
            r"\byour\s+" + _READER_NOUNS
            + r"\b[^.!?]*\b(is|are|says?|reveals?|suggests?|shows?|tells?|reflects?|betrays?|screams?|means?)\b",
            re.IGNORECASE,
        ),
    ),
    # "you seem like…", "you sound like the type…", "you come across as…"
    (
        "reader_inference",
        re.compile(
            r"\byou\s+(?:seem|sound|appear|come across)\b[^.!?]*\b(?:to be|like|as)\b",
            re.IGNORECASE,
        ),
    ),
    # "you're the kind/sort/type of reader/person who…"
    (
        "reader_type_label",
        re.compile(
            r"\byou(?:'re| are)\s+(?:the\s+)?(?:kind|sort|type)\s+of\s+(?:reader|person|traveler|traveller|soul)\b",
            re.IGNORECASE,
        ),
    ),
    # "readers/people/travellers like you…"
    (
        "people_like_you",
        re.compile(
            r"\b(?:readers?|people|travel(?:l)?ers?|souls?|minds?)\s+like\s+you\b",
            re.IGNORECASE,
        ),
    ),
    # "this says/reveals/tells (a lot) about you/your taste"
    (
        "says_about_you",
        re.compile(
            r"\b(?:this|that|it)\s+(?:says|reveals|tells\s+us|suggests|shows)\b[^.!?]*\babout\s+(?:you|your)\b",
            re.IGNORECASE,
        ),
    ),
    # "as a sophisticated/discerning/serious/basic/pretentious reader…" —
    # an identity adjective applied to the reader (plain "as a reader" is NOT
    # matched: an adjective between the article and the noun is required).
    (
        "reader_identity_label",
        re.compile(
            r"\bas\s+an?\s+\w+\s+(?:reader|person|traveler|traveller)\b",
            re.IGNORECASE,
        ),
    ),
    # "you must be (a/an/the kind of)…" — direct identity inference
    (
        "reader_must_be",
        re.compile(
            r"\byou\s+must\s+be\s+(?:a|an|the|quite|really|very|someone)\b",
            re.IGNORECASE,
        ),
    ),
)

# Neutral fit-only fallback used only when sanitizing would otherwise empty a
# required explanation field (never surface an empty or reader-grading line).
_FIT_FALLBACK = "Chosen for how its setting and atmosphere echo the book."

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def flag_reader_directed(text: object) -> List[str]:
    """Return the reader-directed-judgement pattern labels matched in ``text``.

    Empty list means the text is clean. Non-string / empty input -> ``[]``
    (fail-open). This is pure string work — no model/network calls.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return [label for label, pattern in _PATTERNS if pattern.search(text)]


def sanitize_explanation(
    text: object, fallback: Optional[str] = None
) -> Tuple[str, List[str]]:
    """Strip reader-directed clauses from an explanation, sentence by sentence.

    Returns ``(clean_text, matched_labels)``. Any sentence containing a
    prohibited reader-directed pattern is dropped; the remaining fit-only
    sentences are rejoined. If every sentence is dropped (or the input is
    non-string), the neutral fit-only ``fallback`` is returned so a required
    field is never emptied. ``matched_labels`` is empty when nothing was stripped.
    """
    if not isinstance(text, str) or not text.strip():
        return (text if isinstance(text, str) else ""), []

    matched = flag_reader_directed(text)
    if not matched:
        return text, []

    kept: List[str] = []
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        if not sentence.strip():
            continue
        if flag_reader_directed(sentence):
            continue
        kept.append(sentence.strip())

    clean = " ".join(kept).strip()
    if not clean:
        clean = (fallback or _FIT_FALLBACK)
    return clean, matched


def _sanitize_field(container: dict, key: str) -> int:
    """Sanitize ``container[key]`` in place. Returns 1 if it was changed."""
    if not isinstance(container, dict):
        return 0
    original = container.get(key)
    clean, matched = sanitize_explanation(original)
    if matched and clean != original:
        container[key] = clean
        return 1
    return 0


def sanitize_itinerary_explanations(itinerary_dict: Optional[dict]) -> Optional[dict]:
    """Sanitize every reader-facing explanation field in an itinerary dict.

    Covers ``summary_text``, each city's ``overview``, and each stop's
    ``reason``. Mutates and returns the same dict (fail-open on None/empty).
    """
    if not itinerary_dict:
        return itinerary_dict

    changed = _sanitize_field(itinerary_dict, "summary_text")
    for city in itinerary_dict.get("cities") or []:
        changed += _sanitize_field(city, "overview")
        for stop in city.get("stops") or []:
            changed += _sanitize_field(stop, "reason")

    if changed:
        logger.info("itinerary_tone_sanitized", fields=changed)
    return itinerary_dict


def sanitize_expansion_explanations(expansion_dict: Optional[dict]) -> Optional[dict]:
    """Sanitize the ``reason`` on each expansion place (fail-open on None)."""
    if not expansion_dict:
        return expansion_dict

    changed = 0
    for place in expansion_dict.get("places") or []:
        changed += _sanitize_field(place, "reason")

    if changed:
        logger.info("expansion_tone_sanitized", fields=changed)
    return expansion_dict


def sanitize_book_recommendations(rec_dict: Optional[dict]) -> Optional[dict]:
    """Sanitize the ``reason`` on each book recommendation (fail-open on None)."""
    if not rec_dict:
        return rec_dict

    changed = 0
    for rec in rec_dict.get("recommendations") or []:
        changed += _sanitize_field(rec, "reason")

    if changed:
        logger.info("book_recommendations_tone_sanitized", fields=changed)
    return rec_dict
