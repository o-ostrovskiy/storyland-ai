"""Reading highlights for the itinerary page, picked in code (prompt v7).

The page bolds up to two phrases per paragraph (storyland-web
`HighlightedText`). Three live runs on 2026-11-25 showed that asking the
composer to pick them while it writes gives generic picks ("central hub",
"visual splendor") about half the time. So the choice is split:

* **Known names, by code.** The book's title, its author and its main
  characters (discovery's ``book_context.main_characters``) are bolded wherever
  they appear as whole words. Exact every time and free.
* **One specific fact, by the model.** The composer may add one phrase per
  field that is none of those (a named object, work or event, or a number).
  It is kept only if it appears verbatim and holds a name or number, the same
  rule the page applies.

Main places are deliberately NOT bolded: a city name sits in the heading right
above its own overview, so bolding it again guides the eye to nothing.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Tuple

MAX_HIGHLIGHTS_PER_FIELD = 2
MAX_MODEL_PICKS_PER_FIELD = 1

# A first or last name on its own is bolded too ("Theo" for "Theo Decker"),
# unless it is one of these, which read as ordinary words at a sentence start.
_NAME_PART_STOPWORDS = frozenset(
    {"the", "a", "an", "and", "of", "de", "la", "le", "van", "von", "der", "mr", "mrs", "ms", "dr", "sir", "lady", "lord"}
)


def phrase_has_name(phrase: str) -> bool:
    """A capital letter (a name, a title) or a digit. Mirrors the page's ``hasName``."""
    return any(ch.isupper() or ch.isdigit() for ch in phrase)


def known_names(title: Optional[str], author: Optional[str], characters: Iterable[Any]) -> List[str]:
    """The names code bolds: title, author (and surname), characters (and first/last name)."""
    names: List[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            value = value.strip()
            if len(value) >= 3 and phrase_has_name(value) and value not in names:
                names.append(value)

    def add_parts(full: str) -> None:
        parts = [p for p in full.split() if p]
        if len(parts) < 2:
            return
        for part in (parts[0], parts[-1]):
            clean = part.strip(".,;:'\"")
            if len(clean) >= 4 and clean.lower() not in _NAME_PART_STOPWORDS:
                add(clean)

    add(title)
    if isinstance(author, str):
        add(author)
        parts = author.split()
        if len(parts) >= 2:
            add(parts[-1].strip(".,"))  # "Mayes", "Christie"
    for character in characters or []:
        if isinstance(character, str):
            add(character)
            add_parts(character)
    return names


def _find_whole(text: str, phrase: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)
    return (match.start(), match.end()) if match else None


def pick_highlights(text: Any, model_picks: Any, names: List[str]) -> List[str]:
    """Phrases to bold in ``text``: the model's one fact, then known names, max two.

    Each returned phrase appears in ``text`` verbatim and none overlap, so the
    page's own checks keep all of them.
    """
    if not isinstance(text, str) or not text:
        return []
    chosen: List[Tuple[int, int, str]] = []

    def overlaps(span: Tuple[int, int]) -> bool:
        return any(span[0] < e and span[1] > s for s, e, _ in chosen)

    # The model's specific fact first: it is the one thing code cannot find.
    model_kept = 0
    for raw in model_picks if isinstance(model_picks, list) else []:
        if model_kept >= MAX_MODEL_PICKS_PER_FIELD:
            break
        phrase = raw.strip() if isinstance(raw, str) else ""
        if not phrase or not phrase_has_name(phrase):
            continue
        start = text.find(phrase)
        if start < 0 or overlaps((start, start + len(phrase))):
            continue
        chosen.append((start, start + len(phrase), phrase))
        model_kept += 1

    # Then known names, longest first so "Theo Decker" wins over "Theo",
    # each by its first whole-word occurrence.
    for name in sorted(names, key=len, reverse=True):
        if len(chosen) >= MAX_HIGHLIGHTS_PER_FIELD:
            break
        span = _find_whole(text, name)
        if span and not overlaps(span):
            chosen.append((span[0], span[1], name))

    return [phrase for _, _, phrase in sorted(chosen)][:MAX_HIGHLIGHTS_PER_FIELD]


def apply_reading_highlights(itinerary: Any, names: List[str]) -> Any:
    """Rewrite every highlights list in a TripItinerary dict in place; returns it.

    Fields: ``summary_highlights`` for ``summary_text``, ``overview_highlights``
    for each city's ``overview``, ``highlights`` for each stop's ``reason``.
    A non-dict input is returned unchanged.
    """
    if not isinstance(itinerary, dict):
        return itinerary
    itinerary["summary_highlights"] = pick_highlights(
        itinerary.get("summary_text"), itinerary.get("summary_highlights"), names
    )
    for city in itinerary.get("cities") or []:
        if not isinstance(city, dict):
            continue
        city["overview_highlights"] = pick_highlights(
            city.get("overview"), city.get("overview_highlights"), names
        )
        for stop in city.get("stops") or []:
            if isinstance(stop, dict):
                stop["highlights"] = pick_highlights(stop.get("reason"), stop.get("highlights"), names)
    return itinerary


def names_from_state(state_accessor: Any) -> List[str]:
    """Known names from a SessionStateAccessor (book metadata + book_context)."""
    context = getattr(state_accessor, "book_context", None) or {}
    if hasattr(context, "model_dump"):
        context = context.model_dump()
    characters = context.get("main_characters") if isinstance(context, dict) else None
    return known_names(
        getattr(state_accessor, "book_title", None),
        getattr(state_accessor, "author", None),
        characters or [],
    )
