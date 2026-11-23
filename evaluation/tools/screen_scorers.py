"""Deterministic scorers for the screen path (film & series journeys), ai-private#68.

The screen path's specific failure is saying "set here" about a place that was
only a filming location: invisible in a demo, obvious to a fan. These scorers
measure it from day one, offline, with no LLM judge.

Two metrics, both in [0, 1]:

* **claim precision** -- of the stops the composer marked ``screen_claim="set_here"``,
  the share whose place is truly a narrative setting of the work (one of the
  case's ``story_locations``). A ``set_here`` on Dubrovnik for *Game of Thrones*
  (Westeros is invented) or on Ischia for *The Talented Mr. Ripley* (Mongibello
  is invented) lowers it. A journey with stops but no ``set_here`` claim makes no
  false claim: 1.0. Over the set it is **per claim** (correct ``set_here`` ÷ all
  ``set_here``), never a mean of per-title ratios: one title full of false
  claims must weigh by how many it makes (``aggregate``).
  A stop is placed by its own name + address first; a stop whose own name or
  address names a filming-only place (Ischia, Big Sur) is that place, whatever
  city it is filed under. The city is a fallback only when the stop's own words
  name no known place.
* **book-pairing honesty** -- the discovered ``source_work`` is the expected book,
  or there is none. ``none`` is always honest (a missing pairing is a gap, not a
  lie); a book that is not the expected one, or any book for a work with no
  source book, is invented: 0.0. A hard metric (release gate: any miss blocks).

The empty-composition trap (``storyland-eval-harness-empty-composition``): a
``found`` case whose discovery came back with zero stops scores **0** on both
metrics and fails -- an empty run must never read as a perfect one. A
``not_found`` case (a title that does not exist) passes only when nothing was
composed and no book was paired.

Input shape (what ai-private#67's screen discovery emits; see
``evaluation/README.md``)::

    {
      "itinerary": {"cities": [{"name", "country", "stops": [{"name", "address",
                     "screen_claim", "filming_scene", ...}]}]},   # or a ComposerEnvelope
      "screen_context": {"story_locations", "filming_locations",
                         "source_work": {"title", "author"} | null,
                         "is_fictional_world"} | null
    }
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

CLAIM_PRECISION = "claim_precision"
BOOK_PAIRING_HONESTY = "book_pairing_honesty"
NAMED_CLAIMS = "named_claims"


def _tokens(text: Optional[str]) -> List[str]:
    """Casefolded, accent-free word tokens ("Montmartre, Paris" → ["montmartre", "paris"])."""
    decomposed = unicodedata.normalize("NFKD", (text or "").casefold())
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.findall(r"\w+", plain)


def place_matches(place: str, expected: str) -> bool:
    """True when ``expected``'s words appear, in order and adjacent, inside ``place``.

    "Café des Deux Moulins, Paris, France" matches "Paris"; "Romeo's Diner" does
    not match "Rome"; "Belfast, Northern Ireland" matches "Northern Ireland".
    """
    hay, needle = _tokens(place), _tokens(expected)
    if not needle:
        return False
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def _matches_any(place: str, expected: Iterable[str]) -> bool:
    return any(place_matches(place, e) for e in expected)


def _join(*parts: object) -> str:
    return ", ".join(p for p in parts if isinstance(p, str) and p.strip())


def extract_stops(output: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every composed stop as ``{"place", "own", "city", "screen_claim"}``.

    ``own`` is the stop's name + address, ``city`` its city + country, ``place``
    both joined. Accepts a ``TripItinerary`` dict or a ``ComposerEnvelope``
    (``{"itinerary": {...}}``) under ``output["itinerary"]``. A non-dict output
    (or itinerary) yields no stops.
    """
    itinerary = output.get("itinerary") if isinstance(output, dict) else None
    if not isinstance(itinerary, dict):
        return []
    if isinstance(itinerary.get("itinerary"), dict):  # ComposerEnvelope
        itinerary = itinerary["itinerary"]
    stops: List[Dict[str, Any]] = []
    for city in itinerary.get("cities") or []:
        if not isinstance(city, dict):
            continue
        for stop in city.get("stops") or []:
            if not isinstance(stop, dict):
                continue
            own = _join(stop.get("name"), stop.get("address"))
            where = _join(city.get("name"), city.get("country"))
            stops.append({
                "place": _join(own, where),
                "own": own,
                "city": where,
                "screen_claim": stop.get("screen_claim"),
            })
    return stops


def is_setting(
    stop: Dict[str, Any], story_locations: List[str], filming_locations: Iterable[str] = ()
) -> bool:
    """Whether a stop stands in one of the work's narrative settings.

    The stop's own name + address decide first: naming a filming-only place
    (one that is not also a setting) is a no; naming a setting is a yes. Only
    when its own words name no known place does its city decide.
    """
    filming_only = [
        f for f in filming_locations
        if not any(place_matches(f, s) or place_matches(s, f) for s in story_locations)
    ]
    own = stop.get("own", stop.get("place", ""))
    if _matches_any(own, filming_only):
        return False
    if _matches_any(own, story_locations):
        return True
    city = stop.get("city", "")
    return _matches_any(city, story_locations) and not _matches_any(city, filming_only)


def set_here_counts(
    stops: List[Dict[str, Any]], story_locations: List[str], filming_locations: Iterable[str] = ()
) -> Tuple[int, int]:
    """(correct ``set_here`` stops, all ``set_here`` stops)."""
    filming = list(filming_locations)
    set_here = [s for s in stops if s.get("screen_claim") == "set_here"]
    right = sum(1 for s in set_here if is_setting(s, story_locations, filming))
    return right, len(set_here)


def claim_precision(
    stops: List[Dict[str, Any]], story_locations: List[str], filming_locations: Iterable[str] = ()
) -> float:
    """Share of ``set_here`` stops whose place is a narrative setting. No stops → 0.0."""
    if not stops:
        return 0.0
    right, total = set_here_counts(stops, story_locations, filming_locations)
    return right / total if total else 1.0


def _same_book(found: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Same title words (a leading article ignored) and the expected author's surname present."""
    def title(t: Optional[str]) -> List[str]:
        words = _tokens(t)
        return words[1:] if words[:1] in (["the"], ["a"], ["an"]) else words

    # A series can pair with its saga or with its first volume ("A Song of Ice and
    # Fire" / "A Game of Thrones"): the evalset lists the accepted titles.
    accepted = [expected.get("title")] + list(expected.get("also_titles") or [])
    found_title = title(found.get("title"))
    if not found_title or all(found_title != title(t) for t in accepted):
        return False
    surname = (_tokens(expected.get("author")) or [""])[-1]
    # Discovery serializes source_work as a WorkRef, whose author is `creator`
    # (models/work.py); `author` stays accepted for hand-written outputs.
    return not surname or surname in _tokens(found.get("creator") or found.get("author"))


def book_pairing_honesty(
    source_work: Optional[Dict[str, Any]],
    expected: Optional[Dict[str, Any]],
) -> float:
    """1.0 when the paired book is the expected one or there is none; 0.0 for an invented book."""
    if not source_work or not (source_work.get("title") or "").strip():
        return 1.0
    if not expected:
        return 0.0
    return 1.0 if _same_book(source_work, expected) else 0.0


def named_claims(stops: List[Dict[str, Any]], required: List[Dict[str, str]]) -> float:
    """Share of the case's named stops that are present AND carry the required claim.

    A named place that never appears counts as missed: the eval names it because
    a good journey for that title goes there.
    """
    if not required:
        return 1.0
    hits = 0
    for want in required:
        here = [s for s in stops if place_matches(s["place"], want["place"])]
        if here and all(s.get("screen_claim") == want["screen_claim"] for s in here):
            hits += 1
    return hits / len(required)


def score_case(output: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> Dict[str, Any]:
    """Score one title's discovery output against its expected block from the evalset."""
    stops = extract_stops(output)
    context = (output.get("screen_context") if isinstance(output, dict) else None) or {}
    source_work = context.get("source_work") if isinstance(context, dict) else None
    if not isinstance(source_work, dict):
        source_work = None
    expect = expected.get("expect", "found")

    if expect == "not_found":
        honest = not stops and not (source_work or {}).get("title")
        return {
            "expect": expect,
            "n_stops": len(stops),
            CLAIM_PRECISION: None,  # nothing to claim about a title that does not exist
            BOOK_PAIRING_HONESTY: 1.0 if honest else 0.0,
            NAMED_CLAIMS: None,
            "pass": honest,
        }

    if not stops:  # the empty-composition trap: nothing composed is a 0, never a pass
        return {
            "expect": expect,
            "n_stops": 0,
            "n_set_here": 0,
            "n_set_here_right": 0,
            CLAIM_PRECISION: 0.0,
            BOOK_PAIRING_HONESTY: 0.0,
            NAMED_CLAIMS: 0.0,
            "pass": False,
        }

    story = expected.get("story_locations") or []
    filming = expected.get("filming_locations") or []
    right, total = set_here_counts(stops, story, filming)
    precision = right / total if total else 1.0
    honesty = book_pairing_honesty(source_work, expected.get("source_work"))
    named = named_claims(stops, expected.get("required_claims") or [])
    return {
        "expect": expect,
        "n_stops": len(stops),
        "n_set_here": total,
        "n_set_here_right": right,
        CLAIM_PRECISION: precision,
        BOOK_PAIRING_HONESTY: honesty,
        NAMED_CLAIMS: named,
        "pass": precision >= CLAIM_PRECISION_FLOOR and honesty == 1.0 and named == 1.0,
    }


# The card's merge gate for ai-private#67: claim precision ≥ 0.9 over the set,
# book-pairing honesty = 1.0 (any invented book blocks).
CLAIM_PRECISION_FLOOR = 0.9


def _set_wide_claim_precision(case_scores: List[Dict[str, Any]]) -> Optional[float]:
    """Correct ``set_here`` ÷ all ``set_here`` across the set (the card's definition).

    A found case that composed nothing counts as one unsupported claim, so an
    empty run scores 0 and a partly empty one is pulled down, never up. Cases
    from an older summary without the counts fall back to their ratio as one
    claim. No set_here anywhere (and no empty case) is 1.0: no false claim made.
    """
    scored = [c for c in case_scores if c.get(CLAIM_PRECISION) is not None]
    if not scored:
        return None
    right = total = 0.0
    for c in scored:
        if c.get("n_stops") == 0:
            total += 1
        elif "n_set_here" in c:
            right += c.get("n_set_here_right", 0)
            total += c["n_set_here"]
        else:
            right += float(c[CLAIM_PRECISION])
            total += 1
    return right / total if total else 1.0


def aggregate(case_scores: List[Dict[str, Any]]) -> Dict[str, float]:
    """Each metric over the set: claim precision per claim, the others as a mean
    over the cases that carry them (``None`` = not applicable)."""
    out: Dict[str, float] = {}
    precision = _set_wide_claim_precision(case_scores)
    if precision is not None:
        out[CLAIM_PRECISION] = round(precision, 4)
    for metric in (BOOK_PAIRING_HONESTY, NAMED_CLAIMS):
        values = [c[metric] for c in case_scores if c.get(metric) is not None]
        if values:
            out[metric] = round(sum(values) / len(values), 4)
    return out
