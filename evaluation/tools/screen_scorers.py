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
  false claim: 1.0.
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
from typing import Any, Dict, Iterable, List, Optional

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


def extract_stops(output: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every composed stop as ``{"place": <stop + address + city + country>, "screen_claim": ...}``.

    Accepts a ``TripItinerary`` dict or a ``ComposerEnvelope`` (``{"itinerary": {...}}``)
    under ``output["itinerary"]``. The city and country are part of the place so a
    stop named for a café still matches the city it stands in.
    """
    itinerary = (output or {}).get("itinerary") or {}
    if isinstance(itinerary.get("itinerary"), dict):  # ComposerEnvelope
        itinerary = itinerary["itinerary"]
    stops: List[Dict[str, Any]] = []
    for city in itinerary.get("cities") or []:
        if not isinstance(city, dict):
            continue
        for stop in city.get("stops") or []:
            if not isinstance(stop, dict):
                continue
            parts = [stop.get("name"), stop.get("address"), city.get("name"), city.get("country")]
            stops.append({
                "place": ", ".join(p for p in parts if isinstance(p, str) and p.strip()),
                "screen_claim": stop.get("screen_claim"),
            })
    return stops


def claim_precision(stops: List[Dict[str, Any]], story_locations: List[str]) -> float:
    """Share of ``set_here`` stops whose place is a narrative setting. No stops → 0.0."""
    if not stops:
        return 0.0
    set_here = [s for s in stops if s.get("screen_claim") == "set_here"]
    if not set_here:
        return 1.0
    right = sum(1 for s in set_here if _matches_any(s["place"], story_locations))
    return right / len(set_here)


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
    return not surname or surname in _tokens(found.get("author"))


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
    context = (output or {}).get("screen_context") or {}
    source_work = context.get("source_work") if isinstance(context, dict) else None
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
            CLAIM_PRECISION: 0.0,
            BOOK_PAIRING_HONESTY: 0.0,
            NAMED_CLAIMS: 0.0,
            "pass": False,
        }

    precision = claim_precision(stops, expected.get("story_locations") or [])
    honesty = book_pairing_honesty(source_work, expected.get("source_work"))
    named = named_claims(stops, expected.get("required_claims") or [])
    return {
        "expect": expect,
        "n_stops": len(stops),
        "n_set_here": sum(1 for s in stops if s.get("screen_claim") == "set_here"),
        CLAIM_PRECISION: precision,
        BOOK_PAIRING_HONESTY: honesty,
        NAMED_CLAIMS: named,
        "pass": precision >= CLAIM_PRECISION_FLOOR and honesty == 1.0 and named == 1.0,
    }


# The card's merge gate for ai-private#67: claim precision ≥ 0.9 over the set,
# book-pairing honesty = 1.0 (any invented book blocks).
CLAIM_PRECISION_FLOOR = 0.9


def aggregate(case_scores: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean of each metric over the cases that carry it (``None`` = not applicable)."""
    out: Dict[str, float] = {}
    for metric in (CLAIM_PRECISION, BOOK_PAIRING_HONESTY, NAMED_CLAIMS):
        values = [c[metric] for c in case_scores if c.get(metric) is not None]
        if values:
            out[metric] = round(sum(values) / len(values), 4)
    return out
