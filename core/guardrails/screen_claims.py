"""The screen journey's claim guardrail (Movies epic story 4, ai-private#67).

"Set here" is the strong claim on a film or series journey: it says the story
takes place at this spot. A fan notices at once when a place that was only a
filming location is sold as the setting (Dubrovnik is not King's Landing). So a
``set_here`` stop must carry its evidence, the same way a book stop's
``literal``/``historical`` label must (``core.extraction.downgrade_ungrounded_match_types``):

* a ``set_here`` stop labelled ``literal``/``historical`` with no
  ``grounding_source`` is downgraded to the weaker claim, ``filmed_here`` +
  ``vibe``, before any event leaves the service;
* a ``set_here`` stop that was ``literal``/``historical`` but lost its grounding
  in the book-path downgrade (the cited evidence did not trace to research) is
  downgraded the same way.

Stops are never dropped and never upgraded. Book journeys are untouched here:
``clear_screen_claims_unless_screen_work`` nulls every screen claim on them.
"""

from typing import Iterator, List, Optional

from common.logging import get_logger

logger = get_logger("storyland.guardrails.screen_claims")

SCREEN_WORK_KINDS = frozenset({"film", "series"})
_STRONG_MATCH_TYPES = ("literal", "historical")
_WEAKER_CLAIM = "filmed_here"
_WEAKER_MATCH_TYPE = "vibe"


def _stops(holder: Optional[dict], stops_path: str) -> Iterator[dict]:
    if not isinstance(holder, dict):
        return
    if stops_path == "expansion":
        candidates = holder.get("places") or []
    else:
        candidates = [
            stop
            for city in holder.get("cities") or []
            if isinstance(city, dict)
            for stop in city.get("stops") or []
        ]
    for stop in candidates:
        if isinstance(stop, dict):
            yield stop


def _demote(stop: dict) -> None:
    stop["screen_claim"] = _WEAKER_CLAIM
    stop["match_type"] = _WEAKER_MATCH_TYPE
    stop["grounding_source"] = None


def downgrade_unsupported_set_here_claims(
    holder: Optional[dict], work_kind: object, stops_path: str = "itinerary"
) -> Optional[dict]:
    """Downgrade every ``set_here`` stop that claims literal/historical with no source.

    ``stops_path`` is ``"itinerary"`` (cities → stops) or ``"expansion"`` (places).
    No-op unless ``work_kind`` is film or series. Mutates and returns ``holder``.
    """
    if work_kind not in SCREEN_WORK_KINDS:
        return holder
    downgraded = 0
    for stop in _stops(holder, stops_path):
        if (
            stop.get("screen_claim") == "set_here"
            and stop.get("match_type") in _STRONG_MATCH_TYPES
            and not (stop.get("grounding_source") or "").strip()
        ):
            _demote(stop)
            downgraded += 1
    if downgraded:
        logger.info("screen_set_here_downgraded", downgraded=downgraded, reason="no_grounding_source")
    return holder


def strong_set_here_stops(holder: Optional[dict], work_kind: object) -> List[dict]:
    """The ``set_here`` stops that currently stand on literal/historical evidence.

    Taken before the book-path downgrade so ``demote_set_here_that_lost_grounding``
    can tell which of them that downgrade weakened.
    """
    if work_kind not in SCREEN_WORK_KINDS:
        return []
    return [
        stop
        for stop in _stops(holder, "itinerary")
        if stop.get("screen_claim") == "set_here" and stop.get("match_type") in _STRONG_MATCH_TYPES
    ]


def demote_set_here_that_lost_grounding(stops: List[dict]) -> int:
    """Downgrade the stops from ``strong_set_here_stops`` whose evidence did not hold up."""
    demoted = 0
    for stop in stops:
        if stop.get("screen_claim") == "set_here" and stop.get("match_type") not in _STRONG_MATCH_TYPES:
            _demote(stop)
            demoted += 1
    if demoted:
        logger.info("screen_set_here_downgraded", downgraded=demoted, reason="ungrounded_title")
    return demoted
