"""Wikidata as a second source of place -> book facts (storyland-ai-private#81, Part 1/2).

Wikidata states "this literary work is set in this place" as structured data:
property P840 (narrative location) links a work to a place, and every work and
place has a stable id (QID). This module turns a place into the written works
Wikidata says are set there, with their QIDs as receipts. Part 2 wires it into
the ``place_to_book`` graph as its own node beside the search researcher.

Rules this module keeps (the card's acceptance):

* **The model never writes SPARQL.** Every query is a fixed template below,
  filled only with QIDs that passed ``_QID_RE``. A place name reaches Wikidata
  only as the ``search`` parameter of the entity-search API, never inside a
  query.
* **Results are parsed into Pydantic models.** No raw tool text goes on to a
  model as instructions.
* **Fail open.** Slow, down, malformed or empty: the lookup returns a status
  (``unavailable`` / ``ambiguous`` / ``not_found`` / ``empty``), logs one line
  and leaves a Sentry breadcrumb. It never raises into the flow, which then
  continues on search alone.
* **No guessing a place.** "Dublin" is also a city in Ohio. A name resolves only
  when one candidate is a place (it has coordinates, P625), its label or alias
  is the name, and it clearly dominates the other such candidates by sitelink
  count (``_DOMINANCE``). Otherwise the lookup is skipped, not guessed. A caller
  that already holds the place's QID (the gateway's resolver gets one from
  Mapbox, services#210) passes it and skips the search.
* **Written works only.** P840 also covers films and series; the second query
  keeps works that are an instance of a subclass of "literary work" (Q7725634).

Transport: the public endpoints (the entity-search API and the query service at
query.wikidata.org) behind one small interface, ``WikidataSource``. The card's
MCP connection (the Wikidata project's MCP server runs the same SPARQL) is a
second implementation of that interface in Part 2; nothing above the interface
changes. Requests carry a descriptive User-Agent, as Wikimedia's policy asks.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Literal, Optional, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from common.logging import get_logger

logger = get_logger("storyland.core.wikidata")

# Wikimedia's User-Agent policy: name the tool and how to reach its operator.
USER_AGENT = "StorylandBot/1.0 (https://mystoryland.ai; place-to-book lookup)"

SEARCH_API_URL = "https://www.wikidata.org/w/api.php"
SPARQL_URL = "https://query.wikidata.org/sparql"

# Per-request timeout, and the whole lookup's budget (search + 3 queries).
# Measured on 2026-11-25 for Dublin: search ~0.5 s, place check ~2 s, works by
# location ~3.7 s, the bounded type filter ~6.9 s.
REQUEST_TIMEOUT_SECONDS = 12.0
LOOKUP_BUDGET_SECONDS = 20.0

# fullmatch, not ^…$: "$" also matches before a trailing newline.
_QID_RE = re.compile(r"Q[1-9][0-9]{0,11}")
_ENTITY_PREFIX = "http://www.wikidata.org/entity/"

# How many entity-search hits are checked for "is it a place".
_SEARCH_LIMIT = 7
# A name resolves to the top place only when it has at least this many times
# the sitelinks of the next place with the same name. Dublin (Ireland, 239)
# against Dublin (Ohio, 49): resolves. Two towns of similar size: skipped.
_DOMINANCE = 3.0
# Caps: works pulled by location, and works returned.
_MAX_LOCATION_ROWS = 500
_MAX_WORKS = 60

# "literary work": novels, plays, poems, short stories and their subclasses.
LITERARY_WORK_QID = "Q7725634"

LookupStatus = Literal["found", "empty", "not_found", "ambiguous", "unavailable"]


class WikidataError(Exception):
    """Any transport or shape failure talking to Wikidata."""


class PlaceMatch(BaseModel):
    """A place candidate for a name, after the type check."""

    qid: str
    label: str
    sitelinks: int = 0
    has_coordinates: bool = False


class WikidataWork(BaseModel):
    """A written work Wikidata says is set in the place (P840)."""

    work_qid: str
    title: str
    authors: list[str] = Field(default_factory=list)
    # The P840 place the statement names: the place itself, or a place inside it.
    location_qid: str
    isbn13: Optional[str] = None
    isbn10: Optional[str] = None
    open_library_id: Optional[str] = None


class WikidataLookup(BaseModel):
    """The lookup's answer. ``works`` is non-empty only when status is ``found``."""

    status: LookupStatus
    place_qid: Optional[str] = None
    works: list[WikidataWork] = Field(default_factory=list)
    reason: str = ""


class WikidataSource(Protocol):
    """The two calls the lookup needs. HTTP today; the MCP server in Part 2."""

    async def search_items(self, name: str, limit: int) -> list[dict[str, Any]]:
        """Entity search: ``[{"id": "Q1761", "label": ..., "aliases": [...]}, ...]``."""

    async def sparql(self, query: str) -> list[dict[str, Any]]:
        """Run a SPARQL SELECT and return its ``results.bindings`` rows."""


class HttpWikidataSource:
    """``WikidataSource`` over the public entity-search API and query service."""

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        self._client = client

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def search_items(self, name: str, limit: int) -> list[dict[str, Any]]:
        try:
            response = await self._http().get(
                SEARCH_API_URL,
                params={
                    "action": "wbsearchentities",
                    "search": name,
                    "language": "en",
                    "type": "item",
                    "limit": str(limit),
                    "format": "json",
                },
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WikidataError(f"entity search failed: {type(exc).__name__}") from exc
        hits = body.get("search") if isinstance(body, dict) else None
        if not isinstance(hits, list):
            raise WikidataError("entity search answered without a 'search' list")
        return [h for h in hits if isinstance(h, dict)]

    async def sparql(self, query: str) -> list[dict[str, Any]]:
        try:
            # POST: the bounded second query carries up to _MAX_LOCATION_ROWS ids.
            response = await self._http().post(
                SPARQL_URL,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WikidataError(f"SPARQL query failed: {type(exc).__name__}") from exc
        try:
            rows = body["results"]["bindings"]
        except (KeyError, TypeError) as exc:
            raise WikidataError("SPARQL answer has no results.bindings") from exc
        if not isinstance(rows, list):
            raise WikidataError("SPARQL results.bindings is not a list")
        return rows


# --- Fixed templates. Only validated QIDs are ever substituted. ---------------

_PLACE_CHECK_TEMPLATE = """SELECT ?item ?sitelinks ?coord WHERE {{
  VALUES ?item {{ {items} }}
  ?item wikibase:sitelinks ?sitelinks .
  OPTIONAL {{ ?item wdt:P625 ?coord }}
}}"""

# Works whose narrative location is the place, or a place located in it (P131).
_WORKS_BY_LOCATION_TEMPLATE = """SELECT DISTINCT ?work ?loc WHERE {{
  {{ ?work wdt:P840 wd:{place} . BIND(wd:{place} AS ?loc) }}
  UNION
  {{ ?loc wdt:P131 wd:{place} . ?work wdt:P840 ?loc . }}
}} LIMIT {limit}"""

# Bounded to the works found above, so the subclass path stays cheap.
_WRITTEN_WORK_DETAILS_TEMPLATE = """SELECT ?work ?workLabel ?authorLabel ?isbn13 ?isbn10 ?olid WHERE {{
  VALUES ?work {{ {works} }}
  FILTER EXISTS {{ ?work wdt:P31/wdt:P279* wd:{literary} }}
  OPTIONAL {{ ?work wdt:P50 ?author }}
  OPTIONAL {{ ?work wdt:P212 ?isbn13 }}
  OPTIONAL {{ ?work wdt:P957 ?isbn10 }}
  OPTIONAL {{ ?work wdt:P648 ?olid }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""


def is_qid(value: object) -> bool:
    return isinstance(value, str) and _QID_RE.fullmatch(value) is not None


def _require_qids(qids: list[str]) -> list[str]:
    bad = [q for q in qids if not is_qid(q)]
    if bad:
        # A programming error, not a Wikidata answer: never build the query.
        raise ValueError(f"not a QID: {bad[:3]!r}")
    return qids


def place_check_query(qids: list[str]) -> str:
    items = " ".join(f"wd:{q}" for q in _require_qids(qids))
    return _PLACE_CHECK_TEMPLATE.format(items=items)


def works_by_location_query(place_qid: str, limit: int = _MAX_LOCATION_ROWS) -> str:
    (place,) = _require_qids([place_qid])
    return _WORKS_BY_LOCATION_TEMPLATE.format(place=place, limit=int(limit))


def written_work_details_query(work_qids: list[str]) -> str:
    works = " ".join(f"wd:{q}" for q in _require_qids(work_qids))
    return _WRITTEN_WORK_DETAILS_TEMPLATE.format(works=works, literary=LITERARY_WORK_QID)


# --- Parsing ------------------------------------------------------------------


def _value(row: dict[str, Any], key: str) -> Optional[str]:
    cell = row.get(key)
    if not isinstance(cell, dict):
        return None
    value = cell.get("value")
    return value if isinstance(value, str) and value.strip() else None


def _entity_qid(row: dict[str, Any], key: str) -> Optional[str]:
    value = _value(row, key)
    if value is None or not value.startswith(_ENTITY_PREFIX):
        return None
    qid = value[len(_ENTITY_PREFIX):]
    return qid if is_qid(qid) else None


def _same_name(hit: dict[str, Any], name: str) -> bool:
    wanted = name.strip().casefold()
    names = [hit.get("label")] + list(hit.get("aliases") or [])
    match = hit.get("match")
    if isinstance(match, dict):
        names.append(match.get("text"))
    return any(isinstance(n, str) and n.strip().casefold() == wanted for n in names)


def parse_place_matches(hits: list[dict[str, Any]], rows: list[dict[str, Any]], name: str) -> list[PlaceMatch]:
    """Places among the search hits whose label or alias is ``name``, best first."""
    named = {h["id"]: h for h in hits if is_qid(h.get("id")) and _same_name(h, name)}
    # One candidate per QID: an item with several P625 values comes back as one row per
    # coordinate, and two copies of one place would fail pick_place's dominance check.
    by_qid: dict[str, PlaceMatch] = {}
    for row in rows:
        qid = _entity_qid(row, "item")
        if qid is None or qid not in named:
            continue
        try:
            sitelinks = int(_value(row, "sitelinks") or 0)
        except ValueError:
            sitelinks = 0
        has_coordinates = _value(row, "coord") is not None
        seen = by_qid.get(qid)
        if seen is not None:
            has_coordinates = has_coordinates or seen.has_coordinates
            sitelinks = max(sitelinks, seen.sitelinks)
        by_qid[qid] = PlaceMatch(
            qid=qid,
            label=str(named[qid].get("label") or name),
            sitelinks=sitelinks,
            has_coordinates=has_coordinates,
        )
    places = [p for p in by_qid.values() if p.has_coordinates]
    return sorted(places, key=lambda p: p.sitelinks, reverse=True)


def pick_place(places: list[PlaceMatch]) -> Optional[PlaceMatch]:
    """The one clear place, or None when there is none or two are close."""
    if not places:
        return None
    if len(places) == 1:
        return places[0]
    top, runner_up = places[0], places[1]
    if top.sitelinks >= _DOMINANCE * max(runner_up.sitelinks, 1):
        return top
    return None


def parse_works(location_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> list[WikidataWork]:
    """One ``WikidataWork`` per written work, in first-seen order, capped."""
    location_of: dict[str, str] = {}
    for row in location_rows:
        work, loc = _entity_qid(row, "work"), _entity_qid(row, "loc")
        if work and loc:
            location_of.setdefault(work, loc)

    merged: dict[str, dict[str, Any]] = {}
    for row in detail_rows:
        work = _entity_qid(row, "work")
        title = _value(row, "workLabel")
        if work is None or title is None or work not in location_of:
            continue
        # The label service falls back to the QID when a work has no English
        # label; a QID is not a title.
        if is_qid(title):
            continue
        entry = merged.setdefault(
            work,
            {"work_qid": work, "title": title.strip(), "authors": [], "location_qid": location_of[work]},
        )
        author = _value(row, "authorLabel")
        if author and not is_qid(author) and author not in entry["authors"]:
            entry["authors"].append(author.strip())
        for key, field in (("isbn13", "isbn13"), ("isbn10", "isbn10"), ("olid", "open_library_id")):
            value = _value(row, key)
            if value and field not in entry:
                entry[field] = value.strip()

    works: list[WikidataWork] = []
    for entry in merged.values():
        try:
            works.append(WikidataWork.model_validate(entry))
        except ValidationError:
            continue
        if len(works) >= _MAX_WORKS:
            break
    return works


# --- The lookup -----------------------------------------------------------------


def _breadcrumb(message: str, data: dict[str, Any]) -> None:
    try:
        import sentry_sdk

        sentry_sdk.add_breadcrumb(category="wikidata", message=message, level="info", data=data)
    except Exception:  # never let telemetry break the fail-open path
        pass


def _skip(status: LookupStatus, reason: str, place: str, place_qid: Optional[str] = None) -> WikidataLookup:
    logger.info("wikidata_lookup_skipped", status=status, reason=reason, place=place, place_qid=place_qid)
    # The raw place query stays out of the breadcrumb: breadcrumbs ride on every later Sentry event, and
    # `api/sentry.py` does not scrub a plain `place` key (the logging allowlist drops it from the log line).
    _breadcrumb(f"wikidata lookup {status}", {"status": status, "reason": reason, "place_qid": place_qid})
    return WikidataLookup(status=status, place_qid=place_qid, reason=reason)


async def _resolve_place(source: WikidataSource, name: str) -> tuple[Optional[str], str]:
    hits = await source.search_items(name, _SEARCH_LIMIT)
    candidate_qids = [h["id"] for h in hits if is_qid(h.get("id")) and _same_name(h, name)]
    if not candidate_qids:
        return None, "no entity with that name"
    rows = await source.sparql(place_check_query(candidate_qids))
    places = parse_place_matches(hits, rows, name)
    chosen = pick_place(places)
    if chosen is None:
        return None, "no place with that name" if not places else "several places with that name"
    return chosen.qid, ""


async def _lookup(source: WikidataSource, place: str, place_qid: Optional[str]) -> WikidataLookup:
    if place_qid is None:
        place_qid, why = await _resolve_place(source, place)
        if place_qid is None:
            status: LookupStatus = "ambiguous" if why == "several places with that name" else "not_found"
            return _skip(status, why, place)

    location_rows = await source.sparql(works_by_location_query(place_qid))
    work_qids = sorted({q for q in (_entity_qid(r, "work") for r in location_rows) if q})
    if not work_qids:
        return _skip("empty", "no work names this place as its narrative location", place, place_qid)

    detail_rows = await source.sparql(written_work_details_query(work_qids))
    works = parse_works(location_rows, detail_rows)
    if not works:
        return _skip("empty", "no written work among the works set here", place, place_qid)
    logger.info("wikidata_lookup_found", place=place, place_qid=place_qid, works=len(works))
    return WikidataLookup(status="found", place_qid=place_qid, works=works)


async def lookup_works_set_in(
    source: WikidataSource,
    place: str,
    *,
    place_qid: Optional[str] = None,
    budget_seconds: float = LOOKUP_BUDGET_SECONDS,
) -> WikidataLookup:
    """Written works Wikidata says are set in ``place``. Never raises.

    ``place_qid``, when the caller already has it (and it is a valid QID),
    skips the name search. Any failure or a blown ``budget_seconds`` answers
    ``unavailable``: the caller carries on without Wikidata.
    """
    if place_qid is not None and not is_qid(place_qid):
        place_qid = None
    try:
        return await asyncio.wait_for(_lookup(source, place, place_qid), timeout=budget_seconds)
    except asyncio.TimeoutError:
        return _skip("unavailable", f"no answer within {budget_seconds:.0f} s", place, place_qid)
    except WikidataError as exc:
        return _skip("unavailable", str(exc), place, place_qid)
    except Exception as exc:  # fail open on anything unexpected, with its type
        logger.warning("wikidata_lookup_error", place=place, error=type(exc).__name__)
        return _skip("unavailable", f"unexpected {type(exc).__name__}", place, place_qid)
