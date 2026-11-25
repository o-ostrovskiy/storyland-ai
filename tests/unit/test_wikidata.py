"""Wikidata place -> book lookup (storyland-ai-private#81, Part 1/2).

Offline: a fake ``WikidataSource`` answers with rows shaped like the real
endpoints (taken from a 2026-11-25 probe for Dublin). The VCR replay of the
real endpoints lives in tests/integration/test_wikidata_vcr.py.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core import wikidata as wd
from core.wikidata import (
    HttpWikidataSource,
    WikidataError,
    lookup_works_set_in,
    parse_works,
    pick_place,
    PlaceMatch,
    place_check_query,
    works_by_location_query,
    written_work_details_query,
)

E = "http://www.wikidata.org/entity/"


def uri(qid: str) -> dict[str, str]:
    return {"type": "uri", "value": E + qid}


def lit(value: str) -> dict[str, str]:
    return {"type": "literal", "value": value}


DUBLIN_HITS = [
    {"id": "Q1761", "label": "Dublin", "description": "capital and largest city of Ireland"},
    {"id": "Q173500", "label": "County Dublin"},
    {"id": "Q16865980", "label": "Dublin", "description": "family name"},
    {"id": "Q988937", "label": "Dublin", "description": "city in Ohio, United States"},
]
DUBLIN_PLACE_ROWS = [
    {"item": uri("Q1761"), "sitelinks": lit("239"), "coord": lit("Point(-6.26 53.35)")},
    {"item": uri("Q988937"), "sitelinks": lit("49"), "coord": lit("Point(-83.14 40.11)")},
    {"item": uri("Q16865980"), "sitelinks": lit("2")},  # a family name: no coordinates
]
LOCATION_ROWS = [
    {"work": uri("Q6511"), "loc": uri("Q1761")},  # Ulysses
    {"work": uri("Q1770069"), "loc": uri("Q1761")},  # Watt
    {"work": uri("Q999999"), "loc": uri("Q1761")},  # a film: dropped by the type filter
    {"work": uri("Q2356919"), "loc": uri("Q2000000")},  # a place inside Dublin
]
DETAIL_ROWS = [
    {"work": uri("Q6511"), "workLabel": lit("Ulysses"), "authorLabel": lit("James Joyce"), "olid": lit("OL52245W")},
    {"work": uri("Q6511"), "workLabel": lit("Ulysses"), "authorLabel": lit("James Joyce"), "isbn13": lit("978-0-679-72276-4")},
    {"work": uri("Q1770069"), "workLabel": lit("Watt"), "authorLabel": lit("Samuel Beckett")},
    {"work": uri("Q2356919"), "workLabel": lit("The Informer"), "authorLabel": lit("Liam O'Flaherty")},
]


class FakeSource:
    """Answers by query shape; records every call."""

    def __init__(self, hits=DUBLIN_HITS, place_rows=DUBLIN_PLACE_ROWS, location_rows=LOCATION_ROWS, detail_rows=DETAIL_ROWS):
        self.hits, self.place_rows = hits, place_rows
        self.location_rows, self.detail_rows = location_rows, detail_rows
        self.searches: list[str] = []
        self.queries: list[str] = []

    async def search_items(self, name: str, limit: int) -> list[dict[str, Any]]:
        self.searches.append(name)
        return self.hits

    async def sparql(self, query: str) -> list[dict[str, Any]]:
        self.queries.append(query)
        if "wikibase:sitelinks" in query:
            return self.place_rows
        if "wdt:P840" in query:
            return self.location_rows
        return self.detail_rows


def run(coro):
    return asyncio.run(coro)


# --- the lookup -------------------------------------------------------------------


def test_dublin_resolves_to_the_irish_capital_and_returns_written_works():
    source = FakeSource()
    result = run(lookup_works_set_in(source, "Dublin"))
    assert result.status == "found"
    assert result.place_qid == "Q1761"
    assert [w.title for w in result.works] == ["Ulysses", "Watt", "The Informer"]
    ulysses = result.works[0]
    assert ulysses.work_qid == "Q6511"
    assert ulysses.authors == ["James Joyce"]
    assert ulysses.location_qid == "Q1761"
    assert ulysses.open_library_id == "OL52245W"
    assert ulysses.isbn13 == "978-0-679-72276-4"
    # A work set in a place inside Dublin keeps that place's QID as its receipt.
    assert result.works[2].location_qid == "Q2000000"


def test_a_known_place_qid_skips_the_name_search():
    source = FakeSource()
    result = run(lookup_works_set_in(source, "Dublin", place_qid="Q1761"))
    assert result.status == "found"
    assert source.searches == []
    assert not any("wikibase:sitelinks" in q for q in source.queries)


def test_an_invalid_place_qid_is_ignored_and_the_name_is_searched():
    source = FakeSource()
    result = run(lookup_works_set_in(source, "Dublin", place_qid="Q1761 } ; DROP"))
    assert result.status == "found"
    assert source.searches == ["Dublin"]


def test_two_places_of_similar_size_are_skipped_not_guessed():
    rows = [
        {"item": uri("Q1761"), "sitelinks": lit("60"), "coord": lit("Point(0 0)")},
        {"item": uri("Q988937"), "sitelinks": lit("49"), "coord": lit("Point(1 1)")},
    ]
    source = FakeSource(place_rows=rows)
    result = run(lookup_works_set_in(source, "Dublin"))
    assert result.status == "ambiguous"
    assert result.works == []
    assert not any("wdt:P840" in q for q in source.queries)


def test_a_name_with_no_place_is_not_found():
    source = FakeSource(place_rows=[{"item": uri("Q16865980"), "sitelinks": lit("2")}])
    result = run(lookup_works_set_in(source, "Dublin"))
    assert result.status == "not_found"


def test_a_search_with_no_exact_name_is_not_found_and_runs_no_query():
    source = FakeSource(hits=[{"id": "Q173500", "label": "County Dublin"}])
    result = run(lookup_works_set_in(source, "Dublin"))
    assert result.status == "not_found"
    assert source.queries == []


def test_a_place_with_no_works_is_empty():
    source = FakeSource(location_rows=[])
    result = run(lookup_works_set_in(source, "Dublin"))
    assert result.status == "empty"
    assert result.place_qid == "Q1761"


def test_works_that_are_not_written_works_leave_it_empty():
    source = FakeSource(detail_rows=[])
    result = run(lookup_works_set_in(source, "Dublin"))
    assert result.status == "empty"


class Failing(FakeSource):
    def __init__(self, exc: BaseException):
        super().__init__()
        self.exc = exc

    async def sparql(self, query: str):
        raise self.exc


@pytest.mark.parametrize(
    "exc",
    [WikidataError("SPARQL query failed: ReadTimeout"), RuntimeError("boom"), KeyError("results")],
)
def test_any_failure_fails_open_as_unavailable(exc):
    result = run(lookup_works_set_in(Failing(exc), "Dublin"))
    assert result.status == "unavailable"
    assert result.works == []


def test_a_slow_source_is_cut_off_by_the_budget():
    class Slow(FakeSource):
        async def search_items(self, name, limit):
            await asyncio.sleep(1)
            return self.hits

    result = run(lookup_works_set_in(Slow(), "Dublin", budget_seconds=0.05))
    assert result.status == "unavailable"
    assert "no answer within" in result.reason


# --- the templates never carry anything but QIDs ------------------------------


def test_queries_are_fixed_templates_filled_with_qids():
    q = works_by_location_query("Q1761")
    assert "wd:Q1761" in q and "P840" in q and "P131" in q
    assert "wd:Q7725634" in written_work_details_query(["Q6511", "Q1770069"])
    assert "wd:Q1761 wd:Q988937" in place_check_query(["Q1761", "Q988937"])


@pytest.mark.parametrize("bad", ["Dublin", "Q0", "q1761", "Q1761 }", "Q1761\n", "", "P840"])
def test_a_non_qid_never_reaches_a_query(bad):
    with pytest.raises(ValueError):
        works_by_location_query(bad)
    with pytest.raises(ValueError):
        written_work_details_query(["Q6511", bad])
    with pytest.raises(ValueError):
        place_check_query([bad])


# --- parsing --------------------------------------------------------------------


def test_pick_place_needs_a_clear_lead():
    big = PlaceMatch(qid="Q1", label="X", sitelinks=300, has_coordinates=True)
    small = PlaceMatch(qid="Q2", label="X", sitelinks=99, has_coordinates=True)
    close = PlaceMatch(qid="Q3", label="X", sitelinks=101, has_coordinates=True)
    assert pick_place([big, small]) is big
    assert pick_place([big, close]) is None
    assert pick_place([small]) is small
    assert pick_place([]) is None


def test_parse_works_drops_qid_titles_and_malformed_rows():
    location = [{"work": uri("Q1"), "loc": uri("Q1761")}, {"work": uri("Q2"), "loc": uri("Q1761")}]
    details = [
        {"work": uri("Q1"), "workLabel": lit("Q1")},  # no English label: the service echoes the QID
        {"work": uri("Q2"), "workLabel": lit("Dubliners"), "authorLabel": lit("Q42")},  # author without label
        {"work": {"type": "literal", "value": "not a uri"}, "workLabel": lit("Junk")},
        {"workLabel": lit("No work")},
        {"work": uri("Q3"), "workLabel": lit("Not in the location rows")},
    ]
    works = parse_works(location, details)
    assert [(w.work_qid, w.title, w.authors) for w in works] == [("Q2", "Dubliners", [])]


def test_parse_works_caps_the_list():
    location = [{"work": uri(f"Q{i}"), "loc": uri("Q1761")} for i in range(1, 200)]
    details = [{"work": uri(f"Q{i}"), "workLabel": lit(f"Book {i}")} for i in range(1, 200)]
    assert len(parse_works(location, details)) == wd._MAX_WORKS


# --- the HTTP source's shape handling (no network) ----------------------------


class _Resp:
    def __init__(self, status: int, body: Any):
        self.status_code, self._body = status, body

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("bad", request=httpx.Request("GET", "https://x"), response=httpx.Response(self.status_code))

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Client:
    def __init__(self, resp: _Resp):
        self.resp = resp
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url, params=None):
        self.calls.append(("GET", {"url": url, "params": params}))
        return self.resp

    async def post(self, url, data=None, headers=None):
        self.calls.append(("POST", {"url": url, "data": data, "headers": headers}))
        return self.resp


def test_http_source_sends_the_name_only_as_a_search_parameter():
    client = _Client(_Resp(200, {"search": [{"id": "Q1761", "label": "Dublin"}]}))
    hits = run(HttpWikidataSource(client).search_items("Dublin", 7))  # type: ignore[arg-type]
    assert hits == [{"id": "Q1761", "label": "Dublin"}]
    method, call = client.calls[0]
    assert method == "GET" and call["params"]["action"] == "wbsearchentities"
    assert call["params"]["search"] == "Dublin"


@pytest.mark.parametrize(
    "resp",
    [_Resp(503, {}), _Resp(200, ValueError("not json")), _Resp(200, {"head": {}}), _Resp(200, {"results": {"bindings": "x"}})],
)
def test_http_source_turns_bad_answers_into_wikidata_errors(resp):
    with pytest.raises(WikidataError):
        run(HttpWikidataSource(_Client(resp)).sparql("SELECT 1"))  # type: ignore[arg-type]


@pytest.mark.parametrize("resp", [_Resp(500, {}), _Resp(200, {"no": "search"}), _Resp(200, ValueError("x"))])
def test_http_source_search_turns_bad_answers_into_wikidata_errors(resp):
    with pytest.raises(WikidataError):
        run(HttpWikidataSource(_Client(resp)).search_items("Dublin", 7))  # type: ignore[arg-type]


def test_the_default_client_carries_the_descriptive_user_agent():
    source = HttpWikidataSource()
    client = source._http()
    try:
        assert client.headers["User-Agent"] == wd.USER_AGENT
        assert "mystoryland.ai" in wd.USER_AGENT
    finally:
        run(source.aclose())
