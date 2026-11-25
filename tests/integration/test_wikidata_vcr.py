"""Wikidata place -> book lookup against recorded real answers (storyland-ai-private#81).

Recorded once from the public endpoints (entity search + query.wikidata.org) on
2026-11-25; replays offline. No key is involved: Wikidata needs none. Re-record
with `make test-integration-vcr-record` when the queries change.
"""

import asyncio

import pytest

from core.wikidata import HttpWikidataSource, lookup_works_set_in

pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def vcr_config(vcr_config):
    """The shared config, plus: keep only the content type of each response.

    Wikimedia's responses set cookies, and one of them (`GeoIP`) carries the
    recording machine's approximate location. None of that belongs in a
    committed cassette, so every response header but `content-type` is dropped,
    and the request's own cookie header with it.
    """

    def keep_only_content_type(response):
        headers = response.get("headers", {})
        response["headers"] = {k: v for k, v in headers.items() if k.lower() == "content-type"}
        return response

    return {
        **vcr_config,
        "before_record_response": keep_only_content_type,
        "filter_headers": [*vcr_config["filter_headers"], "cookie"],
    }


@pytest.mark.vcr
def test_dublin_lookup_replays_real_wikidata_answers():
    async def go():
        source = HttpWikidataSource()
        try:
            return await lookup_works_set_in(source, "Dublin", budget_seconds=60)
        finally:
            await source.aclose()

    result = asyncio.run(go())
    assert result.status == "found", result.reason
    assert result.place_qid == "Q1761"  # Dublin, Ireland (not Ohio)
    titles = {w.title for w in result.works}
    assert "Ulysses" in titles
    ulysses = next(w for w in result.works if w.title == "Ulysses")
    assert ulysses.work_qid == "Q6511"
    assert "James Joyce" in ulysses.authors
    # Every work keeps its QID and its narrative-location QID as receipts.
    assert all(w.work_qid.startswith("Q") and w.location_qid.startswith("Q") for w in result.works)


def test_the_cassette_carries_no_cookie_or_location_header():
    from pathlib import Path

    cassette = Path(__file__).parent / "cassettes" / "test_dublin_lookup_replays_real_wikidata_answers.yaml"
    text = cassette.read_text().lower()
    for needle in ("set-cookie", "geoip", "wmf-uniq", "cookie:"):
        assert needle not in text, needle
