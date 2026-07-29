import json
import os

import httpx
import pytest

from generate_books_set_in_corpus import (
    GeneratorError,
    build_page,
    fetch_place_to_book,
    place_slug,
    validate_book,
    validate_place,
)

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "barcelona_backend_response.json")


def load_fixture():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


# -- place_slug: must stay in lockstep with the FE's placeSlug() --

def test_place_slug_basic():
    assert place_slug("Edinburgh") == "edinburgh"
    assert place_slug("New York City") == "new-york-city"
    assert place_slug("Saint Petersburg") == "saint-petersburg"


def test_place_slug_strips_accents_and_punctuation():
    assert place_slug("São Paulo") == "sao-paulo"
    assert place_slug("  Multiple   Spaces ") == "multiple-spaces"
    assert place_slug("") == ""


# -- validate_book / validate_place: mirrors booksSetInPlace.ts, must be build-fatal --

def test_validate_book_rejects_missing_required_fields():
    with pytest.raises(GeneratorError, match="title"):
        validate_book({"author": "A", "why": "w", "cover": "c", "matchType": "vibe"}, "where")


def test_validate_book_rejects_literal_without_mapsto():
    with pytest.raises(GeneratorError, match="mapsTo"):
        validate_book(
            {"title": "T", "author": "A", "why": "w", "cover": "c", "matchType": "literal"},
            "where",
        )


def test_validate_book_allows_vibe_without_mapsto():
    book = validate_book(
        {"title": "T", "author": "A", "why": "w", "cover": "c", "matchType": "vibe"},
        "where",
    )
    assert book["matchType"] == "vibe"


def test_validate_place_rejects_slug_place_mismatch():
    with pytest.raises(GeneratorError, match="does not match its place"):
        validate_place(
            {
                "slug": "paris",
                "place": "London",
                "intro": "intro",
                "books": [{"title": "T", "author": "A", "why": "w", "cover": "c", "matchType": "vibe"}],
            }
        )


def test_validate_place_rejects_empty_books():
    with pytest.raises(GeneratorError, match="books"):
        validate_place({"slug": "paris", "place": "Paris", "intro": "intro", "books": []})


# -- build_page: the real mapping/filtering logic, against real captured data --

def test_build_page_on_real_barcelona_fixture_keeps_all_eight_verified_rows():
    resp = load_fixture()
    report = build_page("Barcelona", resp)
    assert report.verified_book_count == 8
    assert report.dropped_unverified_count == 0
    assert report.page["slug"] == "barcelona"
    assert report.page["place"] == "Barcelona"
    # 5 literal + 3 vibe, matching the live response
    literal = [b for b in report.page["books"] if b["matchType"] == "literal"]
    vibe = [b for b in report.page["books"] if b["matchType"] == "vibe"]
    assert len(literal) == 5
    assert len(vibe) == 3
    # Every literal row carries a mapsTo; no vibe row does (mirrors the fixture).
    assert all(b.get("mapsTo") for b in literal)
    assert all(not b.get("mapsTo") for b in vibe)
    # image_url passed through verbatim from the backend response.
    assert report.page["books"][0]["image_url"] == resp["books"][0]["image_url"]
    # The output must itself pass the FE-mirroring validator (red-before-green
    # for this exact assertion: a bad build_page() would raise here already,
    # since build_page calls validate_place internally -- this call re-checks
    # the returned page independently in case that internal call is ever
    # removed by a future edit).
    validate_place(report.page)


def test_build_page_drops_unverified_rows_never_pads():
    resp = load_fixture()
    # Flip one row to unverified -- it must be dropped, not kept or replaced.
    resp = json.loads(json.dumps(resp))  # deep copy
    resp["books"][-1]["grounding"]["verified"] = False
    report = build_page("Barcelona", resp)
    assert report.verified_book_count == 7
    assert report.dropped_unverified_count == 1
    kept_titles = {b["title"] for b in report.page["books"]}
    assert resp["books"][-1]["title"] not in kept_titles


def test_build_page_refuses_a_city_with_zero_verified_rows():
    resp = load_fixture()
    resp = json.loads(json.dumps(resp))
    for b in resp["books"]:
        b["grounding"]["verified"] = False
    with pytest.raises(GeneratorError, match="0 of"):
        build_page("Barcelona", resp)


def test_fetch_place_to_book_raises_on_found_false():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"place": "Nowhereville", "query": "nowhereville", "found": False, "message": "no candidates", "books": []})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client, pytest.raises(GeneratorError, match="not groundable"):
        fetch_place_to_book("Nowhereville", "https://example.invalid", client)


def test_fetch_place_to_book_calls_the_expected_path_and_query():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json=load_fixture())

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        body = fetch_place_to_book("Barcelona", "https://example.invalid", client)

    assert seen["path"] == "/api/v1/place-to-book"
    assert seen["query"] == {"place": "Barcelona"}
    assert body["found"] is True


# -- CLI: --batch must refuse without ever touching the network --

def test_main_batch_flag_refuses_without_network_call(monkeypatch, capsys):
    from generate_books_set_in_corpus import main

    def fail_if_called(*args, **kwargs):
        raise AssertionError("network must not be touched when --batch is passed")

    monkeypatch.setattr(httpx, "Client", fail_if_called)
    exit_code = main(["--batch", "cities.txt"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "not implemented in this PR" in captured.err


def test_main_requires_city_or_batch():
    from generate_books_set_in_corpus import main

    with pytest.raises(SystemExit):
        main([])
