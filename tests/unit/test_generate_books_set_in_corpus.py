import json
import os

import httpx
import pytest

from tools.generate_books_set_in_corpus import (
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


# place_slug: must stay in lockstep with the FE's placeSlug()

def test_place_slug_basic():
    assert place_slug("Edinburgh") == "edinburgh"
    assert place_slug("New York City") == "new-york-city"
    assert place_slug("Saint Petersburg") == "saint-petersburg"


def test_place_slug_strips_accents_and_punctuation():
    assert place_slug("São Paulo") == "sao-paulo"
    assert place_slug("  Multiple   Spaces ") == "multiple-spaces"
    assert place_slug("") == ""


# validate_book / validate_place: mirrors booksSetInPlace.ts, must be build-fatal

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


# build_page: the real mapping/filtering logic, against real captured data

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


# CLI: --batch must refuse without ever touching the network

def test_main_batch_flag_refuses_without_network_call(monkeypatch, capsys):
    from tools.generate_books_set_in_corpus import main

    def fail_if_called(*args, **kwargs):
        raise AssertionError("network must not be touched when --batch is passed")

    monkeypatch.setattr(httpx, "Client", fail_if_called)
    exit_code = main(["--batch", "cities.txt"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "not implemented in this PR" in captured.err


def test_main_requires_city_or_batch():
    from tools.generate_books_set_in_corpus import main

    with pytest.raises(SystemExit):
        main([])


# r2 (fix-list round 1, Engineering Lead)


def _mixed_response(literal: int, vibe: int) -> dict:
    """A minimal PlaceToBook-shaped body with a chosen literal/vibe split.

    Built from the real fixture's row shape rather than invented, so a change to
    the BE contract breaks these tests too instead of letting them drift.
    """
    template = load_fixture()["books"][0]
    books = []
    for i in range(literal):
        row = dict(template, title=f"Literal {i}", match_type="literal", maps_to=f"Anchor {i}")
        books.append(row)
    for i in range(vibe):
        row = dict(template, title=f"Kindred {i}", match_type="vibe")
        row["maps_to"] = None  # exactly what the BE sends for a kindred row
        books.append(row)
    return {"place": "Testville", "query": "Testville", "found": True, "message": None, "books": books}


# item 1: the intro must not claim an anchor the kindred rows do not have


def test_mixed_page_scopes_the_stand_claim_to_the_literal_entries():
    # The reported defect: 3 of Barcelona's 8 verified rows are vibe with
    # maps_to=None, and the intro told every reader "each one points to
    # somewhere you can actually stand".
    report = build_page("Barcelona", load_fixture())
    intro = report.page["intro"]

    assert "the literal entries point to somewhere you can actually stand." in intro
    assert "each one points to" not in intro, (
        "an unscoped claim is false for every kindred row on a mixed page: "
        f"{intro}"
    )


def test_all_literal_page_keeps_the_unscoped_claim_because_it_is_true_there():
    report = build_page("Testville", _mixed_response(literal=4, vibe=0))
    intro = report.page["intro"]

    assert "so each one points to somewhere you can actually stand." in intro
    assert "the literal entries" not in intro


def test_the_scoped_claim_never_overstates_a_row_that_carries_no_map_anchor():
    # Ties the sentence to the DATA rather than to a string: every row the
    # intro claims is standable must actually carry a mapsTo.
    report = build_page("Testville", _mixed_response(literal=2, vibe=3))
    books = report.page["books"]
    anchored = [b for b in books if b.get("mapsTo")]
    unanchored = [b for b in books if not b.get("mapsTo")]

    assert len(unanchored) == 3, "fixture guard: this page must contain unanchored rows"
    assert all(b["matchType"] == "literal" for b in anchored)
    assert all(b["matchType"] == "vibe" for b in unanchored)
    # ...and because unanchored rows exist, the claim is scoped.
    assert "the literal entries point to" in report.page["intro"]


# item 2: the tool's success bar must equal the corpus's publish bar


def test_all_vibe_city_is_refused_because_the_fe_could_never_publish_it():
    # Previously: printed "6 verified book(s) kept", exited 0, produced a page
    # storyland-web's isPublishableBooksSetInPlace (MYS-455) will never publish.
    with pytest.raises(GeneratorError) as exc:
        build_page("Testville", _mixed_response(literal=0, vibe=6))

    message = str(exc.value)
    assert "0 are matchType 'literal'" in message
    assert "isPublishableBooksSetInPlace" in message


def test_a_single_literal_row_is_enough_mirroring_the_fe_gate_exactly():
    # The FE gate is `some(matchType === 'literal')` -- not a ratio. Mirror it,
    # so this tool never refuses a page the corpus would happily publish.
    report = build_page("Testville", _mixed_response(literal=1, vibe=5))
    assert report.verified_book_count == 6


# item 3: a trailing slash on the documented override must not 404


def test_backend_url_trailing_slash_is_stripped_before_the_request(monkeypatch, tmp_path):
    from tools.generate_books_set_in_corpus import main

    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # httpx preserves the doubled slash, so this asserts the real URL path.
        seen["path"] = request.url.path
        return httpx.Response(200, json=load_fixture())

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client  # captured BEFORE patching, or the lambda recurses into itself
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: real_client(transport=transport))

    out = tmp_path / "out.json"
    exit_code = main(["--city", "Barcelona", "--out", str(out), "--backend-url", "http://localhost:8090/"])

    assert exit_code == 0
    assert seen["path"] == "/api/v1/place-to-book", (
        f"a trailing slash produced {seen['path']!r} -- a directly-hosted ASGI app 404s on the doubled slash"
    )
