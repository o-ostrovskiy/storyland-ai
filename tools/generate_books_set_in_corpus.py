"""
books-set-in corpus generator (MYS-431, PR2 of MYS-204).

Offline, one-off batch job that produces per-city book<->place records
conforming to storyland-web's `booksSetInPlace` schema (`src/data/
booksSetInPlace.ts`'s `validatePlace`/`validateBook`, mirrored in
`validate_place`/`validate_book` below).

Design decision, load-bearing: this generator does NOT call Gemini directly
and does NOT duplicate ai-side grounding. It calls the backend's already-live,
already-grounded, already-existence-verified `GET /api/v1/place-to-book`
endpoint (storyland-services `app/place_to_book`), which already runs the
`place_to_book_agent` researcher/formatter pipeline AND decorates every
candidate with a real Google Books / Open Library existence check
(`PlaceBook.grounding.verified`). Re-implementing that pipeline here would
duplicate a paid Gemini call this app already makes for real users, and would
violate the standing rule that book-existence verification lives on the BE via
Google Books, not ai-side (see docs/CLAUDE.md and models/place_to_book.py's own
docstring). One HTTP call to an already-deployed, already-costed endpoint is
the whole "generation" step; this script's only real job is validating and
reshaping that response into the FE's corpus schema.

The bar, restated because it is the whole point (per the ticket): "We'd rather
show fewer titles than invent one." Only rows with `grounding.verified is
True` are kept -- an ungrounded/unverified candidate is dropped, never padded
in. A city that ends up with zero verified rows is refused outright (an
empty `books[]` is build-fatal on the FE loader anyway; better to fail loudly
here than ship a thin or empty page). A city whose verified rows are ALL
kindred (`vibe`) is refused too: storyland-web's `isPublishableBooksSetInPlace`
(MYS-455) requires at least one `literal` row before a page may be published,
so "verified" alone is this tool's bar, not the corpus's.

Usage:
    python generate_books_set_in_corpus.py --city Barcelona --out out.json
    python generate_books_set_in_corpus.py --batch cities.txt   # NOT implemented in this PR (see main())

Environment:
    BACKEND_URL - base URL of storyland-services (default:
        https://mystoryland.ai). No GOOGLE_API_KEY or any Gemini credential
        is read by this script -- see the design note above.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BACKEND_URL = "https://mystoryland.ai"
PLACE_TO_BOOK_PATH = "/api/v1/place-to-book"
REQUEST_TIMEOUT_SECONDS = 30.0

# Cycled by book index within a page, matching the existing 11-city seed
# corpus's rotating placeholder palette (src/data/booksSetInPlace.json) so a
# generated page visually matches the hand-authored ones until a real cover
# image is resolved client-side (BookCover lookupOnMissing) or `image_url`
# (passed through from the BE response below) is used instead.
COVER_GRADIENT_PALETTE: tuple[str, ...] = (
    "linear-gradient(135deg,#4a5e6b,#2c3840)",
    "linear-gradient(135deg,#6b4a5e,#402c38)",
    "linear-gradient(135deg,#4a6b4f,#2c402f)",
    "linear-gradient(135deg,#6b6f4a,#40422c)",
    "linear-gradient(135deg,#3f3550,#262035)",
    "linear-gradient(135deg,#5e4a4a,#382c2c)",
    "linear-gradient(135deg,#4a566b,#2c3340)",
)


class GeneratorError(Exception):
    """Build-fatal error: the generator refuses to write non-conformant output."""


def place_slug(name: str) -> str:
    """Lowercase, hyphenated, ASCII-only slug for a place name.

    MUST stay byte-for-byte in lockstep with storyland-web's
    `src/data/booksSetInPlace.ts::placeSlug()` -- the FE loader rejects any
    record whose `slug` isn't exactly `placeSlug(place)` (MYS-204's build-time
    gate for generated rows). Re-implemented here rather than shared because
    this is a different repo/language. There is deliberately no cross-repo pin
    here -- CI cannot run both sides -- so the real guard is that the FE's
    `validatePlace` is build-fatal on a slug mismatch. (Corrected in r2: this
    docstring used to claim a shared-fixture test that does not exist.)
    """
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", name)
    # Strip combining marks (accents) the way the FE's regex range does.
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = stripped.lower()
    hyphenated = re.sub(r"[^a-z0-9]+", "-", lowered)
    return hyphenated.strip("-")


def _is_nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def validate_book(raw: dict[str, Any], where: str) -> dict[str, Any]:
    """Mirror of booksSetInPlace.ts's validateBook. Raises GeneratorError on
    any violation -- if this generator's output can't pass this check, it
    can't pass the FE build either, so fail here first."""
    if not isinstance(raw, dict):
        raise GeneratorError(f"{where} is not an object")
    if not _is_nonempty_str(raw.get("title")):
        raise GeneratorError(f'{where} has a missing/empty "title"')
    if not _is_nonempty_str(raw.get("author")):
        raise GeneratorError(f'{where} has a missing/empty "author"')
    if not _is_nonempty_str(raw.get("why")):
        raise GeneratorError(f'{where} has a missing/empty "why"')
    if not _is_nonempty_str(raw.get("cover")):
        raise GeneratorError(f'{where} has a missing/empty "cover"')
    match_type = raw.get("matchType")
    if match_type not in ("literal", "vibe"):
        raise GeneratorError(f'{where} has an invalid "matchType" (expected literal|vibe)')
    maps_to = raw.get("mapsTo")
    if match_type == "literal" and not _is_nonempty_str(maps_to):
        raise GeneratorError(
            f"{where} has matchType 'literal' but missing or empty mapsTo -- "
            "literal rows require a map anchor"
        )
    if maps_to is not None and not _is_nonempty_str(maps_to):
        raise GeneratorError(f'{where} has an empty "mapsTo"')
    image_url = raw.get("image_url")
    if image_url is not None and not _is_nonempty_str(image_url):
        raise GeneratorError(f'{where} has an empty "image_url"')
    return raw


def validate_place(raw: dict[str, Any]) -> dict[str, Any]:
    """Mirror of booksSetInPlace.ts's validatePlace (single-record form, since
    this generator emits one record per run)."""
    if not isinstance(raw, dict):
        raise GeneratorError("record is not an object")
    slug = raw.get("slug")
    place = raw.get("place")
    if not _is_nonempty_str(slug):
        raise GeneratorError('record has a missing/empty "slug"')
    if not _is_nonempty_str(place):
        raise GeneratorError('record has a missing/empty "place"')
    expected_slug = place_slug(place.strip())
    if slug.strip() != expected_slug:
        raise GeneratorError(
            f'record has a slug "{slug}" that does not match its place "{place}" -- '
            f'expected "{expected_slug}"'
        )
    if not _is_nonempty_str(raw.get("intro")):
        raise GeneratorError('record has a missing/empty "intro"')
    books = raw.get("books")
    if not isinstance(books, list) or len(books) == 0:
        raise GeneratorError('record has an empty or missing "books" array')
    for i, b in enumerate(books):
        validate_book(b, f'record "{slug}" book #{i}')
    return raw


@dataclass
class GenerationReport:
    """Measured facts about one generator run, for the founder spot-review."""

    city: str
    verified_book_count: int
    dropped_unverified_count: int
    backend_wall_clock_seconds: float
    page: dict[str, Any] = field(repr=False)


def fetch_place_to_book(city: str, backend_url: str, client: httpx.Client) -> dict[str, Any]:
    """Call the backend's already-live, already-grounded, already-existence-
    verified place->book endpoint. Raises on a non-200 or a `found: false`
    response (an ungroundable city -- refuse rather than ship an empty page)."""
    resp = client.get(
        f"{backend_url}{PLACE_TO_BOOK_PATH}",
        params={"place": city},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    body = resp.json()
    if not body.get("found"):
        raise GeneratorError(
            f'"{city}" was not groundable by the backend '
            f'(found=false, message={body.get("message")!r}) -- dropping rather than padding'
        )
    return body


def build_page(city: str, backend_response: dict[str, Any]) -> GenerationReport:
    """Reshape a PlaceToBookResponse-shaped dict into a validated
    BooksSetInPlacePage-shaped dict, keeping only grounding-verified rows."""
    raw_books = backend_response.get("books", [])
    verified = [b for b in raw_books if isinstance(b, dict) and b.get("grounding", {}).get("verified") is True]
    dropped = len(raw_books) - len(verified)

    if not verified:
        raise GeneratorError(
            f'"{city}": 0 of {len(raw_books)} candidate(s) passed the grounding.verified '
            "bar -- refusing to ship an empty/thin page rather than pad it"
        )

    place = backend_response.get("place") or city
    slug = place_slug(place)

    books: list[dict[str, Any]] = []
    for i, b in enumerate(verified):
        book: dict[str, Any] = {
            "title": b["title"],
            "author": b["author"],
            "matchType": b["match_type"],
            "why": b["why_it_fits"],
            "cover": COVER_GRADIENT_PALETTE[i % len(COVER_GRADIENT_PALETTE)],
        }
        if b.get("maps_to"):
            book["mapsTo"] = b["maps_to"]
        if b.get("image_url"):
            book["image_url"] = b["image_url"]
        books.append(book)

    literal_titles = [b["title"] for b in books if b["matchType"] == "literal"]
    vibe_count = len(books) - len(literal_titles)

    # MYS-431 r1 (Eng Lead), item 2: mirror storyland-web's SECOND, stricter
    # gate. The verified bar above is this tool's bar; `isPublishableBooksSetInPlace`
    # (booksSetInPlace.ts, MYS-455) is the CORPUS's bar -- a page needs at least
    # one LITERAL row, because the h1, <title> and ItemList JSON-LD all assert a
    # "set in <place>" relationship an all-kindred page does not contain. Without
    # this, a city returning 6 verified-but-all-vibe rows printed "6 verified
    # book(s) kept", exited 0, and produced a page the FE will never publish --
    # invisible at --city scale, silent at 90 cities.
    if not literal_titles:
        raise GeneratorError(
            f'"{city}": {len(books)} verified row(s) kept but 0 are matchType '
            "'literal' -- storyland-web's isPublishableBooksSetInPlace (MYS-455) "
            "refuses an all-kindred page, so this city is unpublishable: its h1 and "
            "ItemList would assert a set-in relationship the corpus does not contain"
        )

    # Deterministic, no-extra-LLM-call intro: a template, not a second Gemini
    # pass.
    #
    # MYS-431 r1 (Eng Lead), item 1: the closing clause must not claim a
    # property the rows do not have. A `vibe` row legitimately carries no
    # `mapsTo` at all, so on a mixed page the unscoped "each one points to
    # somewhere you can actually stand" is FALSE for every kindred row -- the
    # render-"we-don't-know"-as-a-positive-claim class (MYS-584/MYS-492), and
    # worse here because these are indexable pages whose whole proposition is
    # that we checked. `greece` is the one seed page that solved it, by scoping
    # the clause to "the literal entries"; that form is copied here for any page
    # carrying kindred rows. An all-literal page keeps the unscoped form, which
    # is honest for it. (The zero-literal case can no longer reach this code.)
    opening = f"{literal_titles[0]} is among the books genuinely set in {place}."
    if vibe_count:
        claim = "so the literal entries point to somewhere you can actually stand."
    else:
        claim = "so each one points to somewhere you can actually stand."
    intro = (
        f"{opening} The books below are drawn from Storyland's book↔place "
        f"engine and checked against real locations, {claim}"
    )

    page = {
        "slug": slug,
        "place": place,
        "intro": intro,
        "books": books,
    }
    validate_place(page)

    return GenerationReport(
        city=city,
        verified_book_count=len(books),
        dropped_unverified_count=dropped,
        backend_wall_clock_seconds=0.0,  # filled in by the CLI, which times the HTTP call
        page=page,
    )


def run_one_city(city: str, backend_url: str) -> GenerationReport:
    import time

    with httpx.Client() as client:
        t0 = time.monotonic()
        backend_response = fetch_place_to_book(city, backend_url, client)
        elapsed = time.monotonic() - t0
    report = build_page(city, backend_response)
    report.backend_wall_clock_seconds = elapsed
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", help="Single city to generate (e.g. 'Barcelona')")
    parser.add_argument(
        "--batch",
        help="Path to a newline-separated city list for a full batch run",
    )
    parser.add_argument("--out", default="books_set_in_generated.json", help="Output JSON path")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    args = parser.parse_args(argv)
    # MYS-431 r1 (Eng Lead), item 3 (from Codex): `--backend-url http://host:8090/`
    # would request `//api/v1/place-to-book`, which a directly-hosted ASGI app
    # treats as a distinct path and 404s. The default carries no trailing slash,
    # so this never fired in the Barcelona run -- it only breaks the documented
    # override, for the person most likely to use it (a local backend).
    args.backend_url = args.backend_url.rstrip("/")

    if args.batch:
        # Deliberately not implemented in this PR. MYS-431's own scope note:
        # "Run the generator against ONE city first, report the real
        # per-city token/$ figure, and stop. Do not fire the 40-90-city batch
        # in the same run." Batch mode is a separate PR once the founder has
        # reviewed this run's sample + cost figure and the city list is
        # agreed.
        print(
            "Batch mode is intentionally not implemented in this PR (MYS-431 scope: "
            "one city, report cost, stop). Run --city repeatedly once the founder "
            "has agreed the city list for the batch PR.",
            file=sys.stderr,
        )
        return 2

    if not args.city:
        parser.error("either --city or --batch is required")

    report = run_one_city(args.city, args.backend_url)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([report.page], f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f'"{args.city}": {report.verified_book_count} verified book(s) kept, '
        f"{report.dropped_unverified_count} unverified dropped, "
        f"backend call {report.backend_wall_clock_seconds:.2f}s -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
