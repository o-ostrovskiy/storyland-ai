"""Did discovery research the book the reader asked for?

discover() trusts the title/author it is given and hands them to a
google_search researcher. A search does not fail on a title that names no
book: it resolves it to whatever work the web associates with it. "dddd" by
"ddd" came back as Inio Asano's manga "Dead Dead Demon's Dededede Destruction"
(fans write it "DeDeDeDe"), set in Tokyo, and the run produced a Tokyo trip
titled "dddd by ddd" with nothing to tell the reader which book it was for.

The book_context formatter (prompts v6) now reports whether its research
describes the requested book (``matches_requested_book``) and which book it
does describe (``identified_title`` / ``identified_author``). This module
turns that into a verdict discover() acts on before any region is emitted or
cached:

* ``ok``         -- proceed. Also the answer whenever the formatter is silent
                    (pre-v6 cache entries, a null flag): fail open, because
                    blocking a real book the model forgot to label is worse
                    than today's behaviour.
* ``mismatch``   -- the research is about a different, named book: stop and
                    offer it as "did you mean ...?".
* ``not_found``  -- the research found no such book: stop.

A model that says "false" while naming the requested book itself is
overruled (same title, overlapping author name), so a self-contradicting
label cannot block a real book.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Optional

VerdictKind = Literal["ok", "mismatch", "not_found"]

# Bound what a model-written suggestion may put on screen.
_MAX_SUGGESTION_LEN = 200

NOT_FOUND_MESSAGE = (
    "We couldn't find a book with that title and author. "
    "Check the spelling or try another book."
)


def mismatch_message(title: str, author: Optional[str]) -> str:
    by = f" by {author}" if author else ""
    return (
        "We couldn't find that exact book. "
        f'Did you mean "{title}"{by}?'
    )


@dataclass(frozen=True)
class BookIdentityVerdict:
    kind: VerdictKind
    suggested_title: Optional[str] = None
    suggested_author: Optional[str] = None


def _tokens(text: Optional[str]) -> list[str]:
    if not text:
        return []
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.findall(r"\w+", folded)


def _clean(text: object) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.split())
    if not cleaned or len(cleaned) > _MAX_SUGGESTION_LEN:
        return None
    return cleaned


def _same_book(
    requested_title: str,
    requested_author: str,
    identified_title: Optional[str],
    identified_author: Optional[str],
) -> bool:
    """Same title tokens and at least one shared author-name token."""
    if not identified_title or _tokens(identified_title) != _tokens(requested_title):
        return False
    req_author = {t for t in _tokens(requested_author) if len(t) > 1}
    found_author = {t for t in _tokens(identified_author) if len(t) > 1}
    # No author on either side: the title alone has to carry it.
    if not req_author or not found_author:
        return True
    return bool(req_author & found_author)


def check_book_identity(
    book_context: Optional[dict],
    requested_title: str,
    requested_author: str,
) -> BookIdentityVerdict:
    if not isinstance(book_context, dict):
        return BookIdentityVerdict("ok")
    if book_context.get("matches_requested_book") is not False:
        return BookIdentityVerdict("ok")

    title = _clean(book_context.get("identified_title"))
    author = _clean(book_context.get("identified_author"))

    if _same_book(requested_title, requested_author, title, author):
        return BookIdentityVerdict("ok")
    if title:
        return BookIdentityVerdict("mismatch", title, author)
    return BookIdentityVerdict("not_found")
