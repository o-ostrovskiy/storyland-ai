"""
Canonical, cross-job place identity for a grounded travel region (MYS-460).

## Why this exists

Two books can only be combined into one journey if we can tell that a region
discovered for book A **is the same place** as a region discovered for book B.
Nothing on ``RegionOption`` could carry that:

* ``region_id`` is an **ordinal the model emits inside one response** (1, 2, 3),
  validated per-job against that job's own regions. Comparing it across jobs
  equates "the first region we returned for *Gone with the Wind*" with "the
  first for *The Shadow of the Wind*" — i.e. calls Atlanta and Barcelona the
  same city. It is an array index with a colon in front of it. **Never use it as
  an identity.**
* ``region_name`` is prose: ``"Paris & Île-de-France"`` never string-matches
  ``"Paris, France"`` (a combined branch keyed on it would ship and never fire),
  while ``"Paris, Texas"`` vs ``"Paris, France"`` false-positives on a naive
  normalize — a *fabricated* shared setting, on the product whose one
  differentiator is "grounded in real places, never invented".

So the key is minted from **structured geo fields on a region we already
grounded** — an ISO-3166-1 alpha-2 country code plus the region's principal
locality — and from nothing else. ``US:paris`` != ``FR:paris`` falls out for
free, which is the false-positive we most need to lose.

## What this key is, and is not

It is an **anchor identity**: the region's principal locality, canonicalised.
It is *not* a set-equality of the region's whole city list — two regions with the
same anchor may still differ at the edges. That is the right trade for the
intersection: the anchor is what a reader means when they say "both my books are
set in Paris".

Pure stdlib, no pydantic, no core imports — ``core`` imports ``models``, so this
must stay a leaf or the ``models`` package would drag ADK in behind it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ISO 3166-1 alpha-2: exactly two ASCII letters. Anything else (a country NAME,
# an alpha-3, an empty string) is not a country code and must not be keyed on.
_ALPHA2 = re.compile(r"^[A-Za-z]{2}$")

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    """Lowercase ASCII slug: 'Reykjavík' -> 'reykjavik', 'St. Petersburg' -> 'st-petersburg'."""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return _SLUG_STRIP.sub("-", ascii_only.lower()).strip("-")


def mint_place_key(
    country_code: Optional[str], primary_locality: Optional[str]
) -> Optional[str]:
    """Mint the canonical ``"<cc>:<locality-slug>"`` key, or ``None``.

    ``None`` means *we do not know this region's identity* — and that is the only
    honest answer when the grounded fields are missing. It must NEVER fall back
    to a normalized ``region_name``: a key derived from prose is exactly the
    fabrication this module exists to prevent, and a "just for v1" fallback is
    indistinguishable from a real key downstream.

    A region with no key simply cannot participate in an intersection. That is a
    missed combine — never a wrong one.
    """
    if not isinstance(country_code, str) or not _ALPHA2.match(country_code.strip()):
        return None
    if not isinstance(primary_locality, str):
        return None

    locality = _slug(primary_locality)
    if not locality:
        return None

    return f"{country_code.strip().lower()}:{locality}"
