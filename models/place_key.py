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

# The REAL ISO 3166-1 alpha-2 set — every code the standard has assigned.
# `^[A-Za-z]{2}$` (the shape-only check this replaces) accepts ANY two
# letters: "UK", "EU", "ZZ" all pass it. "UK" is not an ISO code — "GB" is —
# and it is the single most likely thing an LLM emits for a British region
# (London, Edinburgh, Dublin — our top collections). Validating MEMBERSHIP,
# not just shape, is what keeps "uk:london" and "gb:london" from being two
# different keys for the same city.
_ISO_3166_1_ALPHA2: frozenset[str] = frozenset({
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT",
    "AU", "AW", "AX", "AZ",
    "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BL", "BM", "BN",
    "BO", "BQ", "BR", "BS", "BT", "BV", "BW", "BY", "BZ",
    "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN", "CO",
    "CR", "CU", "CV", "CW", "CX", "CY", "CZ",
    "DE", "DJ", "DK", "DM", "DO", "DZ",
    "EC", "EE", "EG", "EH", "ER", "ES", "ET",
    "FI", "FJ", "FK", "FM", "FO", "FR",
    "GA", "GB", "GD", "GE", "GF", "GG", "GH", "GI", "GL", "GM", "GN", "GP",
    "GQ", "GR", "GS", "GT", "GU", "GW", "GY",
    "HK", "HM", "HN", "HR", "HT", "HU",
    "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS", "IT",
    "JE", "JM", "JO", "JP",
    "KE", "KG", "KH", "KI", "KM", "KN", "KP", "KR", "KW", "KY", "KZ",
    "LA", "LB", "LC", "LI", "LK", "LR", "LS", "LT", "LU", "LV", "LY",
    "MA", "MC", "MD", "ME", "MF", "MG", "MH", "MK", "ML", "MM", "MN", "MO",
    "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY", "MZ",
    "NA", "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", "NR", "NU", "NZ",
    "OM",
    "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PM", "PN", "PR", "PS", "PT",
    "PW", "PY",
    "QA",
    "RE", "RO", "RS", "RU", "RW",
    "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM",
    "SN", "SO", "SR", "SS", "ST", "SV", "SX", "SY", "SZ",
    "TC", "TD", "TF", "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TO", "TR",
    "TT", "TV", "TW", "TZ",
    "UA", "UG", "UM", "US", "UY", "UZ",
    "VA", "VC", "VE", "VG", "VI", "VN", "VU",
    "WF", "WS",
    "YE", "YT",
    "ZA", "ZM", "ZW",
})

# The two exceptionally-reserved aliases an LLM will actually produce in
# prose: the UK government's own usage is "UK" everywhere (the ISO code is
# "GB"), and "EL" is the EU's own reservation for Greece (the ISO code is
# "GR"). Anything else outside the real set above is a GUESS, not an alias,
# and must still yield no key — normalising an alias is not the same as
# accepting anything two letters long.
_ALPHA2_ALIASES: dict[str, str] = {"UK": "GB", "EL": "GR"}

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _canonical_country_code(country_code: str) -> Optional[str]:
    """Upper-case, alias-normalise, then require REAL ISO-3166-1 membership."""
    code = country_code.strip().upper()
    code = _ALPHA2_ALIASES.get(code, code)
    if code not in _ISO_3166_1_ALPHA2:
        return None
    return code


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
    if not isinstance(country_code, str):
        return None
    code = _canonical_country_code(country_code)
    if code is None:
        return None
    if not isinstance(primary_locality, str):
        return None

    locality = _slug(primary_locality)
    if not locality:
        return None

    return f"{code.lower()}:{locality}"
