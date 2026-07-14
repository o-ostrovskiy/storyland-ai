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

# Country NAME (or, since MYS-460 review, an alpha-2 CODE/alias -- see
# resolve_country_name below) -> ISO alpha-2, used ONLY to cross-check a
# matched city's own `country` field against the region's `country_code`
# (self-consistency, not a general gazetteer). Covers the full ISO 3166-1
# English short name for every code in _ISO_3166_1_ALPHA2, plus the common
# variant spellings a model actually emits (USA, UK, Holland, Turkiye...).
#
# A name that is NOT in this map does not resolve -- and an unresolvable
# name is TOLERATED (the mint proceeds), never rejected. Rejecting on an
# unrecognised spelling would silently stop the feature firing, which is
# the same "ships and never fires" failure the cache-namespace bump exists
# to prevent -- the asymmetry here is the same one `mint_place_key` stands
# on: a missed combine is cheap, a wrong one is not, but a feature with no
# combines at all is also a failure.
#
# 🔴 Before MYS-460 review round 5, this table had ~50 entries -- Chile,
# Peru, Colombia, Kenya, Israel and most of the world were simply not in
# it, so `cities[].country: "Chile"` on a region claiming `country_code:
# "ES"` resolved to None -> tolerated -> minted `es:santiago` for a
# Chilean city. A short table doesn't make the check honest; it makes
# "unresolvable" mean "not in the fifty I typed", not "genuinely
# unrecognisable". Completed to the full set so tolerate-on-unresolvable
# is finally the honest branch it was always meant to be.
# 311 name/alias entries covering all 249 codes in _ISO_3166_1_ALPHA2.
_COUNTRY_NAME_TO_ALPHA2: dict[str, str] = {
    "andorra": "AD",
    "united arab emirates": "AE", "uae": "AE", "u.a.e.": "AE",
    "afghanistan": "AF",
    "antigua and barbuda": "AG", "antigua": "AG", "barbuda": "AG",
    "anguilla": "AI",
    "albania": "AL",
    "armenia": "AM",
    "angola": "AO",
    "antarctica": "AQ",
    "argentina": "AR",
    "american samoa": "AS",
    "austria": "AT",
    "australia": "AU",
    "aruba": "AW",
    "aland islands": "AX", "åland islands": "AX",
    "azerbaijan": "AZ",
    "bosnia and herzegovina": "BA", "bosnia": "BA",
    "barbados": "BB",
    "bangladesh": "BD",
    "belgium": "BE",
    "burkina faso": "BF",
    "bulgaria": "BG",
    "bahrain": "BH",
    "burundi": "BI",
    "benin": "BJ",
    "saint barthelemy": "BL", "saint barthélemy": "BL", "st barthelemy": "BL",
    "bermuda": "BM",
    "brunei": "BN", "brunei darussalam": "BN",
    "bolivia": "BO",
    "bonaire, sint eustatius and saba": "BQ", "bonaire": "BQ",
    "brazil": "BR",
    "bahamas": "BS", "the bahamas": "BS",
    "bhutan": "BT",
    "bouvet island": "BV",
    "botswana": "BW",
    "belarus": "BY",
    "belize": "BZ",
    "canada": "CA",
    "cocos islands": "CC", "cocos (keeling) islands": "CC",
    "democratic republic of the congo": "CD", "congo (drc)": "CD", "dr congo": "CD", "drc": "CD",
    "central african republic": "CF",
    "republic of the congo": "CG", "congo": "CG",
    "switzerland": "CH",
    "cote d'ivoire": "CI", "côte d'ivoire": "CI", "ivory coast": "CI",
    "cook islands": "CK",
    "chile": "CL",
    "cameroon": "CM",
    "china": "CN",
    "colombia": "CO",
    "costa rica": "CR",
    "cuba": "CU",
    "cabo verde": "CV", "cape verde": "CV",
    "curacao": "CW", "curaçao": "CW",
    "christmas island": "CX",
    "cyprus": "CY",
    "czechia": "CZ", "czech republic": "CZ",
    "germany": "DE",
    "djibouti": "DJ",
    "denmark": "DK",
    "dominica": "DM",
    "dominican republic": "DO",
    "algeria": "DZ",
    "ecuador": "EC",
    "estonia": "EE",
    "egypt": "EG",
    "western sahara": "EH",
    "eritrea": "ER",
    "spain": "ES",
    "ethiopia": "ET",
    "finland": "FI",
    "fiji": "FJ",
    "falkland islands": "FK", "the falkland islands": "FK",
    "micronesia": "FM",
    "faroe islands": "FO",
    "france": "FR",
    "gabon": "GA",
    "uk": "GB", "u.k.": "GB", "united kingdom": "GB", "great britain": "GB", "britain": "GB", "england": "GB", "scotland": "GB", "wales": "GB", "northern ireland": "GB",
    "grenada": "GD",
    "georgia": "GE",
    "french guiana": "GF",
    "guernsey": "GG",
    "ghana": "GH",
    "gibraltar": "GI",
    "greenland": "GL",
    "gambia": "GM", "the gambia": "GM",
    "guinea": "GN",
    "guadeloupe": "GP",
    "equatorial guinea": "GQ",
    "greece": "GR", "hellenic republic": "GR",
    "south georgia and the south sandwich islands": "GS",
    "guatemala": "GT",
    "guam": "GU",
    "guinea-bissau": "GW",
    "guyana": "GY",
    "hong kong": "HK",
    "heard island and mcdonald islands": "HM",
    "honduras": "HN",
    "croatia": "HR",
    "haiti": "HT",
    "hungary": "HU",
    "indonesia": "ID",
    "ireland": "IE", "republic of ireland": "IE",
    "israel": "IL",
    "isle of man": "IM",
    "india": "IN",
    "british indian ocean territory": "IO",
    "iraq": "IQ",
    "iran": "IR",
    "iceland": "IS",
    "italy": "IT",
    "jersey": "JE",
    "jamaica": "JM",
    "jordan": "JO",
    "japan": "JP",
    "kenya": "KE",
    "kyrgyzstan": "KG",
    "cambodia": "KH",
    "kiribati": "KI",
    "comoros": "KM",
    "saint kitts and nevis": "KN", "st kitts and nevis": "KN",
    "north korea": "KP",
    "south korea": "KR", "republic of korea": "KR", "korea": "KR",
    "kuwait": "KW",
    "cayman islands": "KY",
    "kazakhstan": "KZ",
    "laos": "LA",
    "lebanon": "LB",
    "saint lucia": "LC", "st lucia": "LC",
    "liechtenstein": "LI",
    "sri lanka": "LK",
    "liberia": "LR",
    "lesotho": "LS",
    "lithuania": "LT",
    "luxembourg": "LU",
    "latvia": "LV",
    "libya": "LY",
    "morocco": "MA",
    "monaco": "MC",
    "moldova": "MD",
    "montenegro": "ME",
    "saint martin": "MF", "st martin": "MF",
    "madagascar": "MG",
    "marshall islands": "MH",
    "north macedonia": "MK", "macedonia": "MK",
    "mali": "ML",
    "myanmar": "MM", "burma": "MM",
    "mongolia": "MN",
    "macao": "MO", "macau": "MO",
    "northern mariana islands": "MP",
    "martinique": "MQ",
    "mauritania": "MR",
    "montserrat": "MS",
    "malta": "MT",
    "mauritius": "MU",
    "maldives": "MV",
    "malawi": "MW",
    "mexico": "MX",
    "malaysia": "MY",
    "mozambique": "MZ",
    "namibia": "NA",
    "new caledonia": "NC",
    "niger": "NE",
    "norfolk island": "NF",
    "nigeria": "NG",
    "nicaragua": "NI",
    "netherlands": "NL", "the netherlands": "NL", "holland": "NL",
    "norway": "NO",
    "nepal": "NP",
    "nauru": "NR",
    "niue": "NU",
    "new zealand": "NZ",
    "oman": "OM",
    "panama": "PA",
    "peru": "PE",
    "french polynesia": "PF",
    "papua new guinea": "PG",
    "philippines": "PH", "the philippines": "PH",
    "pakistan": "PK",
    "poland": "PL",
    "saint pierre and miquelon": "PM",
    "pitcairn": "PN", "pitcairn islands": "PN",
    "puerto rico": "PR",
    "palestine": "PS", "state of palestine": "PS",
    "portugal": "PT",
    "palau": "PW",
    "paraguay": "PY",
    "qatar": "QA",
    "reunion": "RE", "réunion": "RE",
    "romania": "RO",
    "serbia": "RS",
    "russia": "RU", "russian federation": "RU",
    "rwanda": "RW",
    "saudi arabia": "SA",
    "solomon islands": "SB",
    "seychelles": "SC",
    "sudan": "SD",
    "sweden": "SE",
    "singapore": "SG",
    "saint helena": "SH", "st helena": "SH",
    "slovenia": "SI",
    "svalbard and jan mayen": "SJ",
    "slovakia": "SK",
    "sierra leone": "SL",
    "san marino": "SM",
    "senegal": "SN",
    "somalia": "SO",
    "suriname": "SR",
    "south sudan": "SS",
    "sao tome and principe": "ST", "são tomé and príncipe": "ST",
    "el salvador": "SV",
    "sint maarten": "SX",
    "syria": "SY",
    "eswatini": "SZ", "swaziland": "SZ",
    "turks and caicos islands": "TC",
    "chad": "TD",
    "french southern territories": "TF",
    "togo": "TG",
    "thailand": "TH",
    "tajikistan": "TJ",
    "tokelau": "TK",
    "timor-leste": "TL", "east timor": "TL",
    "turkmenistan": "TM",
    "tunisia": "TN",
    "tonga": "TO",
    "turkey": "TR", "türkiye": "TR", "turkiye": "TR",
    "trinidad and tobago": "TT",
    "tuvalu": "TV",
    "taiwan": "TW",
    "tanzania": "TZ",
    "ukraine": "UA",
    "uganda": "UG",
    "united states minor outlying islands": "UM",
    "usa": "US", "u.s.a.": "US", "united states": "US", "united states of america": "US", "america": "US",
    "uruguay": "UY",
    "uzbekistan": "UZ",
    "vatican city": "VA", "holy see": "VA",
    "saint vincent and the grenadines": "VC", "st vincent and the grenadines": "VC",
    "venezuela": "VE",
    "british virgin islands": "VG",
    "united states virgin islands": "VI", "us virgin islands": "VI",
    "vietnam": "VN", "viet nam": "VN",
    "vanuatu": "VU",
    "wallis and futuna": "WF",
    "samoa": "WS",
    "yemen": "YE",
    "mayotte": "YT",
    "south africa": "ZA",
    "zambia": "ZM",
    "zimbabwe": "ZW",
}


# A model emits abbreviations WITH internal periods -- "U.S.", "U.K." -- that
# are otherwise a valid alpha-2 code or name-table entry once the punctuation
# is gone: "U.S." stripped of periods is "US", a real ISO code, but the raw
# string is neither 2 bare letters nor a table key ("u.s.a." is in the table;
# "u.s." on its own, without the trailing "a", was not -- MYS-460 review
# round 6), so it fell through to unresolvable -> tolerated -> minted. Only
# periods are stripped: hyphens are load-bearing in real country names
# (Timor-Leste, Guinea-Bissau, Bosnia-Herzegovina) and must survive.
_PERIOD = "."


def _normalise_punctuation(value: str) -> str:
    """Strip periods before country-code/name resolution. See module note above."""
    return value.replace(_PERIOD, "")


def resolve_country_name(name: object) -> Optional[str]:
    """Best-effort country NAME (or CODE) -> ISO alpha-2. Unresolved -> None (tolerate).

    `RegionCity.country` is documented as a name, but a model will sometimes
    emit an alpha-2 code (or the UK/EL alias) there instead -- "US" rather
    than "United States". Resolving that FIRST through the same ISO
    membership check `mint_place_key` uses closes that off completely: a
    literal code input no longer falls through to "not in the name table"
    -> tolerated -> a check that is silently name-only again for exactly
    the input it most needs to catch (MYS-460 review).

    Punctuation-normalised (periods stripped) BEFORE both the code check and
    the name-table lookup, so "U.S." and "U.K." resolve the same way their
    unpunctuated spellings already do, instead of silently missing the table
    by one character (MYS-460 review round 6).

    None is the safe direction on both ends of this function: an unrecognised
    spelling is not evidence of a mismatch, so callers must treat it as "could
    not check" rather than "checked and failed".
    """
    if not isinstance(name, str):
        return None
    stripped = name.strip()
    if not stripped:
        return None
    normalised = _normalise_punctuation(stripped)
    as_code = _canonical_country_code(normalised)
    if as_code is not None:
        return as_code
    return _COUNTRY_NAME_TO_ALPHA2.get(normalised.lower())


_SLUG_STRIP = re.compile(r"[^a-z0-9]+")

# Latin letters NFKD does NOT decompose into base + combining mark, so the
# ascii-encode/ignore pass that follows DELETES them outright instead of
# reducing them to an ascii base letter the way it does for e.g. i -> i.
# That is the same city-collapses-to-two-keys defect as UK/GB, one layer
# down: "Łódź" -> "odz" (pl:odz) while "Lodz" -> "lodz" (pl:lodz); "Tromsø"
# -> "troms" (no:troms) while "Tromso" -> "tromso". Transliterate BEFORE the
# NFKD/ascii pass so these letters degrade to a real ascii letter instead of
# vanishing.
_TRANSLITERATIONS: dict[str, str] = {
    "Ł": "L", "ł": "l",
    "Ø": "O", "ø": "o",
    "Đ": "D", "đ": "d",
    "Ð": "D", "ð": "d",
    "Þ": "Th", "þ": "th",
    "ß": "ss",
    "Æ": "Ae", "æ": "ae",
    "Œ": "Oe", "œ": "oe",
    "Ħ": "H", "ħ": "h",
    "Ŀ": "L", "ŀ": "l",
}


def _transliterate(value: str) -> str:
    """Replace letters NFKD leaves untouched, before the ascii pass drops them."""
    return "".join(_TRANSLITERATIONS.get(ch, ch) for ch in value)


def _canonical_country_code(country_code: str) -> Optional[str]:
    """Upper-case, alias-normalise, then require REAL ISO-3166-1 membership."""
    code = country_code.strip().upper()
    code = _ALPHA2_ALIASES.get(code, code)
    if code not in _ISO_3166_1_ALPHA2:
        return None
    return code


def slug(value: str) -> str:
    """Lowercase ASCII slug: 'Reykjavík' -> 'reykjavik', 'Łódź' -> 'lodz'.

    Public: the self-consistency check in core/regions.py (primary_locality
    must be one of the region's own `cities`) needs to slug-compare the two
    fields with the exact same function that mints the key, or a spelling
    difference the key-minting side tolerates becomes a false mismatch.
    """
    transliterated = _transliterate(value)
    decomposed = unicodedata.normalize("NFKD", transliterated)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return _SLUG_STRIP.sub("-", ascii_only.lower()).strip("-")


def locality_matches_cities(
    primary_locality: Optional[str],
    cities,
    country_code: Optional[str] = None,
) -> bool:
    """Self-consistency: does `primary_locality` match one of the region's own
    `cities` -- on the PAIR (name AND country), not on name alone.

    ``models/discovery.py`` states this as a MUST in prose (`primary_locality`
    must be one of the cities listed in `cities`) and nothing enforced it. A
    region grouping Bath/Winchester that emits `primary_locality: "London"`
    would otherwise mint a valid-looking `gb:london` that WRONGLY intersects
    with a real London region -- the one outcome this whole design forbids.
    This check costs nothing: both fields are already in the same model
    response, so it can never be starved of the data it needs.

    Matching on NAME alone is not enough. A region with `country_code: "US"`
    and `primary_locality: "Paris"` would match a `cities` entry
    `{"name": "Paris", "country": "France"}` on name alone and mint
    `us:paris` -- the key of Paris, TEXAS -- for a region that is actually in
    France. Ambiguous city names shared by two countries are not exotic:
    Paris, Odessa, Cambridge, Melbourne, Athens, St Petersburg all exist in
    more than one, and they are literary cities, not edge cases.

    So once a city's NAME matches, its own `country` field (if present and
    resolvable) is cross-checked against `country_code`:
    - names match AND countries agree (or the city's country can't be
      resolved to a code) -> match.
    - names match but the resolved countries DISAGREE -> a demonstrated
      mismatch; keep scanning the rest of `cities` for another match instead
      of failing outright, since a duplicate-named city elsewhere in the same
      list could still be the right one.

    An UNRESOLVABLE country name is deliberately NOT a mismatch: rejecting on
    a spelling this module doesn't recognise would cost a missed combine for
    no reason -- the same "ships and never fires" failure the cache-namespace
    bump exists to prevent.

    A locality that fails this check yields no key: a missed combine, never a
    wrong one -- the same asymmetry `mint_place_key` already stands on.
    """
    if not isinstance(primary_locality, str):
        return False
    target = slug(primary_locality)
    if not target:
        return False
    region_alpha2 = (
        _canonical_country_code(country_code)
        if isinstance(country_code, str)
        else None
    )
    for city in cities or ():
        if not isinstance(city, dict):
            continue
        name = city.get("name")
        if not (isinstance(name, str) and slug(name) == target):
            continue
        if region_alpha2 is not None:
            resolved = resolve_country_name(city.get("country"))
            if resolved is not None and resolved != region_alpha2:
                continue  # demonstrated mismatch -- not this city, keep scanning
        return True
    return False


def mint_checked_place_key(
    country_code: Optional[str],
    primary_locality: Optional[str],
    cities,
) -> Optional[str]:
    """The ONE checked seam: self-consistency (name+country pair) THEN mint.

    Both callers that mint a place_key from a region's raw fields --
    `enrich_region_analysis` (core/regions.py) and
    `RegionOption.place_key` (models/discovery.py) -- must go through this,
    never through `mint_place_key` directly on unchecked fields.
    `mint_place_key` on its own only refuses missing/invalid fields; it has
    no way to know whether `primary_locality` is even one of the region's
    own `cities`. Two mint paths with two different rules is exactly how a
    caller ends up reading the unchecked answer (MYS-460 review).
    """
    if not locality_matches_cities(primary_locality, cities, country_code):
        return None
    return mint_place_key(country_code, primary_locality)


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

    locality = slug(primary_locality)
    if not locality:
        return None

    return f"{code.lower()}:{locality}"
