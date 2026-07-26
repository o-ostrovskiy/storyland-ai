"""
Result extraction and validation.

Handles extracting structured data from agent responses, with fallback
to JSON text parsing when output_key doesn't work.

Extracted from api/streaming.py lines 89-104 and 564-598.
"""

import json
import re
from typing import Optional, Tuple

from pydantic import ValidationError

from common.logging import get_logger
from core.guardrails import (
    sanitize_itinerary_explanations,
    sanitize_expansion_explanations,
    sanitize_book_recommendations,
)
from models.book import BookRecommendationsResult
from models.itinerary import TripItinerary, ComposerEnvelope, ExpansionResult
from models.place_key import resolve_country_name, slug

logger = get_logger("storyland.core.extraction")


def validate_trip_itinerary(value: object) -> Optional[dict]:
    """Validate an itinerary payload against TripItinerary schema.

    Returns validated dict or None if invalid.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if not isinstance(value, dict):
        return None

    try:
        validated = TripItinerary.model_validate(value)
        return validated.model_dump()
    except ValidationError:
        return None


def validate_composer_envelope(value: object) -> Optional[Tuple[dict, list]]:
    """Validate a ComposerEnvelope payload.

    Returns (itinerary_dict, suggestions_list) or None if invalid.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if not isinstance(value, dict):
        return None

    try:
        envelope = ComposerEnvelope.model_validate(value)
        return envelope.itinerary.model_dump(), [s.model_dump() for s in envelope.suggestions]
    except ValidationError:
        return None


def validate_expansion_result(value: object) -> Optional[dict]:
    """Validate an ExpansionResult payload.

    Returns validated dict or None if invalid.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if not isinstance(value, dict):
        return None

    try:
        validated = ExpansionResult.model_validate(value)
        return validated.model_dump()
    except ValidationError:
        return None


def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract first JSON object from text (finds outermost braces)."""
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        try:
            return json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return None
    return None


# Match types that assert a grounded, verifiable book<->place connection and
# therefore must trace to the grounded discovery research. The other two
# ("thematic", "vibe") are explicitly weaker/atmospheric claims that need no
# source, so they are never touched here.
_GROUNDED_MATCH_TYPES = ("literal", "historical")

# The weakest, safest claim. An ungroundable strong claim is downgraded to this
# (never dropped, never upgraded), mirroring the schema default and the org
# "choose the weaker claim when unsure" guardrail.
_DOWNGRADE_TARGET = "vibe"


def downgrade_ungrounded_match_types(
    itinerary_dict: Optional[dict], grounding_text: str
) -> Optional[dict]:
    """Downgrade literal/historical stops that don't trace to grounded research.

    The trust core of the hallucination guardrail: the formatter self-labels each
    stop's ``match_type``, but a self-label is not evidence. Here we *derive* the
    label server-side from a real check — any stop the formatter marked
    ``literal``/``historical`` whose name is not grounded in the discovery
    research (city/landmark/author/context; see ``is_title_grounded`` for the
    token-overlap matching rule) is downgraded to the
    weakest claim (``vibe``) and has its ``grounding_source`` cleared, since the
    cited evidence didn't hold up. Stops are never dropped and never upgraded.

    Conservative and fail-open by design (it must never make results worse):
      * No grounding text captured -> return unchanged (we cannot prove anything
        is ungrounded, e.g. the local-atmosphere path has no discovery research).
      * ``thematic``/``vibe`` stops are left untouched (no source required).
    """
    if not itinerary_dict:
        return itinerary_dict

    haystack = grounding_token_set(grounding_text)
    if not haystack:
        return itinerary_dict

    downgraded = 0
    for city in itinerary_dict.get("cities") or []:
        for stop in city.get("stops") or []:
            if stop.get("match_type") not in _GROUNDED_MATCH_TYPES:
                continue
            if is_title_grounded(stop.get("name"), haystack):
                continue
            stop["match_type"] = _DOWNGRADE_TARGET
            stop["grounding_source"] = None
            downgraded += 1

    if downgraded:
        logger.info("itinerary_match_type_downgraded", downgraded=downgraded)
    return itinerary_dict


# MYS-660: a stop's own `address` is composer-written free text with no
# guaranteed shape -- "<place>, <neighbourhood>, <city>, <country>" and
# "<place>, <neighbourhood>, <city>, <state>, <country>" are BOTH observed,
# so the city is not reliably at a fixed comma position (r1 guessed
# "second-to-last when the last segment is a country", which read
# "...Salem, Massachusetts, USA" as locality "Massachusetts" -- a state, not
# a city -- and dropped a legitimate Salem stop; MYS-660 r3 / Codex P1).
#
# Rather than guess a position, every comma segment of the address is
# checked against the CityPlans actually on THIS itinerary: an address never
# invents a new city to file a stop under, so the only signal worth acting
# on is "one of my segments names a CityPlan already on this trip". No
# segment matching any known CityPlan is not evidence of a mismatch -- it's
# simply not a signal (fail open, same convention this file's other
# guardrails use).
def _address_city_candidates(address: object) -> Optional[list]:
    """Comma-separated, stripped segments of a free-text stop address.

    Returns None (never a guess) for a missing/empty address or one with a
    single fragment -- there's no separable locality to check at all.
    """
    if not isinstance(address, str):
        return None
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    return parts


def _find_reconciliation_target(
    address: object,
    cities: list,
    city_indices_by_slug: dict,
    own_index: int,
) -> Optional[int]:
    """Does this stop's address positively name a DIFFERENT CityPlan already
    on this itinerary? Returns that CityPlan's index, or None when no single
    unambiguous different-city signal exists in the address (fail open --
    never guess, and never drop on an unresolved tail).

    Scans every comma segment (see `_address_city_candidates`) rather than a
    fixed position. A segment only counts when it slug-matches a CityPlan
    name already on this trip -- an off-trip city name (e.g. a real place
    the address mentions that simply isn't part of this itinerary) yields no
    match here, same as a segment that doesn't parse as a place at all; both
    are "no signal", not "signal of a mismatch".

    Country-qualified for EVERY segment, not just ones with a same-named
    collision (MYS-660 r5 / Codex P2, same lesson as MYS-548 -- identity
    matching must not be blind to a same-named disambiguator): a real
    regression had a stop addressed "..., Paris, Texas, USA" get re-filed to
    a same-trip "Paris, France" CityPlan, because the OLD code only
    country-checked a segment when more than one CityPlan on the trip shared
    its slug -- a lone same-named CityPlan skipped the check entirely and
    matched on name alone, ignoring that the address's own country
    disagreed. There is only ONE matching path now: every segment's
    slug-matched candidates (whether there's one or several) are narrowed by
    the address's trailing country segment when one resolves; a candidate
    whose own `country` resolves and actively disagrees with the address's
    country is excluded, an unresolvable country on either side is
    tolerated (kept in, matching this module's "no signal is not evidence
    of a mismatch" convention). If narrowing empties the candidate set,
    the segment yields no match at all -- not a mismatch signal, since a
    same-named different-country city existing elsewhere doesn't mean this
    address referred to it.

    Own-city evidence always wins, across the WHOLE address, not just its
    own segment (MYS-660 r4 / Codex P1): a real regression had a stop filed
    under Buffalo with address "..., Buffalo, New York, USA" -- one segment
    ("Buffalo") positively confirms the stop is already home, but a LATER
    segment ("New York") also happens to name a different same-trip
    CityPlan. Scanning segment-by-and-adding-to-`matched_indices`
    unconditionally let that later segment outvote the earlier "already
    home" evidence and re-filed a correct stop into the wrong city. The
    fix: any segment that attests the stop is still home (`own_attested`)
    is recorded across the full scan, and wins over any different-city
    match found elsewhere in the SAME address, regardless of which segment
    came first.

    A single unified pass keeps both invariants: no segment's match is ever
    accepted (own-city OR different-city) without surviving the same
    country-qualification test, and own-city evidence anywhere still beats
    every different-city match in the same address.
    """
    parts = _address_city_candidates(address)
    if parts is None:
        return None

    address_country_code = resolve_country_name(parts[-1])
    # A trailing segment that resolved as a country is the country field,
    # not a locality candidate -- the city can be at any other position.
    candidate_segments = parts[:-1] if address_country_code is not None else parts

    matched_indices: set = set()
    own_attested = False
    for segment in candidate_segments:
        segment_slug = slug(segment)
        if not segment_slug:
            continue
        all_indices = city_indices_by_slug.get(segment_slug, ())
        if not all_indices:
            continue

        # Country-qualify every candidate this segment's slug names -- one
        # match or several, same rule either way (MYS-660 r5). An
        # unresolvable country (either side) is tolerated, not treated as a
        # mismatch, matching `locality_matches_cities`'s convention.
        if address_country_code is not None:
            qualified = [
                idx
                for idx in all_indices
                if (
                    resolved := resolve_country_name(
                        cities[idx].get("country") if isinstance(cities[idx], dict) else None
                    )
                )
                is None
                or resolved == address_country_code
            ]
        else:
            qualified = list(all_indices)

        if not qualified:
            # Every same-named candidate's own country actively disagrees
            # with the address's country -- this segment names a place
            # that isn't any CityPlan on this trip. No signal either way.
            continue

        if own_index in qualified:
            # The stop's own city hasn't been ruled out as this address's
            # match -- could still plausibly be home, so this segment is
            # positive "still home" evidence, regardless of who else also
            # qualifies.
            own_attested = True
            continue

        distinct_others = set(qualified) - {own_index}
        if len(distinct_others) == 1:
            matched_indices.add(next(iter(distinct_others)))
        # else: zero (unreachable -- qualified is non-empty and own_index
        # isn't in it) or multiple qualified candidates besides the
        # (already ruled-out) own city -- still ambiguous, this segment
        # yields no match.

    if own_attested:
        # Own-city evidence anywhere in the address beats a different-city
        # match found in another segment of the SAME address (MYS-660 r4) --
        # never re-file a stop the address also positively confirms is
        # still home.
        return None
    if len(matched_indices) == 1:
        return next(iter(matched_indices))
    # Zero, or more than one, distinct different-city match across every
    # segment -- no single unambiguous signal. Fail open.
    return None


def reconcile_stop_city_grouping(itinerary_dict: Optional[dict]) -> Optional[dict]:
    """MYS-660: never render a stop under a city its own address contradicts.

    The composer groups stops by city itself, as free-form text, with no
    coordinates and nothing that cross-checks a stop's ``address`` against
    the ``CityPlan`` it ends up filed under. Both fields are already on the
    record we hold (zero new upstream calls, zero spend) -- confirmed live:
    a real itinerary filed 3 Cartagena-addressed restaurants under
    Aracataca, ~250km away, while Cartagena was ALSO its own city on the
    same trip.

    Policy, per AC-2 (a wholesale envelope reject is explicitly the "last
    resort, not the default" -- not implemented here; see PR notes):
      * A stop whose address positively, unambiguously names a DIFFERENT
        CityPlan already on this same itinerary is RE-FILED there (the
        Colombia case: the 3 mismatched stops move to the existing
        Cartagena entry).
      * A stop whose address names no CityPlan on this itinerary at all --
        whether because it has no discernible locality, or because every
        comma segment fails to match any CityPlan here, or because a match
        is ambiguous (same-named cities, unresolvable country) -- is left
        exactly where the composer put it. "No signal" is not evidence of a
        mismatch (see ``_find_reconciliation_target``; this also closes the
        MYS-660 r3 regression where a state/administrative segment like
        "Massachusetts" used to be misread as the locality and a genuinely
        correct Salem stop was dropped for not matching anything).
      * A CityPlan left with zero stops after this pass is dropped entirely
        (MYS-268: an empty city section is worse than no section).

    Deterministic post-processing on whatever the composer emitted -- holds
    regardless of MYS-563's composition non-determinism, which is a
    *different* defect (a truncated/degenerate envelope) this validation
    layer does not attempt to fix.
    """
    if not itinerary_dict:
        return itinerary_dict

    cities = itinerary_dict.get("cities")
    if not isinstance(cities, list) or not cities:
        return itinerary_dict

    # EVERY CityPlan index sharing a given name slug (not just the first) --
    # needed so a same-named different-country city can be disambiguated
    # rather than silently shadowed by an earlier entry of the same name.
    city_indices_by_slug: dict = {}
    for i, city in enumerate(cities):
        if not isinstance(city, dict):
            continue
        name_slug = slug(city.get("name")) if isinstance(city.get("name"), str) else ""
        if name_slug:
            city_indices_by_slug.setdefault(name_slug, []).append(i)

    # A city that arrived with zero stops was never this pass's business --
    # only a city THIS PASS emptied (had >=1 stop before, none after) is
    # "worse than no section" per MYS-268. Recorded before any mutation.
    city_had_stops: dict = {}
    for i, city in enumerate(cities):
        if isinstance(city, dict):
            existing = city.get("stops")
            city_had_stops[i] = bool(isinstance(existing, list) and existing)

    refiled_into: dict = {i: [] for i in range(len(cities))}
    refiled_count = 0

    for i, city in enumerate(cities):
        if not isinstance(city, dict):
            continue
        stops = city.get("stops")
        if not isinstance(stops, list):
            continue
        kept = []
        for stop in stops:
            if not isinstance(stop, dict):
                kept.append(stop)
                continue
            target = _find_reconciliation_target(
                stop.get("address"), cities, city_indices_by_slug, own_index=i
            )
            if target is None:
                kept.append(stop)
                continue
            refiled_into[target].append(stop)
            refiled_count += 1
            logger.warning(
                "stop_city_mismatch_refiled",
                stop=stop.get("name"),
                filed_under=city.get("name"),
                refiled_to=cities[target].get("name") if isinstance(cities[target], dict) else None,
            )
        city["stops"] = kept

    for i, extra in refiled_into.items():
        if extra:
            cities[i]["stops"] = list(cities[i].get("stops") or []) + extra

    if refiled_count:
        logger.info("itinerary_stop_city_reconciled", refiled=refiled_count)

    # A city THIS PASS emptied carries nothing honest to show -- drop it
    # (MYS-268). A city that arrived with zero stops already was never
    # touched by this guard and is left exactly as the composer emitted it
    # (e.g. a legitimate zero-stop city elsewhere in the pipeline's own
    # fixtures) -- this pass only removes what it itself created.
    itinerary_dict["cities"] = [
        c
        for i, c in enumerate(cities)
        if not isinstance(c, dict) or c.get("stops") or not city_had_stops.get(i, True)
    ]
    return itinerary_dict


def _city_names(itinerary_dict: Optional[dict]) -> set:
    """The (lowercased) names of every CityPlan currently on this itinerary.

    MYS-660 r6: `_finalize` diffs this before/after `reconcile_stop_city_
    grouping` to learn which cities the guard just removed, rather than
    changing that function's return signature -- every existing caller
    (this module's own tests included) expects a single `dict` back.
    """
    if not itinerary_dict:
        return set()
    cities = itinerary_dict.get("cities")
    if not isinstance(cities, list):
        return set()
    return {
        city["name"].lower()
        for city in cities
        if isinstance(city, dict)
        and isinstance(city.get("name"), str)
        and city["name"].strip()
    }


def _drop_suggestions_naming_removed_cities(suggestions: list, removed_names: set) -> list:
    """MYS-660 r6: a suggestion chip whose `action_prompt` names a city
    `reconcile_stop_city_grouping` just removed can no longer resolve to any
    persisted city. Left alone, `executor.expand()`'s target-city scan
    (`core/executor.py`, ``city_name.lower() in action_prompt_lower``) finds
    no match on any CURRENTLY-persisted city and silently falls back to
    ``cities[0]`` -- the expansion's new places get appended, and PERSISTED,
    under whatever city happens to be first. That is the exact "stop under
    the wrong city" defect this ticket exists to kill, newly reachable
    through this guard's own side effect: before this guard existed no city
    was ever removed, so a chip's named city always existed; a city this
    pass drops (r1-r5's fix) can now orphan a suggestion that named it.

    Uses the SAME case-insensitive substring check `expand()` itself makes,
    so a chip survives this filter if and only if `expand()` would still
    resolve it to a real, persisted city -- never a stricter or looser test
    than the one that actually matters.
    """
    if not removed_names or not isinstance(suggestions, list):
        return suggestions
    kept = []
    for chip in suggestions:
        if not isinstance(chip, dict):
            kept.append(chip)
            continue
        prompt = chip.get("action_prompt")
        prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
        if prompt_lower and any(name in prompt_lower for name in removed_names):
            logger.warning("suggestion_dropped_removed_city", action_prompt=prompt)
            continue
        kept.append(chip)
    return kept


def extract_itinerary_from_response(
    final_response, state_accessor
) -> Optional[Tuple[dict, list]]:
    """Two-phase extraction: session state first, then text fallback.

    Returns (itinerary_dict, suggestions_list). Suggestions may be empty for
    legacy responses that predate ComposerEnvelope.

    Args:
        final_response: The final ADK event from runner.run_async()
        state_accessor: SessionStateAccessor wrapping session.state

    Returns:
        (itinerary_dict, suggestions_list) tuple or None
    """
    grounding_text = state_accessor.grounding_research_text

    def _finalize(itinerary_dict, suggestions):
        itinerary_dict = downgrade_ungrounded_match_types(itinerary_dict, grounding_text)
        names_before = _city_names(itinerary_dict)
        itinerary_dict = reconcile_stop_city_grouping(itinerary_dict)
        # MYS-660 r6: a city the guard above just removed can orphan a
        # suggestion chip that named it -- see _drop_suggestions_naming_
        # removed_cities's own doc comment for why that matters.
        removed_names = names_before - _city_names(itinerary_dict)
        if removed_names:
            suggestions = _drop_suggestions_naming_removed_cities(suggestions, removed_names)
        itinerary_dict = sanitize_itinerary_explanations(itinerary_dict)
        return itinerary_dict, suggestions

    # Primary: composer_envelope from session state (set by output_key="composer_envelope")
    envelope_data = state_accessor.composer_envelope
    if envelope_data is not None:
        result = validate_composer_envelope(envelope_data)
        if result is not None:
            logger.info("itinerary_from_envelope")
            return _finalize(*result)

    # Legacy fallback: bare TripItinerary from state (set by output_key="final_itinerary")
    state_itinerary = state_accessor.final_itinerary
    itinerary_result = validate_trip_itinerary(state_itinerary)
    if itinerary_result is not None:
        logger.info("itinerary_from_state")
        return _finalize(itinerary_result, [])

    # Text fallback: parse from final response text
    if (
        final_response
        and final_response.content
        and final_response.content.parts
    ):
        for part in final_response.content.parts:
            if hasattr(part, "text") and part.text:
                candidate = extract_json_from_text(part.text)
                if candidate is not None:
                    # Try envelope first
                    env_result = validate_composer_envelope(candidate)
                    if env_result is not None:
                        logger.info("itinerary_from_text_envelope_fallback")
                        return _finalize(*env_result)
                    # Then bare itinerary
                    itinerary_result = validate_trip_itinerary(candidate)
                    if itinerary_result is not None:
                        logger.info("itinerary_from_text_fallback")
                        return _finalize(itinerary_result, [])

    return None


def extract_expansion_from_state(state_accessor) -> Optional[dict]:
    """Extract and validate the last expansion result from session state."""
    return sanitize_expansion_explanations(
        validate_expansion_result(state_accessor.last_expansion)
    )


def validate_book_recommendations_result(value: object) -> Optional[dict]:
    """Validate a BookRecommendationsResult payload.

    Returns validated dict or None if invalid.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if not isinstance(value, dict):
        return None

    try:
        validated = BookRecommendationsResult.model_validate(value)
        return validated.model_dump()
    except ValidationError:
        return None


def extract_book_recommendations_from_state(state_accessor) -> Optional[dict]:
    """Extract and validate the last book recommendations result from session state."""
    return sanitize_book_recommendations(
        validate_book_recommendations_result(state_accessor.last_book_recommendations)
    )


# Leading/interior articles carry no grounding signal and are the usual source
# of surface-form mismatches ("Pump Room" vs "The Grand Pump Room").
_ARTICLES = frozenset({"the", "a", "an"})

# 1-2 token titles require every token in the grounding text; 3+ token titles
# tolerate exactly this many missing tokens, which absorbs surface variants
# like "The Grand Pump Room" vs "Pump Room". A flat overlap ratio is
# deliberately avoided: it would let long titles pass with several unsupported
# tokens, and long place names are where scattered-word false matches bite.
_GROUNDING_MAX_MISSING = 1


def grounding_token_set(text: object) -> frozenset:
    """Tokenize grounding text once for repeated is_title_grounded() checks.

    Word tokens (unicode-aware, lowercased, punctuation stripped). An empty
    result means "no usable evidence" — every caller treats that as its
    fail-open branch, same as the old empty-string check.
    """
    if not isinstance(text, str):
        return frozenset()
    return frozenset(re.findall(r"\w+", text.lower()))


def is_title_grounded(title: object, haystack_tokens: frozenset) -> bool:
    """Shared grounding-match primitive for the output-side guards.

    Token containment replacing naive substring matching, which failed in
    both directions: "The Mill" false-matched any text containing the
    substring (e.g. "the millionaire"), while a grounded "The Grand Pump
    Room" was missed when the research said "Pump Room". Matching on whole
    word tokens fixes the former; tolerating at most one missing token on
    3+ token titles fixes the latter. The allowance is a fixed count, not a
    ratio, so a long title cannot pass on scattered partial support.

    A title with no significant tokens (empty/None/articles-only) is never
    grounded — identical to the old empty-normalized-title behavior.
    """
    title_tokens = grounding_token_set(title) - _ARTICLES
    if not title_tokens:
        return False
    missing = len(title_tokens - haystack_tokens)
    allowed_missing = _GROUNDING_MAX_MISSING if len(title_tokens) >= 3 else 0
    return missing <= allowed_missing


def filter_grounded_recommendations(
    rec_data: Optional[dict], researcher_text: str
) -> Optional[dict]:
    """Drop recommendations whose title is not grounded in the researcher's text.

    The book-recommendation formatter is tool-less and instructed never to
    invent books, but as a defensive post-validation we drop any recommendation
    whose title is not grounded in the researcher candidate text (per the
    ``is_title_grounded`` token-overlap rule). This
    is the output-side complement to relaxing the schema floor: relaxing the
    floor removes the *pressure* to fabricate; this removes any title that was
    fabricated anyway.

    Conservative and fail-open by design (it must never make results worse):
      * No researcher text captured -> return ``rec_data`` unchanged (we cannot
        prove anything is ungrounded, so we never drop on missing evidence).
      * Filtering that would drop *every* recommendation -> return ``rec_data``
        unchanged (treat as a capture/formatting mismatch, never surface empty).
      * Otherwise return the grounded subset with ``limited_matches=True`` so the
        caller can signal an honest "fewer real matches" state instead of padding.
    """
    if rec_data is None:
        return None

    recs = rec_data.get("recommendations") or []
    haystack = grounding_token_set(researcher_text)
    if not haystack:
        return rec_data

    grounded = [rec for rec in recs if is_title_grounded(rec.get("title"), haystack)]

    if not grounded or len(grounded) == len(recs):
        return rec_data

    result = dict(rec_data)
    result["recommendations"] = grounded
    result["limited_matches"] = True
    logger.info(
        "book_recommendations_grounding_filtered",
        kept=len(grounded),
        dropped=len(recs) - len(grounded),
    )
    return result
