"""
Region validation logic.

Extracted from api/streaming.py lines 476-501 to eliminate duplication
between API and CLI delivery mechanisms.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from models.place_key import mint_place_key


def get_valid_region_ids(regions: List[dict]) -> Set[int]:
    """Extract valid integer region_ids from a region list.

    Filters out None, missing, and non-int region_ids (guards against
    malformed agent output).
    """
    return {
        r.get("region_id")
        for r in regions
        if isinstance(r.get("region_id"), int)
    }


def validate_region_selection(
    region_ids: List[int],
    all_regions: List[dict],
) -> Tuple[List[dict], List[int]]:
    """Validate selected region_ids against available regions.

    Returns:
        Tuple of (selected_regions, invalid_ids).
        If invalid_ids is non-empty, the selection is invalid.
    """
    valid_ids = get_valid_region_ids(all_regions)
    invalid_ids = [rid for rid in region_ids if rid not in valid_ids]

    if invalid_ids:
        return [], invalid_ids

    selected = [r for r in all_regions if r.get("region_id") in region_ids]
    return selected, []


def enrich_region_analysis(region_analysis: Optional[dict]) -> dict:
    """Return a copy of ``region_analysis`` with a ``place_key`` on every region.

    The agent's output lands in ADK session state as a RAW DICT parsed from the
    model's JSON — it is never round-tripped through ``RegionOption``, so the
    model's computed ``place_key`` property never materialises at runtime. This
    is the seam where the key is actually minted, and it must be applied on BOTH
    discovery paths (fresh run and cache replay) or a region reaches the wire
    with no identity and the combined branch silently never fires.

    Pure: builds new dicts, mutates nothing (SessionStateAccessor's setters are
    silent no-ops against persisted state — MYS-172 — so we never write back).
    Idempotent: re-enriching an enriched payload is a no-op, and a region that
    already carries a key keeps it.

    A region whose grounded geo fields are missing gets ``place_key: None``. That
    is deliberate: it cannot be intersected, which costs us a combine. Deriving a
    key from ``region_name`` instead would cost us a WRONG combine — a fabricated
    shared setting — which is the failure this whole capability must not have.
    """
    source: Dict[str, Any] = dict(region_analysis or {})
    regions = source.get("regions")
    if not isinstance(regions, list):
        return source

    enriched: List[Any] = []
    for region in regions:
        if not isinstance(region, dict):
            enriched.append(region)
            continue
        out = dict(region)
        # ALWAYS overwrite — never trust an incoming place_key. ADK writes the
        # model's raw parsed JSON dict into session state (the premise this
        # whole seam exists on), so a model that emits an unasked-for
        # place_key anyway — or a cached/legacy entry carrying a bad one —
        # would otherwise ride straight through untouched. Minting is
        # deterministic from the grounded fields, so overwriting costs
        # nothing: idempotency holds (re-enriching yields the same key) and a
        # HOSTILE key (e.g. "fr:paris" stapled onto a US/Paris region) is
        # replaced with the one actually derived from the grounded fields.
        out["place_key"] = mint_place_key(
            out.get("country_code"), out.get("primary_locality")
        )
        enriched.append(out)

    source["regions"] = enriched
    return source
