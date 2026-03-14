"""
Region validation logic.

Extracted from api/streaming.py lines 476-501 to eliminate duplication
between API and CLI delivery mechanisms.
"""

from typing import List, Set, Tuple


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
