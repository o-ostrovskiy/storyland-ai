"""
Prompt builders for workflow phases.

Extracted from api/streaming.py to eliminate duplication across
delivery mechanisms (API, CLI, Streamlit).
"""

import json
from typing import List


def build_discovery_prompt(book_title: str, author: str) -> str:
    """Build the user prompt for the discovery phase (phase 2)."""
    return (
        f'Discover travel locations for "{book_title}" by {author}.\n\n'
        f"Find cities, landmarks, and author-related sites, "
        f"then group them into practical travel regions."
    )


def build_composition_prompt(
    book_title: str,
    author: str,
    selected_regions: List[dict],
) -> str:
    """Build the user prompt for the composition phase (phase 3)."""
    return (
        f'Create a travel itinerary for "{book_title}" by {author}.\n\n'
        f"Use ONLY the cities from the selected region(s): "
        f"{json.dumps(selected_regions)}\n\n"
        f"Create a personalized itinerary based on user preferences "
        f"and the selected region(s).\n"
        f"Include ALL cities from the selected regions in your itinerary."
    )
