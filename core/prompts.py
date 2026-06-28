"""
Prompt builders for workflow phases.

Extracted from api/streaming.py to eliminate duplication across
delivery mechanisms (API, CLI, Streamlit).
"""

import json
from typing import List


def build_discovery_prompt(
    book_title: str, author: str, vibe: str | None = None
) -> str:
    """Build the user prompt for the discovery phase (phase 2).

    When ``vibe`` is provided it is appended as an explicit atmosphere
    preference that re-weights *already-valid* candidates toward that mood and
    asks the connection note to name it. It deliberately does NOT relax the
    grounding requirement: places must still be genuinely connected to the book,
    so the vibe can only tilt selection among honest candidates, never invent a
    link. When ``vibe`` is None the returned prompt is byte-identical to before.
    """
    prompt = (
        f'Discover travel locations for "{book_title}" by {author}.\n\n'
        f"Find cities, landmarks, and author-related sites, "
        f"then group them into practical travel regions."
    )
    if vibe:
        prompt += (
            f"\n\nThe reader has chosen a \"{vibe}\" mood. Among places that are "
            f"genuinely and verifiably connected to the book, prefer those whose "
            f"atmosphere best fits a {vibe} feeling, and name that {vibe} "
            f"atmosphere in each place's connection note. Do NOT include or "
            f"invent any place that is not truly tied to the book just to match "
            f"the mood — factual grounding always wins over vibe."
        )
    return prompt


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


def build_local_atmosphere_prompt(
    book_title: str,
    author: str,
    location_label: str,
    radius_km: int,
) -> str:
    """Build the user prompt for the local-atmosphere flow.

    The detailed instructions live on the agent (see
    ``local_atmosphere_researcher`` / ``local_atmosphere_formatter`` in
    ``agents/prompts/v2.json``); this prompt just frames the task and pins
    the user's location and radius into the conversation history.
    """
    return (
        f'The reader cannot travel to the actual setting of "{book_title}" by '
        f'{author}. Build an atmospheric local outing instead.\n\n'
        f"User location: {location_label}\n"
        f"Search radius: ~{radius_km} km from that location (≈ 1 hour drive).\n\n"
        f"Find real places near the user whose mood, era, and sensory feel "
        f"evoke the book. Group them into 1-3 nearby towns and return a "
        f"TripItinerary that respects the user's preferences."
    )
