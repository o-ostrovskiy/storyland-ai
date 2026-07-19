"""
Prompt builders for workflow phases.

Extracted from api/streaming.py to eliminate duplication across
delivery mechanisms (API, CLI, Streamlit).
"""

import json
from typing import List


def build_discovery_prompt(
    book_title: str,
    author: str,
    vibe: str | None = None,
    taste_context: dict | None = None,
) -> str:
    """Build the user prompt for the discovery phase (phase 2).

    When ``vibe`` is provided it is appended as an explicit atmosphere
    preference that re-weights *already-valid* candidates toward that mood and
    asks the connection note to name it. It deliberately does NOT relax the
    grounding requirement: places must still be genuinely connected to the book,
    so the vibe can only tilt selection among honest candidates, never invent a
    link. When ``vibe`` is None the returned prompt is byte-identical to before.

    When ``taste_context`` (an optional, already-bounded
    ``{"titles": [...], "moods": [...]}`` block derived from a reader's
    imported reading history) is provided, it is appended as a second,
    independent bias clause that prefers grounded places whose atmosphere
    resonates with that demonstrated taste and lets the connection note
    name the resonance. Like ``vibe`` it NEVER relaxes grounding. ``vibe``
    and ``taste_context`` compose independently; absent/empty
    ``taste_context`` leaves the prompt byte-identical to a no-taste
    request.
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
    if taste_context:
        titles = [t for t in (taste_context.get("titles") or []) if t]
        moods = [m for m in (taste_context.get("moods") or []) if m]
        descriptors = []
        if titles:
            descriptors.append("books such as " + ", ".join(titles))
        if moods:
            descriptors.append("moods like " + ", ".join(moods))
        if descriptors:
            taste_desc = "; ".join(descriptors)
            prompt += (
                f"\n\nThe reader's own reading history leans toward {taste_desc}. "
                f"Among places that are genuinely and verifiably connected to the "
                f"book, prefer those whose atmosphere resonates with that taste, and "
                f"you may name that resonance in the place's connection note. Do NOT "
                f"include or invent any place that is not truly tied to the book just "
                f"to match the reader's taste — factual grounding always wins over "
                f"taste."
            )
    return prompt


def build_composition_prompt(
    book_title: str,
    author: str,
    selected_regions: List[dict],
    book_context: dict | None = None,
    city_discovery: object | None = None,
    landmark_discovery: object | None = None,
    author_sites: object | None = None,
    preferences: dict | None = None,
) -> str:
    """Build the user prompt for the composition phase (phase 3).

    Under the ADK 2 graph runtime an invocation's conversation is scoped to
    its trigger chain — the composer (a separate invocation from discovery)
    sees NOTHING of the discovery conversation. On the 1.x template runtime
    it implicitly saw all of it. The grounded research the composer needs is
    therefore passed EXPLICITLY here, read from session state (where the
    discovery formatters put it via output_key): book context for
    theme/setting fit, and the three discovery payloads so stops come from
    researched places instead of the model's unaided world knowledge.
    Explicit-and-bounded beats the old implicit-and-unbounded history: the
    composer gets exactly the validated payloads, not every intermediate
    researcher turn.
    """
    sections = [
        f'Create a travel itinerary for "{book_title}" by {author}.\n',
        (
            "Use ONLY the cities from the selected region(s): "
            f"{json.dumps(selected_regions)}\n"
        ),
    ]
    if book_context:
        sections.append(
            "Book context (setting, time period, themes) from research:\n"
            f"{json.dumps(book_context)}\n"
        )
    grounded = {
        "cities": city_discovery,
        "landmarks": landmark_discovery,
        "author_sites": author_sites,
    }
    grounded = {k: v for k, v in grounded.items() if v}
    if preferences:
        # user:preferences reach the composer explicitly (MYS-436 removed the
        # reader_profile agent that used to surface them as a conversation
        # turn; the API — and the eval harness — still supply them).
        sections.append(
            "READER PREFERENCES — honor these when choosing stops, pacing, "
            "and trip length:\n"
            f"{json.dumps(preferences)}\n"
        )
    if grounded:
        sections.append(
            "Grounded discovery research — prefer these real, researched "
            "places (within the selected regions) when composing stops, and "
            "label match_type/grounding_source from this evidence:\n"
            f"{json.dumps(grounded)}\n"
        )
    sections.append(
        "Create a personalized itinerary based on user preferences "
        "and the selected region(s).\n"
        "Include ALL cities from the selected regions in your itinerary."
    )
    return "\n".join(sections)


def build_local_atmosphere_prompt(
    book_title: str,
    author: str,
    location_label: str,
    radius_km: int,
    preferences: dict | None = None,
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
        + (
            "\n\nREADER PREFERENCES — honor these when choosing stops and "
            "pacing:\n" + json.dumps(preferences)
            if preferences
            else ""
        )
    )
