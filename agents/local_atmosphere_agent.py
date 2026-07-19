"""
Local atmosphere agent.

Two-stage pipeline that finds real places near the user's current location
whose mood and sensory character evoke a chosen book — used when the reader
cannot travel to the book's actual setting.
"""

from google.adk.agents import LlmAgent

from models.itinerary import ComposerEnvelope
from agents.prompts import AgentPrompts, load_prompts, preferences_block


def create_local_atmosphere_agents(
    model,
    google_search_tool,
    location_label: str,
    radius_km: int,
    preferences: dict | None = None,
    prompts: AgentPrompts | None = None,
):
    """
    Create the local-atmosphere pipeline.

    Researcher uses google_search to find atmospheric places within ~radius_km
    of the user; formatter shapes the result into a TripItinerary.

    Args:
        model: The LLM model to use.
        google_search_tool: The Google Search tool.
        location_label: Human-readable location string (e.g. "New York, NY 10013").
        radius_km: Maximum distance in km from the user's location.
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        (researcher, formatter) LlmAgent pair that researches and formats a local-atmosphere itinerary.
    """
    if prompts is None:
        prompts = load_prompts()

    # Graph-scoped context (ADR #24): the initial user prompt reaches only
    # the first node (book_context_researcher) — these two agents sit 3-4
    # nodes downstream, so preferences must be baked into their instructions.
    researcher_instruction = prompts.local_atmosphere_researcher.format(
        location_label=location_label, radius_km=radius_km
    ) + preferences_block(preferences)
    formatter_instruction = prompts.local_atmosphere_formatter.format(
        location_label=location_label, radius_km=radius_km
    ) + preferences_block(preferences)

    researcher = LlmAgent(
        name="local_atmosphere_researcher",
        model=model,
        tools=[google_search_tool],
        instruction=researcher_instruction,
    )

    formatter = LlmAgent(
        name="local_atmosphere_formatter",
        model=model,
        output_schema=ComposerEnvelope,
        output_key="composer_envelope",
        instruction=formatter_instruction,
    )

    return researcher, formatter
