"""
Local atmosphere agent.

Two-stage pipeline that finds real places near the user's current location
whose mood and sensory character evoke a chosen book — used when the reader
cannot travel to the book's actual setting.
"""

from google.adk.agents import LlmAgent, SequentialAgent

from models.itinerary import ComposerEnvelope
from agents.prompts import AgentPrompts, load_prompts


def create_local_atmosphere_pipeline(
    model,
    google_search_tool,
    location_label: str,
    radius_km: int,
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
        SequentialAgent that researches and formats a local-atmosphere itinerary.
    """
    if prompts is None:
        prompts = load_prompts()

    researcher_instruction = prompts.local_atmosphere_researcher.format(
        location_label=location_label, radius_km=radius_km
    )
    formatter_instruction = prompts.local_atmosphere_formatter.format(
        location_label=location_label, radius_km=radius_km
    )

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

    return SequentialAgent(
        name="local_atmosphere_pipeline",
        sub_agents=[researcher, formatter],
    )
