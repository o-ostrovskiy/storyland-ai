"""
Trip composer agent.

Synthesizes all discovery results into a complete, cohesive travel itinerary.
"""

from google.adk.agents import LlmAgent
from models.itinerary import ComposerEnvelope
from agents.prompts import AgentPrompts, load_prompts


def create_trip_composer_agent(model, prompts: AgentPrompts | None = None):
    """
    Create the trip composer agent.

    Args:
        model: The LLM model to use
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        LlmAgent that composes the final travel itinerary wrapped in a ComposerEnvelope
    """
    if prompts is None:
        prompts = load_prompts()
    return LlmAgent(
        name="trip_composer",
        model=model,
        output_schema=ComposerEnvelope,
        output_key="composer_envelope",
        instruction=prompts.trip_composer,
    )
