"""
Reader profile agent.

Reads user preferences from session state to personalize travel itineraries.
"""

from google.adk.agents import LlmAgent
from tools.preferences import get_preferences_tool
from agents.prompts import AgentPrompts, load_prompts


def create_reader_profile_agent(model, prompts: AgentPrompts | None = None):
    """
    Create the reader profile agent.

    This agent reads user preferences from session state (user:preferences)
    and provides personalization context for the trip composer.

    Args:
        model: The LLM model to use
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        LlmAgent that reads preferences and provides personalization context
    """
    if prompts is None:
        prompts = load_prompts()
    return LlmAgent(
        name="reader_profile_agent",
        model=model,
        output_key="reader_profile",
        tools=[get_preferences_tool],
        instruction=prompts.reader_profile,
    )
