"""
Reader profile agent.

Reads user preferences from session state to personalize travel itineraries.
"""

from google.adk.agents import LlmAgent
from tools.preferences import get_preferences_tool


def create_reader_profile_agent(model):
    """
    Create the reader profile agent.

    This agent reads user preferences from session state (user:preferences)
    and provides personalization context for the trip composer.

    Args:
        model: The LLM model to use

    Returns:
        LlmAgent that reads preferences and provides personalization context
    """
    return LlmAgent(
        name="reader_profile_agent",
        model=model,
        output_key="reader_profile",
        tools=[get_preferences_tool],
        instruction="""You are a reader profile specialist focused on personalization.

FIRST, call the get_user_preferences tool to retrieve user preferences from session state.

The preferences may include:
- budget: "budget", "moderate", or "luxury"
- preferred_pace: "relaxed", "moderate", or "fast-paced"
- prefers_museums: true/false
- travels_with_kids: true/false
- favorite_genres: list of genres

Based on the tool response, provide a summary for the trip composer:

1. If preferences exist (found=true), summarize them clearly:
   "User prefers [budget] budget, [pace] pace, [does/doesn't] like museums..."

2. If no preferences found (found=false), state:
   "No user preferences found. Using defaults: moderate budget, balanced pace, museum-friendly."

Keep your response concise - just summarize the preferences for the trip composer to use.""",
    )
