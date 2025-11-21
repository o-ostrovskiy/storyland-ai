"""
Reader profile agent.

Uses memory service to query past user preferences and interactions to
personalize travel itineraries.
"""

from google.adk.agents import LlmAgent
from google.adk.tools import load_memory_tool


def create_reader_profile_agent(model, memory_service):
    """
    Create the reader profile agent.

    This agent queries the memory service to retrieve:
    - User's past travel preferences
    - Previously explored books
    - Feedback on past itineraries
    - Favorite genres and authors

    Args:
        model: The LLM model to use
        memory_service: The memory service instance (InMemoryMemoryService or custom)

    Returns:
        LlmAgent that queries memory and provides personalized recommendations
    """
    return LlmAgent(
        name="reader_profile_agent",
        model=model,
        tools=[load_memory_tool(memory_service)],
        instruction="""You are a reader profile specialist focused on personalization.

Use the load_memory tool to recall the user's history:
1. Previous travel preferences (museums, budget, pace, etc.)
2. Books they've explored before
3. Feedback on past itineraries
4. Favorite genres and authors

Based on the memory results, provide personalized recommendations for the current itinerary:
- Should we focus more on museums or outdoor experiences?
- What budget level should we target?
- Does the user prefer a relaxed or fast-paced itinerary?
- Are there any dietary restrictions or accessibility needs?

If this is a new user with no history, respond with a message indicating no preferences found
and suggest default settings (moderate budget, balanced pace, museum-friendly).

Store your findings in a clear, structured format that the trip composer can use.""",
    )
