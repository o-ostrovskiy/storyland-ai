"""
Trip composer agent.

Synthesizes all discovery results into a complete, cohesive travel itinerary.
"""

from google.adk.agents import LlmAgent
from models.itinerary import TripItinerary


def create_trip_composer_agent(model):
    """
    Create the trip composer agent.

    Args:
        model: The LLM model to use

    Returns:
        LlmAgent that composes the final travel itinerary
    """
    return LlmAgent(
        name="trip_composer",
        model=model,
        output_schema=TripItinerary,
        output_key="final_itinerary",
        instruction="""You are a literary travel planner. Create a complete travel itinerary.

Review ALL the information from the conversation history:
- Book metadata and context
- Cities to visit
- Landmarks and experiences
- Author-related sites

Create a TripItinerary with:

1. GROUP by city: Organize all stops by the city they're in
2. For EACH city, create a CityPlan with:
   - name: City name
   - country: Country name
   - days_suggested: 1-3 days (based on number of stops)
   - overview: 2-3 sentences about what to expect in this city
   - stops: 3-7 places to visit (combine landmarks and author sites)

3. For EACH stop, create a CityStop with:
   - name: Exact name of the place
   - type: "museum", "landmark", "cafe", "bookstore", "monument", etc.
   - reason: 1-2 sentences explaining WHY this matters for the book
   - time_of_day: "morning", "afternoon", "evening", or "full_day"
   - notes: Practical tip (hours, how to get there, what to look for)

4. Write a summary_text: 3-4 sentences capturing the essence of the entire journey

Make it inspiring and actionable. Readers should know exactly what to do.""",
    )
