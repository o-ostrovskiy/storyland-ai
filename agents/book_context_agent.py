"""
Book context research agent.

Two-stage pipeline that researches the book's setting, time period, and themes
using Google Search.
"""

from google.adk.agents import LlmAgent, SequentialAgent
from models.book import BookContext


def create_book_context_pipeline(model, google_search_tool):
    """
    Create the book context research pipeline.

    Args:
        model: The LLM model to use
        google_search_tool: The Google Search tool

    Returns:
        SequentialAgent that researches and formats book context
    """
    book_context_researcher = LlmAgent(
        name="book_context_researcher",
        model=model,
        tools=[google_search_tool],
        instruction="""You are a literary research specialist. Research the book's setting and themes.

The book information is in the conversation history above.

Use google_search to find:
1. PRIMARY LOCATIONS: Where does the story take place? (cities, countries, regions)
2. TIME PERIOD: When does it take place? (specific year, era, or historical period)
3. MAIN THEMES: What are the central themes? (war, love, survival, etc.)

Search queries to try:
- "[book title] setting location"
- "[book title] time period historical context"
- "[book title] themes analysis"

Provide detailed findings with specific locations and time periods.""",
    )

    book_context_formatter = LlmAgent(
        name="book_context_formatter",
        model=model,
        output_schema=BookContext,
        output_key="book_context",
        instruction="""Format the research into a validated BookContext object.

Extract:
- primary_locations: List of specific locations (cities, countries, regions)
- time_period: The historical era or specific timeframe
- themes: List of main themes from the book

Be specific and factual. Use proper names for locations.""",
    )

    return SequentialAgent(
        name="book_context_pipeline",
        sub_agents=[book_context_researcher, book_context_formatter],
    )
