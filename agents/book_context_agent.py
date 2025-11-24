"""
Book context research agent.

Two-stage pipeline that researches the book's setting, time period, and themes
using Google Search.
"""

from google.adk.agents import LlmAgent, SequentialAgent
from models.book import BookContext


def create_book_context_pipeline(model, google_search_tool, book_title: str, author: str):
    """
    Create the book context research pipeline.

    Args:
        model: The LLM model to use
        google_search_tool: The Google Search tool
        book_title: The exact book title from book_metadata
        author: The exact author name from book_metadata

    Returns:
        SequentialAgent that researches and formats book context
    """
    book_context_researcher = LlmAgent(
        name="book_context_researcher",
        model=model,
        tools=[google_search_tool],
        instruction=f"""You are a literary research specialist. Research the book's setting and themes.

BOOK: "{book_title}" by {author}

Use google_search to find:
1. PRIMARY LOCATIONS: Where does the story take place? (cities, countries, regions)
2. TIME PERIOD: When does it take place? (specific year, era, or historical period)
3. MAIN THEMES: What are the central themes? (war, love, survival, etc.)

Search queries to use:
- "{book_title} {author} setting location"
- "{book_title} {author} time period historical context"
- "{book_title} {author} themes analysis"

Provide detailed findings with specific locations and time periods.""",
    )

    book_context_formatter = LlmAgent(
        name="book_context_formatter",
        model=model,
        output_schema=BookContext,
        output_key="book_context",
        instruction="""Format the research into a validated BookContext object.

IMPORTANT: If the research found no context information, return empty lists/null fields - do not hallucinate.

Extract:
- primary_locations: List of specific locations (cities, countries, regions)
- time_period: The historical era or specific timeframe
- themes: List of main themes from the book

Be specific and factual. Use proper names for locations. Include only information actually found in the research.""",
    )

    return SequentialAgent(
        name="book_context_pipeline",
        sub_agents=[book_context_researcher, book_context_formatter],
    )
