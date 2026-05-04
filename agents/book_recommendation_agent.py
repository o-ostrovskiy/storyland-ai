"""
Book recommendation agent.

Single-agent pipeline that recommends 5 books based on the current book,
destination cities, and themes. Mirrors the expansion_agent pattern but uses
one LlmAgent (books are well-known entities; two-stage pipeline not needed).
"""

from google.adk.agents import LlmAgent

from models.book import BookRecommendationsResult
from agents.prompts import AgentPrompts, load_prompts


def create_book_recommendation_agent(
    model,
    google_search_tool,
    book_title: str,
    author: str,
    destinations: str,
    themes: str,
    prompts: AgentPrompts | None = None,
) -> LlmAgent:
    """Create the book recommendation agent.

    Uses google_search to find 5 real books based on the source book + destinations
    + themes, balanced across destination/themes/author recommendation bases.

    Args:
        model: The LLM model to use.
        google_search_tool: The Google Search tool.
        book_title: Exact book title.
        author: Exact author name.
        destinations: Comma-separated city names from the user's itinerary.
        themes: Comma-separated themes/mood descriptors from the book context.
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        LlmAgent that researches and formats book recommendations.
    """
    if prompts is None:
        prompts = load_prompts()

    instruction = prompts.book_recommendation.format(
        book_title=book_title,
        author=author,
        destinations=destinations,
        themes=themes,
    )

    return LlmAgent(
        name="book_recommendation_agent",
        model=model,
        tools=[google_search_tool],
        output_schema=BookRecommendationsResult,
        output_key="last_book_recommendations",
        instruction=instruction,
    )
