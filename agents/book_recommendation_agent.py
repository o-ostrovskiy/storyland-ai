"""
Book recommendation agent.

Two-stage pipeline that recommends 5 books based on the current book,
destination cities, and themes. Mirrors the expansion_agent pattern:
researcher uses google_search, formatter emits structured output only.

ADK forbids combining `tools` with `output_schema` on a single LlmAgent —
the model can either reply with structured output OR call tools, not both.
The split below lets us keep both: researcher does the searches, formatter
shapes the results into BookRecommendationsResult.
"""

from google.adk.agents import LlmAgent

from models.book import BookRecommendationsResult
from agents.prompts import AgentPrompts, load_prompts


def create_book_recommendation_agents(
    model,
    google_search_tool,
    book_title: str,
    author: str,
    destinations: str,
    themes: str,
    prompts: AgentPrompts | None = None,
):
    """Create the book recommendation pipeline.

    Researcher uses google_search to find candidate books matching destination,
    themes, and author bases; formatter shapes the result into a
    BookRecommendationsResult with exactly 5 BookRecommendation entries.

    Args:
        model: The LLM model to use.
        google_search_tool: The Google Search tool.
        book_title: Exact book title.
        author: Exact author name.
        destinations: Comma-separated city names from the user's itinerary.
        themes: Comma-separated themes/mood descriptors from the book context.
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        (researcher, formatter) LlmAgent pair that researches and formats book recommendations.
    """
    if prompts is None:
        prompts = load_prompts()

    researcher_instruction = prompts.book_recommendation_researcher.format(
        book_title=book_title,
        author=author,
        destinations=destinations,
        themes=themes,
    )
    formatter_instruction = prompts.book_recommendation_formatter.format(
        book_title=book_title,
        author=author,
        destinations=destinations,
        themes=themes,
    )

    researcher = LlmAgent(
        name="book_recommendation_researcher",
        model=model,
        tools=[google_search_tool],
        instruction=researcher_instruction,
    )

    formatter = LlmAgent(
        name="book_recommendation_formatter",
        model=model,
        output_schema=BookRecommendationsResult,
        output_key="last_book_recommendations",
        instruction=formatter_instruction,
    )

    return researcher, formatter
