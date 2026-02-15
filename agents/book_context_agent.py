"""
Book context research agent.

Two-stage pipeline that researches the book's setting, time period, and themes
using Google Search.
"""

from google.adk.agents import LlmAgent, SequentialAgent
from models.book import BookContext


def create_book_context_pipeline(model, google_search_tool, book_title: str = "", author: str = ""):
    """
    Create the book context research pipeline.

    Args:
        model: The LLM model to use
        google_search_tool: The Google Search tool
        book_title: The exact book title from book_metadata (optional for eval workflow)
        author: The exact author name from book_metadata (optional for eval workflow)

    Returns:
        SequentialAgent that researches and formats book context
    """
    # When title is known at creation time, bake it in.
    # Otherwise (eval workflow), instruct the agent to read from prior context.
    normalized_title = (book_title or "").strip()
    normalized_author = (author or "").strip()
    if normalized_title:
        if normalized_author:
            book_ref = f'BOOK: "{normalized_title}" by {normalized_author}'
            title_author_query = f"{normalized_title} {normalized_author}"
        else:
            book_ref = f'BOOK: "{normalized_title}"'
            title_author_query = normalized_title
        search_hint = (
            f'Search queries to use:\n'
            f'- "{title_author_query} setting location"\n'
            f'- "{title_author_query} time period historical context"\n'
            f'- "{title_author_query} themes analysis"'
        )
    else:
        book_ref = (
            "BOOK: Read the book title and author from the book_metadata "
            "provided by the previous agent in the conversation history."
        )
        search_hint = (
            "Search queries to use (substitute the actual title and author):\n"
            '- "<title> <author> setting location"\n'
            '- "<title> <author> time period historical context"\n'
            '- "<title> <author> themes analysis"'
        )

    book_context_researcher = LlmAgent(
        name="book_context_researcher",
        model=model,
        tools=[google_search_tool],
        instruction=f"""You are a literary research specialist. Research the book's setting and themes.

{book_ref}

Use google_search to find:
1. PRIMARY LOCATIONS: Where does the story take place? (cities, countries, regions)
2. TIME PERIOD: When does it take place? (specific year, era, or historical period)
3. MAIN THEMES: What are the central themes? (war, love, survival, etc.)

{search_hint}

Provide detailed findings with specific locations and time periods.

ERROR HANDLING & GRACEFUL DEGRADATION:
- If search returns limited results, report what you found
- If certain aspects (location, time, themes) are unclear, note this explicitly
- If search fails, acknowledge the error and provide any context you can infer
- Prioritize factual information over completeness
- Clearly distinguish verified facts from inferences""",
    )

    book_context_formatter = LlmAgent(
        name="book_context_formatter",
        model=model,
        output_schema=BookContext,
        output_key="book_context",
        instruction="""Format the research into a validated BookContext object.

IMPORTANT:
- If the research found no context information, return empty lists/null fields - do not hallucinate
- Include only information actually found in the research results
- Accept partial information if complete context is unavailable

Extract:
- primary_locations: List of specific locations (cities, countries, regions)
- time_period: The historical era or specific timeframe
- themes: List of main themes from the book

GRACEFUL DEGRADATION:
- If locations are unclear, use broader regions (e.g., "Europe" instead of specific city)
- If time period is unknown, leave as null rather than guessing
- Accept any number of themes (even 1-2 if that's all that was found)
- Be specific and factual with what information IS available""",
    )

    return SequentialAgent(
        name="book_context_pipeline",
        sub_agents=[book_context_researcher, book_context_formatter],
    )
