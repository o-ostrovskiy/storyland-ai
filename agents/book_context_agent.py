"""
Book context research agent.

Two-stage pipeline that researches the book's setting, time period, and themes
using Google Search.
"""

from google.adk.agents import LlmAgent
from models.book import BookContext
from agents.prompts import AgentPrompts, load_prompts


def create_book_context_agents(
    model,
    google_search_tool,
    book_title: str = "",
    author: str = "",
    prompts: AgentPrompts | None = None,
):
    """
    Create the book context research pipeline.

    Args:
        model: The LLM model to use
        google_search_tool: The Google Search tool
        book_title: The exact book title from book_metadata (optional for eval workflow)
        author: The exact author name from book_metadata (optional for eval workflow)
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        (researcher, formatter) LlmAgent pair that researches and formats book context
    """
    if prompts is None:
        prompts = load_prompts()

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
        instruction=prompts.book_context_researcher.format(
            book_ref=book_ref, search_hint=search_hint
        ),
    )

    book_context_formatter = LlmAgent(
        name="book_context_formatter",
        model=model,
        output_schema=BookContext,
        output_key="book_context",
        instruction=prompts.book_context_formatter,
    )

    return book_context_researcher, book_context_formatter
