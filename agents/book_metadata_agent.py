"""
Book metadata extraction agent.

Two-stage pipeline that searches Google Books API and formats the result
into validated BookMetadata.
"""

from google.adk.agents import LlmAgent, SequentialAgent
from models.book import BookMetadata


def create_book_metadata_pipeline(model, google_books_tool):
    """
    Create the book metadata extraction pipeline.

    Args:
        model: The LLM model to use
        google_books_tool: The Google Books FunctionTool

    Returns:
        SequentialAgent that extracts and formats book metadata
    """
    # Stage 1: Tool agent
    book_metadata_researcher = LlmAgent(
        name="book_metadata_researcher",
        model=model,
        tools=[google_books_tool],
        instruction="""You are a book metadata specialist. Extract complete book information.

1. Use the search_book tool with the book title provided by the user
2. Return ALL the information you receive from the tool
3. Do not summarize or modify the data - pass it through completely""",
    )

    # Stage 2: Pydantic formatter
    book_metadata_formatter = LlmAgent(
        name="book_metadata_formatter",
        model=model,
        output_schema=BookMetadata,
        output_key="book_metadata",
        instruction="""Format the book metadata into a validated BookMetadata object.

Extract from the previous response:
- book_title: The full book title
- author: Primary author name
- description: Book synopsis/description
- published_date: Publication date
- categories: List of genres/categories
- image_url: Cover image URL

Return a properly structured BookMetadata object.""",
    )

    # Pipeline
    return SequentialAgent(
        name="book_metadata_pipeline",
        sub_agents=[book_metadata_researcher, book_metadata_formatter],
    )
