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

1. Use the search_book tool with BOTH title AND author parameters when available
   - If user provides "The Great Gatsby by F. Scott Fitzgerald", use title="The Great Gatsby" and author="F. Scott Fitzgerald"
   - If user only provides a title, use just the title parameter
2. If the tool returns an error, report it clearly and explain what went wrong
3. If no results found, acknowledge this and suggest the user verify the title/author
4. Return ALL the information from successful results - do not summarize

IMPORTANT: Always pass the author parameter if mentioned - this prevents returning wrong books with same titles.""",
    )

    # Stage 2: Pydantic formatter
    book_metadata_formatter = LlmAgent(
        name="book_metadata_formatter",
        model=model,
        output_schema=BookMetadata,
        output_key="book_metadata",
        instruction="""Format the book metadata into a validated BookMetadata object.

IMPORTANT: If the researcher found no book or reported an error, return empty/null fields - do not hallucinate metadata.

Extract from the previous response:
- book_title: The full book title
- author: Primary author name
- description: Book synopsis/description
- published_date: Publication date
- categories: List of genres/categories
- image_url: Cover image URL

Return a properly structured BookMetadata object. Include only data actually found in the research.""",
    )

    # Pipeline
    return SequentialAgent(
        name="book_metadata_pipeline",
        sub_agents=[book_metadata_researcher, book_metadata_formatter],
    )
