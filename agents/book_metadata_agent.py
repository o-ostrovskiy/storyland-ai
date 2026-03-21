"""
Book metadata formatting agent.

Single-stage agent that formats provided book title and author
into a validated BookMetadata schema.

WHY SINGLE STAGE?
The book title and author are now pre-confirmed by the upstream service
(Backend → storyland-ai). There is no need to call Google Books API here.
The agent's job is only to validate and structure the data already in session
state into the BookMetadata schema.
"""

from google.adk.agents import LlmAgent
from models.book import BookMetadata


def create_book_metadata_pipeline(model):
    """
    Create the book metadata formatting agent.

    Reads book_title and author from session state (provided by the caller)
    and structures them into a validated BookMetadata object.

    Args:
        model: The LLM model to use

    Returns:
        LlmAgent that formats book metadata from session state
    """
    return LlmAgent(
        name="book_metadata_pipeline",
        model=model,
        output_schema=BookMetadata,
        output_key="book_metadata",
        instruction="""Format the book information from the conversation into a BookMetadata object.

The book title and author have been confirmed and are provided in the conversation.
Extract them and structure as BookMetadata:
- book_title: The full book title as provided
- author: The author name as provided
- description: Leave empty string if not provided
- published_date: Leave empty string if not provided
- categories: Empty list if not provided
- image_url: null if not provided
- book_found: true (the caller has confirmed this book exists)

Use only data explicitly present in the conversation. Do not invent any details.""",
    )
