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
        instruction="""Extract book title and author from the user's message and format as BookMetadata.

EXTRACTION RULES:
- book_title: Extract the title from patterns like "for {title}", "based on {title}",
  "from {title}", "{title} by {author}", or the main subject of the request.
  For author-focused requests with no specific title (e.g. "visiting places connected
  to Ernest Hemingway's life and works"), set book_title to "{Author}'s works"
  (e.g. "Ernest Hemingway's works").
- author: Extract from "by {author}" or possessive patterns like "{author}'s".
  If the author is not mentioned in the message, use an empty string — do NOT invent
  or guess an author name.
- description, published_date, categories: Leave empty/null — do not infer or hallucinate.
- image_url: null
- book_found: true

CRITICAL: Never invent an author that was not explicitly stated in the user's message.
If only a title is provided (e.g. "Pride and Prejudice"), set author to "".""",
    )
