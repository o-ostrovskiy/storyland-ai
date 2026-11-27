"""
Book-related Pydantic models.

Contains models for book metadata from Google Books API and book context
(setting, time period, themes).
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class BookMetadata(BaseModel):
    """Book metadata from Google Books API"""

    book_title: str = Field(description="Full book title")
    author: str = Field(description="Primary author name")
    description: Optional[str] = Field(
        default=None, description="Book description/synopsis"
    )
    published_date: Optional[str] = Field(default=None, description="Publication date")
    categories: List[str] = Field(
        default_factory=list, description="Book categories/genres"
    )
    image_url: Optional[str] = Field(
        default=None, description="Book cover image URL"
    )
    book_found: bool = Field(
        default=True,
        description="Whether the book was found via Google Books API. False when search returned no results.",
    )


class BookContext(BaseModel):
    """Book setting and context information"""

    primary_locations: List[str] = Field(
        description="Main locations where story takes place"
    )
    time_period: Optional[str] = Field(
        default=None, description="Historical era or time period of the story"
    )
    themes: List[str] = Field(description="Main themes of the book")
    # Prompt v7 (2026-11-25): the names the itinerary page bolds wherever they
    # appear (core/reading_highlights.py). Optional: older cached discovery
    # and the local-atmosphere flow leave it empty.
    main_characters: List[str] = Field(
        default_factory=list,
        description="Main characters (or, for non-fiction, main real people), as the book spells them",
    )
    # Book identity (prompts v6). A web search can resolve a title to a
    # DIFFERENT work -- "dddd" by "ddd" came back as Inio Asano's "Dead Dead
    # Demon's Dededede Destruction", set in Tokyo -- and every later agent then
    # plans a trip for that other book. discover() reads these three fields
    # and stops the run (core/book_identity.py). All optional: v1-v4 prompts
    # and the local-atmosphere flow never fill them, which means "unknown".
    matches_requested_book: Optional[bool] = Field(
        default=None,
        description=(
            "True if the research describes the requested book (same work and "
            "author), false if it describes a different book or found none, "
            "null if the research does not say"
        ),
    )
    identified_title: Optional[str] = Field(
        default=None,
        description="Title of the book the research actually describes; null if none",
    )
    identified_author: Optional[str] = Field(
        default=None,
        description="Author of the book the research actually describes; null if none",
    )


class BookRecommendation(BaseModel):
    """A single book recommended based on the current book and destination."""

    title: str = Field(description="Full book title")
    author: str = Field(description="Primary author name")
    description: Optional[str] = Field(default=None, description="Brief synopsis or description")
    published_date: Optional[str] = Field(default=None, description="Publication year or date")
    image_url: Optional[str] = Field(default=None, description="Cover image URL if known; leave null otherwise")
    reason: str = Field(description="1-2 sentences explaining why this book fits the reader's context")
    recommendation_basis: Literal["destination", "themes", "author"] = Field(
        description="Primary reason for this recommendation: 'destination' (set in same place), 'themes' (similar mood/themes), or 'author' (same author)"
    )


def _rec_min_results() -> int:
    """Minimum number of recommendations the formatter schema accepts.

    Lowered from a hard 5 to a tunable floor (default 3) so the tool-less
    formatter is never forced to invent a book to satisfy ``min_length`` when
    the grounded researcher returns fewer than 5 real candidates. Read from
    ``REC_MIN_RESULTS`` (env-driven, safe default), clamped to 1..5.
    """
    import os

    try:
        value = int(os.getenv("REC_MIN_RESULTS", "3"))
    except (TypeError, ValueError):
        value = 3
    return max(1, min(value, 5))


REC_MIN_RESULTS = _rec_min_results()


class BookRecommendationsResult(BaseModel):
    """Result from the book recommendation agent: 3-5 recommended books.

    The floor is ``REC_MIN_RESULTS`` (default 3) rather than a hard 5: a few
    grounded recommendations are better than padding with an invented book.
    """

    recommendations: List[BookRecommendation] = Field(
        min_length=REC_MIN_RESULTS,
        max_length=5,
        description=(
            f"{REC_MIN_RESULTS}-5 book recommendations, balanced across "
            "destination/themes/author bases; never pad with invented books"
        ),
    )


class BookInfo(BaseModel):
    """Book information from Google Books API (internal use)"""

    title: str
    authors: List[str]
    description: Optional[str] = None
    published_date: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    image_url: Optional[str] = Field(
        default=None, description="Book cover image URL"
    )
