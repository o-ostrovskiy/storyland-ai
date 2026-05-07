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


class BookRecommendationsResult(BaseModel):
    """Result from the book recommendation agent: 5 recommended books."""

    recommendations: List[BookRecommendation] = Field(
        min_length=5,
        max_length=5,
        description="Exactly 5 book recommendations, balanced across destination/themes/author bases",
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
