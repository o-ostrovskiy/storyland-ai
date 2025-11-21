"""
Book-related Pydantic models.

Contains models for book metadata from Google Books API and book context
(setting, time period, themes).
"""

from pydantic import BaseModel, Field
from typing import Optional, List


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


class BookContext(BaseModel):
    """Book setting and context information"""

    primary_locations: List[str] = Field(
        description="Main locations where story takes place"
    )
    time_period: str = Field(description="Historical era or time period of the story")
    themes: List[str] = Field(description="Main themes of the book")


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
