"""
Google Books API tool.

Provides search functionality for books using the Google Books API.
"""

import json
import requests
from typing import List, Optional

from google.adk.tools import FunctionTool
from models.book import BookInfo, BookMetadata
from common.logging import get_logger

logger = get_logger(__name__)


def search_books(
    title: str, author: Optional[str] = None, max_results: int = 5
) -> List[BookInfo]:
    """
    Search Google Books API and return matching books.

    Args:
        title: Book title to search for
        author: Optional author name
        max_results: Maximum number of results

    Returns:
        List of BookInfo objects
    """
    logger.info("google_books_search", title=title, author=author)

    try:
        # Build query
        query_parts = []
        if title:
            query_parts.append(f"intitle:{title}")
        if author:
            query_parts.append(f"inauthor:{author}")

        query = "+".join(query_parts)

        # API request
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {"q": query, "maxResults": max_results, "printType": "books"}

        logger.debug("google_books_query", query=query, url=url)
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])

        # Parse results
        results = []
        for item in items:
            vol = item.get("volumeInfo", {})

            # Extract image URL (prefer larger sizes)
            image_links = vol.get("imageLinks", {})
            image_url = (
                image_links.get("large")
                or image_links.get("medium")
                or image_links.get("thumbnail")
                or image_links.get("smallThumbnail")
                or None
            )

            # Upgrade HTTP to HTTPS for Google Books images
            if image_url and image_url.startswith("http://"):
                image_url = image_url.replace("http://", "https://")

            results.append(
                BookInfo(
                    title=vol.get("title", "Unknown"),
                    authors=vol.get("authors", []),
                    description=vol.get("description"),
                    published_date=vol.get("publishedDate"),
                    categories=vol.get("categories", []),
                    image_url=image_url,
                )
            )

        logger.info("google_books_results", count=len(results))

        # Log all found books
        for i, book in enumerate(results):
            author_str = ", ".join(book.authors) if book.authors else "Unknown"
            logger.debug(
                "google_books_result_item",
                index=i,
                title=book.title,
                author=author_str,
                published_date=book.published_date or "N/A"
            )

        return results

    except Exception as e:
        logger.error("google_books_search_failed", error=str(e), error_type=type(e).__name__)
        raise


def search_book(title: str, author: str = "") -> str:
    """
    Search Google Books API and return Pydantic-validated book metadata.

    This tool searches for books, automatically selects the best match,
    and returns JSON validated against the BookMetadata Pydantic schema.

    Args:
        title: Book title to search for (e.g., "The Nightingale", "1984")
        author: Optional author name (e.g., "Kristin Hannah")

    Returns:
        JSON string conforming to BookMetadata schema
    """
    logger.info("search_book_called", title=title, author=author)

    try:
        # Search for books
        books = search_books(title=title, author=author or None, max_results=5)

        if not books:
            logger.warning("search_book_no_results", title=title, author=author)
            return json.dumps(
                {"error": "No books found", "query": {"title": title, "author": author}}
            )

        # Select the first/best match
        selected = books[0]
        author_str = ", ".join(selected.authors) if selected.authors else "Unknown"

        logger.info("google_books_selected", title=selected.title, author=author_str)

        # Create and validate with Pydantic
        book_metadata = BookMetadata(
            book_title=selected.title,
            author=author_str,
            description=selected.description,
            published_date=selected.published_date,
            categories=selected.categories or [],
            image_url=selected.image_url,
        )

        # Return Pydantic-validated JSON
        return book_metadata.model_dump_json()

    except Exception as e:
        logger.error("search_book_failed", error=str(e), error_type=type(e).__name__)
        return json.dumps({"error": str(e), "type": type(e).__name__})


# Create FunctionTool
google_books_tool = FunctionTool(search_book)
