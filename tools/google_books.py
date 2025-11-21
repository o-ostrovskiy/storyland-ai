"""
Google Books API tool.

Provides search functionality for books using the Google Books API.
"""

import json
import logging
import requests
from typing import List, Optional

from google.adk.tools import FunctionTool
from models.book import BookInfo, BookMetadata

logger = logging.getLogger(__name__)


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
    logger.info(f"Searching Google Books for: {title}")

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

        print(f"Searching: {query}")
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

        logger.info(f"Found {len(results)} books")
        print(f"Found {len(results)} books")

        # Log all found books
        for i, book in enumerate(results):
            author_str = ", ".join(book.authors) if book.authors else "Unknown"
            print(
                f"   [{i}] {book.title} by {author_str} ({book.published_date or 'N/A'})"
            )

        return results

    except Exception as e:
        logger.error(f"Google Books search failed: {str(e)}")
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
    logger.info(f"search_book called: title='{title}', author='{author}'")

    try:
        # Search for books
        books = search_books(title=title, author=author or None, max_results=5)

        if not books:
            logger.warning(f"No books found for: {title}")
            return json.dumps(
                {"error": "No books found", "query": {"title": title, "author": author}}
            )

        # Select the first/best match
        selected = books[0]
        author_str = ", ".join(selected.authors) if selected.authors else "Unknown"

        print(f"✅ Selected: {selected.title} by {author_str}")
        logger.info(f"Selected book: {selected.title} by {author_str}")

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
        logger.error(f"Book search failed: {str(e)}", exc_info=True)
        return json.dumps({"error": str(e), "type": type(e).__name__})


# Create FunctionTool
google_books_tool = FunctionTool(search_book)
