"""
Google Books API tool.

Provides search functionality for books using the Google Books API.
"""

import json
import requests
from typing import List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)
from google.adk.tools import FunctionTool
from models.book import BookInfo, BookMetadata
from common.logging import get_logger

logger = get_logger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout)),
    before_sleep=before_sleep_log(logger, "warning"),
    after=after_log(logger, "info"),
)
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

    Raises:
        requests.exceptions.HTTPError: For rate limiting (429) or other HTTP errors
        requests.exceptions.Timeout: For timeout errors
        requests.exceptions.RequestException: For other request errors

    Note:
        This function implements exponential backoff retry logic:
        - Retries up to 3 times
        - Wait times: 2s, 4s, 8s (exponential backoff)
        - Retries on RequestException and Timeout
        - Logs retry attempts to Langfuse via structlog
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

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Handle rate limiting specifically
            if e.response.status_code == 429:
                logger.warning(
                    "google_books_rate_limited",
                    status_code=429,
                    message="Rate limit exceeded, will retry with backoff"
                )
            raise
        except requests.exceptions.Timeout as e:
            logger.warning(
                "google_books_timeout",
                timeout=10,
                message="Request timed out, will retry"
            )
            raise

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

    except requests.exceptions.RequestException as e:
        logger.error(
            "google_books_search_failed",
            error=str(e),
            error_type=type(e).__name__,
            message="All retry attempts exhausted"
        )
        raise
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
            return json.dumps({
                "error": "No books found matching your search",
                "query": {"title": title, "author": author},
                "suggestion": "Please verify the book title and author spelling, or try with just the title"
            })

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

    except requests.exceptions.RequestException as e:
        # Network/API errors - provide user-friendly message
        logger.error("search_book_failed", error=str(e), error_type=type(e).__name__)
        error_msg = "Failed to connect to Google Books API"
        if "429" in str(e):
            error_msg = "Rate limit exceeded - please try again in a few minutes"
        elif "timeout" in str(e).lower():
            error_msg = "Request timed out - please check your internet connection"
        return json.dumps({
            "error": error_msg,
            "type": type(e).__name__,
            "details": str(e)
        })
    except Exception as e:
        logger.error("search_book_failed", error=str(e), error_type=type(e).__name__)
        return json.dumps({
            "error": f"Unexpected error while searching for book: {str(e)}",
            "type": type(e).__name__
        })


# Create FunctionTool
google_books_tool = FunctionTool(search_book)
