"""
StoryLand AI - CLI Entry Point

Transform books into travel adventures using AI agents.
"""

import asyncio
import uuid
import logging
from typing import Optional

from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner

# Import configuration
from common.config import load_config
from common.logging import setup_logger

# Import services
from services.session_service import create_session_service
from services.memory_service import create_memory_service
from services.context_manager import ContextManager

# Import tools and agents
from tools.google_books import google_books_tool
from agents.orchestrator import create_workflow


async def create_itinerary(
    book_title: str,
    author: Optional[str] = None,
    user_id: str = "user1",
    use_database: bool = False,
    use_memory: bool = False,
):
    """
    Create a literary travel itinerary for a book.

    Args:
        book_title: Title of the book
        author: Optional author name
        user_id: User ID for session tracking
        use_database: If True, use SQLite for session persistence
        use_memory: If True, enable memory service for personalization

    Returns:
        TripItinerary object with the complete travel plan
    """
    # Load configuration
    config = load_config()

    # Setup logging
    logger = setup_logger("storyland_main", level=config.log_level)
    logger.info(f"Creating itinerary for: {book_title}")

    # Configure model with retry options
    retry_config = types.HttpRetryOptions(
        attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
    )

    model = Gemini(
        model=config.model_name, api_key=config.google_api_key, retry_options=retry_config
    )

    # Create services
    session_service = create_session_service(
        connection_string=config.database_url, use_database=use_database or config.use_database
    )

    memory_service = None
    if use_memory or config.use_memory:
        memory_service = create_memory_service()
        logger.info("Memory service enabled")

    context_manager = ContextManager(max_events=config.session_max_events)

    # Create workflow
    workflow = create_workflow(model, google_books_tool)
    logger.info("Workflow created")

    # Create runner
    runner = Runner(
        agent=workflow, app_name="storyland", session_service=session_service
    )

    # Create session
    session_id = str(uuid.uuid4())
    await session_service.create_session(
        app_name="storyland",
        user_id=user_id,
        session_id=session_id,
        state={"book_title": book_title, "author": author or ""},
    )
    logger.info(f"Session created: {session_id[:8]}...")

    # Create prompt
    prompt = f"""Create a literary travel itinerary for "{book_title}" by {author or 'unknown author'}.

Execute all steps and return the complete combined results."""

    user_message = types.Content(role="user", parts=[types.Part(text=prompt)])

    # Run workflow
    print(f"\n{'='*70}")
    print(f"Creating itinerary for: {book_title}")
    if author:
        print(f"Author: {author}")
    print(f"{'='*70}\n")

    final_response = None
    event_count = 0

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=user_message
    ):
        event_count += 1

        # Show progress
        if event.author:
            print(f"[{event_count}] {event.author}")

        if event.is_final_response():
            final_response = event
            print(f"\n✅ Workflow complete ({event_count} events)")

    # Extract result
    result_data = None
    if final_response and final_response.content and final_response.content.parts:
        for part in final_response.content.parts:
            if hasattr(part, "text") and part.text:
                import json

                json_start = part.text.find("{")
                json_end = part.text.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    try:
                        result_data = json.loads(part.text[json_start:json_end])
                        break
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON: {e}")

    # Store session in memory (if enabled)
    if memory_service:
        session = await session_service.get_session(
            app_name="storyland", user_id=user_id, session_id=session_id
        )
        await memory_service.add_session_to_memory(session)
        logger.info("Session added to memory")

    return result_data


def display_itinerary(result_data: dict):
    """
    Display the travel itinerary in a formatted way.

    Args:
        result_data: The TripItinerary data dictionary
    """
    if not result_data:
        print("\n❌ No itinerary data found")
        return

    cities_list = result_data.get("cities", [])

    if not cities_list:
        print("\n❌ No cities found in result")
        return

    print(f"\n{'='*70}")
    print(f"TRAVEL ITINERARY - {len(cities_list)} Cities")
    print(f"{'='*70}\n")

    for i, city_plan in enumerate(cities_list, 1):
        print(f"City #{i}: {city_plan.get('name')}, {city_plan.get('country')}")
        print(f"Days suggested: {city_plan.get('days_suggested', 1)}")

        if city_plan.get("overview"):
            print(f"\nOverview:")
            print(f"  {city_plan.get('overview')}")

        stops = city_plan.get("stops", [])
        if stops:
            print(f"\nStops ({len(stops)}):")
            print("-" * 66)

            for j, stop in enumerate(stops, 1):
                print(f"\n{j}. {stop.get('name')}")
                print(f"   Type: {stop.get('type', 'other')}")
                print(f"   Time: {stop.get('time_of_day', 'unknown').upper()}")

                if stop.get("reason"):
                    print(f"   Why: {stop.get('reason')}")

                if stop.get("notes"):
                    print(f"   Notes: {stop.get('notes')}")

        print("\n")

    # Summary
    summary_text = result_data.get("summary_text")
    if summary_text:
        print("=" * 70)
        print("TRIP SUMMARY")
        print("=" * 70)
        print(f"\n{summary_text}\n")

    # Statistics
    total_cities = len(cities_list)
    total_stops = sum(len(city.get("stops", [])) for city in cities_list)
    total_days = sum(city.get("days_suggested", 0) for city in cities_list)

    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Total cities: {total_cities}")
    print(f"Total stops: {total_stops}")
    print(f"Total days: {total_days}")
    print()


async def main():
    """Main entry point for the CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create literary travel itineraries from books"
    )
    parser.add_argument("book_title", help="Title of the book")
    parser.add_argument("--author", "-a", help="Author name (optional)")
    parser.add_argument("--user-id", "-u", default="user1", help="User ID")
    parser.add_argument(
        "--database", "-d", action="store_true", help="Use SQLite database for sessions"
    )
    parser.add_argument(
        "--memory", "-m", action="store_true", help="Enable memory service"
    )

    args = parser.parse_args()

    # Create itinerary
    result = await create_itinerary(
        book_title=args.book_title,
        author=args.author,
        user_id=args.user_id,
        use_database=args.database,
        use_memory=args.memory,
    )

    # Display results
    display_itinerary(result)


if __name__ == "__main__":
    asyncio.run(main())
