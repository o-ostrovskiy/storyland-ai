"""
StoryLand AI - CLI Entry Point

Transform books into travel adventures using AI agents.

Usage:
    python main.py "1984" --author "George Orwell"
    python main.py "1984" -v   # verbose logging
    python main.py --dev       # ADK Web UI
"""

import asyncio
import uuid
import subprocess
import sys
from typing import Optional

from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner

from common.config import load_config
from common.logging import configure_logging, get_logger
from services.session_service import create_session_service
from services.context_manager import ContextManager
from tools.google_books import google_books_tool
from agents.orchestrator import create_workflow
from google.adk.plugins.logging_plugin import LoggingPlugin


async def create_itinerary(
    book_title: str,
    author: Optional[str] = None,
    user_id: str = "user1",
    use_database: bool = False,
    preferences: Optional[dict] = None,
    verbose: bool = False,
):
    """
    Create a literary travel itinerary for a book.

    Args:
        book_title: Title of the book
        author: Optional author name
        user_id: User ID for session tracking
        use_database: If True, use SQLite for session persistence
        preferences: Optional user preferences for personalization
        verbose: Enable DEBUG logging

    Returns:
        TripItinerary object with the complete travel plan
    """
    config = load_config()

    # Configure logging
    log_level = "DEBUG" if verbose else config.log_level
    configure_logging(level=log_level, enable_adk_debug=verbose or config.enable_adk_debug)

    logger = get_logger("storyland.main")
    logger.info("itinerary_request", book_title=book_title, author=author)

    # Configure model
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
    context_manager = ContextManager(max_events=config.session_max_events)

    # Create workflow with logging plugin
    workflow = create_workflow(model, google_books_tool)
    runner = Runner(
        agent=workflow,
        app_name="storyland",
        session_service=session_service,
        plugins=[LoggingPlugin()],
    )

    # Build initial state
    initial_state = {"book_title": book_title, "author": author or ""}
    if preferences:
        initial_state["user:preferences"] = preferences

    # Create session
    session_id = str(uuid.uuid4())
    await session_service.create_session(
        app_name="storyland",
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )
    logger.info("session_created", session_id=session_id[:8])

    # Create prompt
    prompt = f"""Create a literary travel itinerary for "{book_title}" by {author or 'unknown author'}.

Execute all steps and return the complete combined results."""

    user_message = types.Content(role="user", parts=[types.Part(text=prompt)])

    # Run workflow
    print(f"\n{'='*70}")
    print(f"Creating itinerary for: {book_title}")
    if author:
        print(f"Author: {author}")
    if preferences:
        print(f"Preferences: {', '.join(f'{k}={v}' for k, v in preferences.items())}")
    print(f"{'='*70}\n")

    final_response = None
    event_count = 0

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=user_message
    ):
        event_count += 1
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
                        logger.error("json_parse_error", error=str(e))

    # Display context statistics
    session = await session_service.get_session(
        app_name="storyland", user_id=user_id, session_id=session_id
    )
    stats = context_manager.get_context_stats(session.events)
    print(f"\n📊 Context: {stats['num_events']} events, ~{stats['estimated_tokens']} tokens")

    return result_data


def display_itinerary(result_data: dict):
    """Display the travel itinerary in a formatted way."""
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
            print(f"\nOverview:\n  {city_plan.get('overview')}")

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

    if result_data.get("summary_text"):
        print("=" * 70)
        print("TRIP SUMMARY")
        print("=" * 70)
        print(f"\n{result_data.get('summary_text')}\n")

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


def launch_adk_web():
    """Launch the ADK web UI for development."""
    print("\n🚀 Launching ADK Web UI...")
    print("   Open http://localhost:8000 to interact with the workflow.")
    print("   Note: Plugins are NOT supported in ADK web mode.\n")

    try:
        # ADK expects agents in subdirectories with agent.py files
        subprocess.run([".venv/bin/adk", "web", "--log_level", "DEBUG", "agents/"], check=True)
    except FileNotFoundError:
        print("Error: 'adk' command not found. Install with: pip install google-adk")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nADK Web UI stopped.")


async def main():
    """Main entry point for the CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create literary travel itineraries from books",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "1984" --author "George Orwell"
  python main.py "1984" -v --budget luxury
  python main.py --dev
        """,
    )

    parser.add_argument("book_title", nargs="?", help="Title of the book")
    parser.add_argument("--author", "-a", help="Author name (optional)")
    parser.add_argument("--user-id", "-u", default="user1", help="User ID")
    parser.add_argument("--database", "-d", action="store_true", help="Use SQLite for sessions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--dev", action="store_true", help="Launch ADK Web UI")

    # Preference arguments
    parser.add_argument("--budget", choices=["budget", "moderate", "luxury"])
    parser.add_argument("--pace", choices=["relaxed", "moderate", "fast-paced"])
    parser.add_argument("--museums", action="store_true", dest="prefers_museums")
    parser.add_argument("--no-museums", action="store_false", dest="prefers_museums")
    parser.add_argument("--with-kids", action="store_true")
    parser.set_defaults(prefers_museums=None)

    args = parser.parse_args()

    if args.dev:
        launch_adk_web()
        return

    if not args.book_title:
        parser.error("book_title is required (or use --dev for ADK Web UI)")

    preferences = {}
    if args.budget:
        preferences["budget"] = args.budget
    if args.pace:
        preferences["preferred_pace"] = args.pace
    if args.prefers_museums is not None:
        preferences["prefers_museums"] = args.prefers_museums
    if args.with_kids:
        preferences["travels_with_kids"] = True

    result = await create_itinerary(
        book_title=args.book_title,
        author=args.author,
        user_id=args.user_id,
        use_database=args.database,
        preferences=preferences if preferences else None,
        verbose=args.verbose,
    )

    display_itinerary(result)


if __name__ == "__main__":
    asyncio.run(main())
