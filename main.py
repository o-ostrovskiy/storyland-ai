"""
StoryLand AI - CLI Entry Point

Transform books into travel adventures using AI agents.

Usage:
    python main.py "1984" --author "George Orwell"
    python main.py "1984" -v   # verbose logging
    python main.py --dev       # ADK Web UI
"""

import asyncio
import json
import uuid
import subprocess
import sys
from typing import Optional, List

from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner

from common.config import load_config
from common.logging import configure_logging, get_logger
from services.session_service import create_session_service
from services.context_manager import ContextManager
from tools.google_books import google_books_tool
from agents.orchestrator import (
    create_metadata_stage,
    create_discovery_workflow,
    create_composition_workflow,
)
from google.adk.plugins.logging_plugin import LoggingPlugin


class WorkflowTimeoutError(Exception):
    """Raised when workflow execution exceeds the configured timeout."""
    pass


def display_region_options(region_analysis: dict) -> None:
    """Display region options to the user."""
    regions = region_analysis.get("regions", [])

    if not regions:
        print("\nNo regions found in analysis.")
        return

    print(f"\n{'='*70}")
    print("TRAVEL REGION OPTIONS")
    print(f"{'='*70}")
    print(f"\n{region_analysis.get('analysis_note', '')}\n")

    for region in regions:
        region_id = region.get("region_id", "?")
        region_name = region.get("region_name", "Unknown")
        cities = region.get("cities", [])
        days = region.get("estimated_days", "?")
        travel_note = region.get("travel_note", "")
        highlights = region.get("highlights", "")

        city_names = ", ".join(
            f"{c.get('name', '?')}" for c in cities
        )

        print(f"[{region_id}] {region_name}")
        print(f"    Cities: {city_names}")
        print(f"    Duration: ~{days} days")
        print(f"    Travel: {travel_note}")
        print(f"    Highlights: {highlights}")
        print()


def get_region_selection(region_analysis: dict) -> List[dict]:
    """
    Get user's region selection (supports multiple regions).

    Args:
        region_analysis: The region analysis dict from the agent

    Returns:
        List of selected region dicts
    """
    regions = region_analysis.get("regions", [])

    if not regions:
        return []

    # If only one region, auto-select it
    if len(regions) == 1:
        print(f"\nOnly one region available - automatically selected.")
        return [regions[0]]

    # Interactive selection
    valid_ids = [r.get("region_id") for r in regions]

    print(f"\nEnter region number(s) separated by commas (e.g., '1,2' for multiple regions)")
    print(f"Or press Enter to select all regions")

    while True:
        try:
            choice = input(f"Which region(s) would you like to explore? [{'/'.join(map(str, valid_ids))}]: ")

            # If empty, select all
            if not choice.strip():
                print(f"\nSelected all {len(regions)} regions")
                return regions

            # Parse comma-separated IDs
            selected_ids = [int(x.strip()) for x in choice.split(",")]
            selected = []

            for region in regions:
                if region.get("region_id") in selected_ids:
                    selected.append(region)

            if selected:
                names = ", ".join(r.get("region_name", "?") for r in selected)
                print(f"\nSelected {len(selected)} region(s): {names}")
                return selected

            print(f"Invalid choice. Please enter one or more of: {valid_ids}")
        except ValueError:
            print(f"Please enter number(s) separated by commas: {valid_ids}")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled.")
            return []


async def create_itinerary(
    book_title: str,
    author: Optional[str] = None,
    user_id: str = "user1",
    use_database: bool = False,
    preferences: Optional[dict] = None,
    verbose: bool = False,
    timeout: Optional[int] = None,
):
    """
    Create a literary travel itinerary for a book.

    Uses a three-phase workflow with human-in-the-loop region selection:
    1. Metadata extraction - identify book and author
    2. Discovery - find locations and group into travel regions
    3. Composition - create itinerary for selected region(s)

    Args:
        book_title: Title of the book
        author: Optional author name
        user_id: User ID for session tracking
        use_database: If True, use SQLite for session persistence
        preferences: Optional user preferences for personalization
        verbose: Enable DEBUG logging
        timeout: Maximum seconds for workflow execution (overrides config)

    Returns:
        TripItinerary dict with the complete travel plan, or None on failure

    Raises:
        WorkflowTimeoutError: If workflow execution exceeds timeout
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

    # Determine workflow timeout
    workflow_timeout = timeout if timeout is not None else config.workflow_timeout
    logger.info(
        "workflow_starting",
        book_title=book_title,
        author=author,
        timeout_seconds=workflow_timeout
    )

    # Display header
    print(f"\n{'='*70}")
    print(f"Creating itinerary for: {book_title}")
    if author:
        print(f"Author: {author}")
    if preferences:
        print(f"Preferences: {', '.join(f'{k}={v}' for k, v in preferences.items())}")
    logger.debug("workflow_timeout_config", timeout_seconds=workflow_timeout)
    print(f"{'='*70}\n")

    final_response = None
    event_count = 0

    try:
        async with asyncio.timeout(workflow_timeout):
            # Phase 1: Extract book metadata
            logger.info("phase_1_start", phase="metadata_extraction")
            print("Phase 1: Extracting book metadata...")
            metadata_stage = create_metadata_stage(model, google_books_tool)
            metadata_runner = Runner(
                agent=metadata_stage,
                app_name="storyland",
                session_service=session_service,
                plugins=[LoggingPlugin()],
            )

            metadata_prompt = f"""Find book metadata for "{book_title}" by {author or 'unknown author'}."""
            metadata_message = types.Content(role="user", parts=[types.Part(text=metadata_prompt)])

            async with metadata_runner:
                async for event in metadata_runner.run_async(
                    user_id=user_id, session_id=session_id, new_message=metadata_message
                ):
                    event_count += 1
                    if event.author:
                        logger.debug("agent_event", event_count=event_count, author=event.author)

            # Get metadata from session state
            session = await session_service.get_session(
                app_name="storyland", user_id=user_id, session_id=session_id
            )
            book_metadata = session.state.get("book_metadata", {})
            exact_title = book_metadata.get("book_title", book_title)
            exact_author = book_metadata.get("author", author or "Unknown")

            logger.info(
                "metadata_extracted",
                exact_title=exact_title,
                exact_author=exact_author
            )
            print(f"\nFound: \"{exact_title}\" by {exact_author}")

            # Phase 2: Discovery - find locations and analyze regions
            logger.info("phase_2_start", phase="location_discovery")
            print("\nPhase 2: Discovering locations and analyzing travel regions...")
            discovery_workflow = create_discovery_workflow(
                model, book_title=exact_title, author=exact_author
            )
            discovery_runner = Runner(
                agent=discovery_workflow,
                app_name="storyland",
                session_service=session_service,
                plugins=[LoggingPlugin()],
            )

            discovery_prompt = f"""Discover travel locations for "{exact_title}" by {exact_author}.

Find cities, landmarks, and author-related sites, then group them into practical travel regions."""
            discovery_message = types.Content(
                role="user", parts=[types.Part(text=discovery_prompt)]
            )

            async with discovery_runner:
                async for event in discovery_runner.run_async(
                    user_id=user_id, session_id=session_id, new_message=discovery_message
                ):
                    event_count += 1
                    if event.author:
                        logger.debug("agent_event", event_count=event_count, author=event.author)

            # Get region analysis from session state
            session = await session_service.get_session(
                app_name="storyland", user_id=user_id, session_id=session_id
            )
            region_analysis = session.state.get("region_analysis", {})

            logger.info(
                "regions_discovered",
                num_regions=len(region_analysis.get("regions", []))
            )

            # Display region options and get user selection
            display_region_options(region_analysis)
            selected_regions = get_region_selection(region_analysis)

            if not selected_regions:
                print("\nNo valid region selected. Using all discovered locations.")
                selected_regions = region_analysis.get("regions", [])

            # Validate that we have regions to work with
            if not selected_regions:
                error_msg = (
                    "No regions available to create an itinerary. "
                    "The discovery phase did not find enough locations to group into travel regions. "
                    "Try a different book or check the discovery results."
                )
                logger.error("no_regions_available", book_title=exact_title, author=exact_author)
                print(f"\n❌ Error: {error_msg}")
                raise WorkflowTimeoutError(error_msg)

            # Store selected regions in session state for trip_composer to access
            # Note: State changes are automatically persisted by DatabaseSessionService
            session = await session_service.get_session(
                app_name="storyland", user_id=user_id, session_id=session_id
            )
            session.state["selected_regions"] = selected_regions

            logger.info("selected_regions_stored", region_count=len(selected_regions))

            # Phase 3: Composition - create itinerary for selected region(s)
            if len(selected_regions) == 1:
                region_name = selected_regions[0].get("region_name", "all locations")
            else:
                region_name = f"{len(selected_regions)} regions"
            logger.info("phase_3_start", phase="itinerary_composition", region_count=len(selected_regions))
            print(f"\nPhase 3: Creating itinerary for {region_name}...")

            composition_workflow = create_composition_workflow(model)
            composition_runner = Runner(
                agent=composition_workflow,
                app_name="storyland",
                session_service=session_service,
                plugins=[LoggingPlugin()],
            )

            composition_prompt = f"""Create a travel itinerary for "{exact_title}" by {exact_author}.

Use ONLY the cities from the selected region(s): {json.dumps(selected_regions)}

Create a personalized itinerary based on user preferences and the selected region(s).
Include ALL cities from the selected regions in your itinerary."""
            composition_message = types.Content(
                role="user", parts=[types.Part(text=composition_prompt)]
            )

            async with composition_runner:
                async for event in composition_runner.run_async(
                    user_id=user_id, session_id=session_id, new_message=composition_message
                ):
                    event_count += 1
                    if event.author:
                        logger.debug("agent_event", event_count=event_count, author=event.author)
                    if event.is_final_response():
                        final_response = event
                        logger.info("workflow_complete", total_events=event_count)

    except asyncio.TimeoutError:
        logger.error(
            "workflow_timeout",
            book_title=book_title,
            timeout_seconds=workflow_timeout,
            events_processed=event_count
        )
        raise WorkflowTimeoutError(
            f"Workflow exceeded {workflow_timeout}s timeout. "
            f"Processed {event_count} events before timeout. "
            "Consider increasing WORKFLOW_TIMEOUT or simplifying the request."
        )

    # Extract result
    result_data = None
    if final_response and final_response.content and final_response.content.parts:
        for part in final_response.content.parts:
            if hasattr(part, "text") and part.text:
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
    logger.info("context_stats", num_events=stats['num_events'], estimated_tokens=stats['estimated_tokens'])
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
    parser.add_argument("--timeout", "-t", type=int, help="Workflow timeout in seconds (default: 300)")
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

    # Configure logging early for error handling
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(level=log_level)
    logger = get_logger("storyland.cli")

    try:
        result = await create_itinerary(
            book_title=args.book_title,
            author=args.author,
            user_id=args.user_id,
            use_database=args.database,
            preferences=preferences if preferences else None,
            verbose=args.verbose,
            timeout=args.timeout,
        )
        display_itinerary(result)
    except WorkflowTimeoutError as e:
        logger.error("workflow_failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
