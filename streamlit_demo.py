"""
StoryLand AI - Streamlit Demo

Interactive demo showing how to transform books into travel itineraries.

Usage:
    streamlit run streamlit_demo.py
"""

import asyncio
import json
import uuid
from typing import Optional, List

import streamlit as st
from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import LoggingPlugin
from plugins.langfuse_plugin import LangfusePlugin

from common.config import load_config
from common.logging import configure_logging, get_logger
from common.langfuse_init import initialize_langfuse
from services.session_service import create_session_service
from models.book import BookMetadata
from agents.orchestrator import (
    create_discovery_workflow,
    create_composition_workflow,
)


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="StoryLand AI",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📚 StoryLand AI")
    st.markdown("**Transform your favorite books into meaningful travel experiences**")
    st.markdown("---")


def display_book_info(book_metadata: dict, compact: bool = False) -> None:
    """Display book information with cover image.

    Args:
        book_metadata: Dictionary containing book information
        compact: If True, show a smaller, more compact version
    """
    title = book_metadata.get("book_title", "Unknown Title")
    author = book_metadata.get("author", "Unknown Author")
    description = book_metadata.get("description", "")
    published_date = book_metadata.get("published_date", "")
    categories = book_metadata.get("categories", [])
    image_url = book_metadata.get("image_url")

    if compact:
        # Compact version - show inline
        cols = st.columns([1, 4])
        with cols[0]:
            if image_url:
                try:
                    st.image(image_url, width=100)
                except Exception as e:
                    st.caption("📖 No image")
            else:
                st.caption("📖")
        with cols[1]:
            st.markdown(f"**{title}** by {author}")
            if published_date:
                st.caption(f"Published: {published_date}")
    else:
        # Full version
        st.markdown("### 📚 Book Information")
        col1, col2 = st.columns([1, 2])

        with col1:
            if image_url:
                try:
                    st.image(image_url, use_container_width=True)
                except Exception as e:
                    st.warning(f"📖 Cover image unavailable")
                    # Debug info (can be removed in production)
                    with st.expander("Debug Info"):
                        st.code(f"Image URL: {image_url}\nError: {str(e)}")
            else:
                st.info("📖 No cover image available")

        with col2:
            st.markdown(f"**Title:** {title}")
            st.markdown(f"**Author:** {author}")
            if published_date:
                st.markdown(f"**Published:** {published_date}")
            if categories:
                st.markdown(f"**Genres:** {', '.join(categories[:3])}")

        if description:
            with st.expander("📖 Book Description", expanded=False):
                st.markdown(description)


def display_execution_trace(trace_events: list) -> None:
    """Display execution trace of agent activity."""
    if not trace_events:
        return

    with st.expander(f"🔍 Execution Trace ({len(trace_events)} events)", expanded=False):
        st.markdown("**Agent Execution Timeline:**")

        # Group by phase
        phase1_events = [e for e in trace_events if e.get("phase") == "Phase 1"]
        phase2_events = [e for e in trace_events if e.get("phase") == "Phase 2"]

        if phase1_events:
            st.markdown("**Phase 1: Metadata Extraction**")
            for event in phase1_events:
                agent_name = event.get("agent", "unknown")
                st.code(f"→ {agent_name}", language=None)

        if phase2_events:
            st.markdown("**Phase 2: Location Discovery**")
            for event in phase2_events:
                agent_name = event.get("agent", "unknown")
                st.code(f"→ {agent_name}", language=None)

        st.caption(f"Total execution events: {len(trace_events)}")


def display_discoveries_preview(session_state_data: dict) -> None:
    """Display preview of discovered locations."""
    st.markdown("### 🔍 Discovered Locations Preview")

    # Get discoveries from session state
    city_discovery = session_state_data.get("city_discovery", {})
    landmark_discovery = session_state_data.get("landmark_discovery", {})
    author_sites = session_state_data.get("author_sites", {})

    cities = city_discovery.get("cities", [])
    landmarks = landmark_discovery.get("landmarks", [])
    sites = author_sites.get("author_sites", [])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🏙️ Cities Found", len(cities))
        if cities:
            with st.expander("View Cities", expanded=False):
                for city in cities[:5]:
                    st.markdown(f"**{city.get('name', '?')}, {city.get('country', '?')}**")
                    st.caption(city.get('relevance', ''))
                if len(cities) > 5:
                    st.caption(f"...and {len(cities) - 5} more")

    with col2:
        st.metric("🏛️ Landmarks Found", len(landmarks))
        if landmarks:
            with st.expander("View Landmarks", expanded=False):
                for landmark in landmarks[:5]:
                    st.markdown(f"**{landmark.get('name', '?')}**")
                    st.caption(f"{landmark.get('city', '?')}")
                if len(landmarks) > 5:
                    st.caption(f"...and {len(landmarks) - 5} more")

    with col3:
        st.metric("✍️ Author Sites Found", len(sites))
        if sites:
            with st.expander("View Author Sites", expanded=False):
                for site in sites[:5]:
                    st.markdown(f"**{site.get('name', '?')}**")
                    st.caption(f"{site.get('type', '?')} - {site.get('city', '?')}")
                if len(sites) > 5:
                    st.caption(f"...and {len(sites) - 5} more")


def display_region_card(region: dict, region_id: int) -> None:
    """Display a single region as a card."""
    region_name = region.get("region_name", "Unknown Region")
    cities = region.get("cities", [])
    days = region.get("estimated_days", "?")
    travel_note = region.get("travel_note", "")
    highlights = region.get("highlights", "")

    city_names = ", ".join(c.get("name", "?") for c in cities)

    st.markdown(f"""
    <div style="border: 2px solid #1f77b4; border-radius: 10px; padding: 15px; margin: 10px 0;">
        <h4 style="color: #1f77b4; margin-top: 0;">🌍 {region_name}</h4>
        <p><strong>Cities:</strong> {city_names}</p>
        <p><strong>Duration:</strong> ~{days} days</p>
        <p><strong>Travel:</strong> {travel_note}</p>
        <p><strong>Highlights:</strong> {highlights}</p>
    </div>
    """, unsafe_allow_html=True)


def display_itinerary_card(city_plan: dict, city_number: int) -> None:
    """Display a city itinerary as an expandable card."""
    city_name = city_plan.get("name", "Unknown City")
    country = city_plan.get("country", "")
    days = city_plan.get("days_suggested", 1)
    overview = city_plan.get("overview", "")
    stops = city_plan.get("stops", [])

    with st.expander(f"🏙️ {city_name}, {country} ({days} days)", expanded=True):
        if overview:
            st.markdown(f"**Overview:** {overview}")
            st.markdown("---")

        if stops:
            st.markdown(f"**Stops ({len(stops)}):**")
            for i, stop in enumerate(stops, 1):
                stop_name = stop.get("name", "Unknown")
                stop_type = stop.get("type", "other")
                time_of_day = stop.get("time_of_day", "anytime").upper()
                reason = stop.get("reason", "")
                notes = stop.get("notes", "")

                # Icon mapping for stop types
                icons = {
                    "landmark": "🏛️",
                    "museum": "🏛️",
                    "author_site": "✍️",
                    "restaurant": "🍽️",
                    "cafe": "☕",
                    "bookstore": "📚",
                    "other": "📍"
                }
                icon = icons.get(stop_type, "📍")

                # Use native Streamlit container instead of HTML for dark mode support
                with st.container():
                    st.markdown(f"**{i}. {icon} {stop_name}**")
                    st.caption(f"{stop_type.replace('_', ' ').title()} • {time_of_day}")
                    if reason:
                        st.markdown(f"**Why:** {reason}")
                    if notes:
                        st.markdown(f"**Notes:** {notes}")
                    st.markdown("")  # Add spacing between stops


async def run_workflow(
    book_title: str,
    author: Optional[str],
    book_metadata: dict,
    preferences: dict,
    progress_placeholder,
    status_placeholder,
    trace_placeholder=None
) -> Optional[dict]:
    """
    Run the discovery workflow and return location data.

    Book metadata must be pre-resolved before calling this function.
    This runs Phase 2 (discovery) only.

    Args:
        book_title: The book title
        author: The author name
        book_metadata: Pre-resolved book metadata dict
        preferences: User travel preferences
        progress_placeholder: Streamlit progress bar
        status_placeholder: Streamlit status message
        trace_placeholder: Optional placeholder for displaying execution traces

    Returns:
        Dictionary with keys: 'region_analysis', 'book_metadata', discoveries
    """
    trace_events = []  # Collect trace events for display
    config = load_config()

    # Configure logging
    configure_logging(level="INFO", enable_adk_debug=False)
    logger = get_logger("storyland.streamlit")

    # Initialize Langfuse tracing if configured
    initialize_langfuse(
        secret_key=config.langfuse_secret_key,
        public_key=config.langfuse_public_key,
        host=config.langfuse_host,
    )

    # Configure model
    retry_config = types.HttpRetryOptions(
        attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
    )
    model = Gemini(
        model=config.model_name, api_key=config.google_api_key, retry_options=retry_config
    )

    # Create services
    session_service = create_session_service(
        connection_string=config.database_url, use_database=False  # Use in-memory for demo
    )

    # Initialize Langfuse plugin
    langfuse_plugin = LangfusePlugin(
        secret_key=config.langfuse_secret_key,
        public_key=config.langfuse_public_key,
        host=config.langfuse_host,
    )

    # Build initial state with pre-resolved book metadata
    initial_state = {
        "book_title": book_title,
        "author": author or "",
        "book_metadata": book_metadata,
    }
    if preferences:
        initial_state["user:preferences"] = preferences

    # Create session
    session_id = str(uuid.uuid4())
    user_id = "streamlit_demo"

    await session_service.create_session(
        app_name="storyland",
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )

    workflow_data = {}

    # Book metadata must be pre-resolved and passed in via session state
    book_metadata = initial_state.get("book_metadata")
    if not book_metadata:
        raise ValueError("book_metadata must be set in initial_state before calling run_workflow")

    try:
        exact_title = book_metadata.get("book_title", book_title)
        exact_author = book_metadata.get("author", author or "Unknown")
        published_date = book_metadata.get("published_date", "")

        workflow_data["book_metadata"] = book_metadata

        logger.info("book_metadata_resolved",
                   title=exact_title,
                   author=exact_author,
                   has_image=bool(book_metadata.get("image_url")))

        book_info_msg = f"**Book:** \"{exact_title}\" by {exact_author}"
        if published_date:
            book_info_msg += f" ({published_date})"
        status_placeholder.success(book_info_msg)

        # Phase 2: Discovery - find locations and analyze regions
        progress_placeholder.progress(0.3)
        status_placeholder.info("🗺️ **Phase 2:** Researching book setting and themes...")

        discovery_workflow = create_discovery_workflow(
            model, book_title=exact_title, author=exact_author
        )
        discovery_runner = Runner(
            agent=discovery_workflow,
            app_name="storyland",
            session_service=session_service,
            plugins=[LoggingPlugin(), langfuse_plugin],
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
                # Collect trace events
                if event.author:
                    trace_events.append({
                        "phase": "Phase 2",
                        "agent": event.author,
                        "type": "agent_event",
                        "timestamp": str(event.create_time) if hasattr(event, 'create_time') else "N/A"
                    })
                    # Update trace display in real-time
                    if trace_placeholder:
                        trace_text = "**🔍 Live Execution Trace:**\n\n"
                        for t in trace_events:
                            trace_text += f"➤ [{t['phase']}] {t['agent']}\n"
                        trace_placeholder.code(trace_text, language=None)

                # Update status with more detail
                if event.author == "city_pipeline":
                    progress_placeholder.progress(0.4)
                    status_placeholder.info("🏙️ **Phase 2:** Finding cities related to the book...")
                elif event.author == "landmark_pipeline":
                    progress_placeholder.progress(0.5)
                    status_placeholder.info("🏛️ **Phase 2:** Discovering landmarks and key locations...")
                elif event.author == "author_pipeline":
                    progress_placeholder.progress(0.6)
                    status_placeholder.info("✍️ **Phase 2:** Locating author-related sites...")
                elif event.author == "region_analyzer":
                    progress_placeholder.progress(0.7)
                    status_placeholder.info("🌍 **Phase 2:** Analyzing geographic regions...")

        # Get region analysis and discoveries from session state
        session = await session_service.get_session(
            app_name="storyland", user_id=user_id, session_id=session_id
        )
        region_analysis = session.state.get("region_analysis", {})
        workflow_data["region_analysis"] = region_analysis

        # Also get discovery data for preview
        workflow_data["city_discovery"] = session.state.get("city_discovery", {})
        workflow_data["landmark_discovery"] = session.state.get("landmark_discovery", {})
        workflow_data["author_sites"] = session.state.get("author_sites", {})

        # Calculate totals for success message
        num_cities = len(workflow_data["city_discovery"].get("cities", []))
        num_landmarks = len(workflow_data["landmark_discovery"].get("landmarks", []))
        num_sites = len(workflow_data["author_sites"].get("author_sites", []))
        num_regions = len(region_analysis.get("regions", []))

        progress_placeholder.progress(1.0)
        status_placeholder.success(
            f"✅ **Discovery Complete!** Found {num_cities} cities, {num_landmarks} landmarks, "
            f"{num_sites} author sites across {num_regions} travel region(s)"
        )

        # Add traces to workflow data
        workflow_data["trace_events"] = trace_events

        # Add token usage stats
        if langfuse_plugin.enabled:
            workflow_data["token_stats"] = langfuse_plugin.get_session_stats()
            await langfuse_plugin.flush()

        # Return here to let user select regions
        return workflow_data

    except Exception as e:
        logger.error("workflow_error", error=str(e))
        status_placeholder.error(f"❌ Error: {str(e)}")
        return None


async def create_itinerary_for_regions(
    selected_regions: List[dict],
    book_metadata: dict,
    preferences: dict,
    progress_placeholder,
    status_placeholder
) -> Optional[dict]:
    """Create itinerary for selected regions."""
    config = load_config()
    configure_logging(level="INFO", enable_adk_debug=False)
    logger = get_logger("storyland.streamlit")

    # Configure model
    retry_config = types.HttpRetryOptions(
        attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
    )
    model = Gemini(
        model=config.model_name, api_key=config.google_api_key, retry_options=retry_config
    )

    # Create services
    session_service = create_session_service(
        connection_string=config.database_url, use_database=False
    )

    # Initialize Langfuse plugin
    langfuse_plugin = LangfusePlugin(
        secret_key=config.langfuse_secret_key,
        public_key=config.langfuse_public_key,
        host=config.langfuse_host,
    )

    # Create session with selected regions
    session_id = str(uuid.uuid4())
    user_id = "streamlit_demo"

    exact_title = book_metadata.get("book_title", "")
    exact_author = book_metadata.get("author", "")

    initial_state = {
        "book_title": exact_title,
        "author": exact_author,
        "book_metadata": book_metadata,
        "selected_regions": selected_regions,
    }
    if preferences:
        initial_state["user:preferences"] = preferences

    await session_service.create_session(
        app_name="storyland",
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )

    try:
        # Phase 3: Composition
        progress_placeholder.progress(0.1)
        region_count = len(selected_regions)
        region_names = ", ".join(r.get("region_name", "?") for r in selected_regions)
        status_placeholder.info(f"✈️ **Phase 3:** Creating personalized itinerary for {region_names}...")

        composition_workflow = create_composition_workflow(model)
        composition_runner = Runner(
            agent=composition_workflow,
            app_name="storyland",
            session_service=session_service,
            plugins=[LoggingPlugin(), langfuse_plugin],
        )

        composition_prompt = f"""Create a travel itinerary for "{exact_title}" by {exact_author}.

Use ONLY the cities from the selected region(s): {json.dumps(selected_regions)}

Create a personalized itinerary based on user preferences and the selected region(s).
Include ALL cities from the selected regions in your itinerary."""
        composition_message = types.Content(
            role="user", parts=[types.Part(text=composition_prompt)]
        )

        final_response = None
        async with composition_runner:
            async for event in composition_runner.run_async(
                user_id=user_id, session_id=session_id, new_message=composition_message
            ):
                # Update progress during composition
                if event.author == "trip_composer":
                    progress_placeholder.progress(0.5)
                    status_placeholder.info("🗺️ **Phase 3:** Composing detailed itinerary with personalized recommendations...")
                if event.is_final_response():
                    final_response = event

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

        progress_placeholder.progress(1.0)

        # Show success message with details
        if result_data:
            num_cities = len(result_data.get("cities", []))
            total_stops = sum(len(city.get("stops", [])) for city in result_data.get("cities", []))
            status_placeholder.success(
                f"✅ **Itinerary Complete!** Created personalized plan with {num_cities} cities and {total_stops} stops"
            )
        else:
            status_placeholder.success("✅ Itinerary created successfully!")

        # Add token usage stats
        if langfuse_plugin.enabled:
            token_stats = langfuse_plugin.get_session_stats()
            if result_data:
                result_data["token_stats"] = token_stats
            await langfuse_plugin.flush()

        return result_data

    except Exception as e:
        logger.error("composition_error", error=str(e))
        status_placeholder.error(f"❌ Error: {str(e)}")
        return None


def main():
    """Main Streamlit app."""
    setup_page()

    # Initialize session state
    if 'workflow_data' not in st.session_state:
        st.session_state.workflow_data = None
    if 'itinerary_data' not in st.session_state:
        st.session_state.itinerary_data = None
    if 'book_search_results' not in st.session_state:
        st.session_state.book_search_results = None
    if 'selected_book_metadata' not in st.session_state:
        st.session_state.selected_book_metadata = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'input'  # input, book_selection, region_selection, itinerary

    # Sidebar for inputs
    with st.sidebar:
        st.header("📖 Book Selection")

        book_title = st.text_input(
            "Book Title",
            placeholder="e.g., Pride and Prejudice",
            help="Enter the title of the book you want to explore"
        )

        author = st.text_input(
            "Author",
            placeholder="e.g., Jane Austen",
            help="Author name is required"
        )

        st.markdown("---")
        st.header("⚙️ Travel Preferences")

        budget = st.selectbox(
            "Budget",
            options=["moderate", "budget", "luxury"],
            help="Your travel budget preference"
        )

        pace = st.selectbox(
            "Travel Pace",
            options=["moderate", "relaxed", "fast-paced"],
            help="How fast-paced you want your trip to be"
        )

        prefers_museums = st.checkbox("Prefer Museums", value=True)
        travels_with_kids = st.checkbox("Traveling with Kids", value=False)

        st.markdown("---")
        start_button = st.button("🚀 Create Itinerary", type="primary", use_container_width=True)

        # Reset button
        if st.session_state.current_step != 'input':
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state.workflow_data = None
                st.session_state.itinerary_data = None
                st.session_state.book_search_results = None
                st.session_state.selected_book_metadata = None
                st.session_state.current_step = 'input'
                st.rerun()

    # Build preferences
    preferences = {
        "budget": budget,
        "preferred_pace": pace,
        "prefers_museums": prefers_museums,
        "travels_with_kids": travels_with_kids,
    }

    # Main content area
    if start_button and st.session_state.current_step == 'input':
        if not book_title:
            st.error("Please enter a book title")
            return
        if not author or not author.strip():
            st.error("Please enter the author name")
            return

        # Build BookMetadata directly from provided title and author
        metadata = BookMetadata(
            book_title=book_title.strip(),
            author=author.strip(),
            book_found=True,
        )
        st.session_state.selected_book_metadata = metadata.model_dump()
        st.session_state.current_step = 'run_discovery'
        st.rerun()

    # Run discovery workflow (after book is selected)
    if st.session_state.current_step == 'run_discovery' and st.session_state.selected_book_metadata:
        book_metadata = st.session_state.selected_book_metadata
        display_book_info(book_metadata, compact=True)

        progress_placeholder = st.progress(0)
        status_placeholder = st.empty()
        trace_placeholder = st.empty()

        try:
            workflow_data = asyncio.run(
                run_workflow(
                    book_metadata.get("book_title", ""),
                    book_metadata.get("author", ""),
                    book_metadata,
                    preferences,
                    progress_placeholder,
                    status_placeholder,
                    trace_placeholder,
                )
            )

            if workflow_data:
                st.session_state.workflow_data = workflow_data
                st.session_state.current_step = 'region_selection'
                st.rerun()
            else:
                st.error("Failed to discover locations. Please try again.")
                return
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
            return

    # Region selection screen
    if st.session_state.current_step == 'region_selection' and st.session_state.workflow_data:
        workflow_data = st.session_state.workflow_data

        # Display book information (compact version with smaller image)
        st.markdown("---")
        book_metadata = workflow_data.get("book_metadata", {})
        if book_metadata:
            display_book_info(book_metadata, compact=True)

        # Display execution trace
        trace_events = workflow_data.get("trace_events", [])
        if trace_events:
            st.markdown("---")
            display_execution_trace(trace_events)

        # Display discoveries preview
        st.markdown("---")
        display_discoveries_preview(workflow_data)

        # Display regions
        st.markdown("---")
        st.header("🌍 Select Travel Region(s)")

        region_analysis = workflow_data.get("region_analysis", {})
        regions = region_analysis.get("regions", [])

        if not regions:
            st.error("❌ No regions found. Try a different book.")
            return

        analysis_note = region_analysis.get("analysis_note", "")
        if analysis_note:
            st.info(analysis_note)

        # Display regions as cards
        st.markdown("### Available Regions:")
        for region in regions:
            display_region_card(region, region.get("region_id", 0))

        # Region selection
        st.markdown("---")
        selected_region_ids = st.multiselect(
            "Select one or more regions to explore:",
            options=[r.get("region_id") for r in regions],
            default=[regions[0].get("region_id")] if regions else [],
            format_func=lambda rid: next(
                (r.get("region_name", f"Region {rid}") for r in regions if r.get("region_id") == rid),
                f"Region {rid}"
            )
        )

        if not selected_region_ids:
            st.warning("⚠️ Please select at least one region to continue")
            return

        # Get selected region objects
        selected_regions = [r for r in regions if r.get("region_id") in selected_region_ids]

        # Button to create itinerary
        if st.button("✈️ Create Detailed Itinerary", type="primary"):
            # Create placeholders for progress
            progress_placeholder = st.progress(0)
            status_placeholder = st.empty()

            # Run Phase 3 (composition)
            try:
                itinerary_data = asyncio.run(
                    create_itinerary_for_regions(
                        selected_regions,
                        workflow_data.get("book_metadata", {}),
                        preferences,
                        progress_placeholder,
                        status_placeholder
                    )
                )

                if itinerary_data:
                    st.session_state.itinerary_data = itinerary_data
                    st.session_state.current_step = 'itinerary'
                    st.rerun()
                else:
                    st.error("❌ Failed to create itinerary. Please try again.")
                    return
            except Exception as e:
                st.error(f"❌ Error creating itinerary: {str(e)}")
                st.exception(e)
                return

    # Itinerary display screen
    if st.session_state.current_step == 'itinerary' and st.session_state.itinerary_data:
        # Display book information at the top (compact version)
        if st.session_state.workflow_data:
            book_metadata = st.session_state.workflow_data.get("book_metadata", {})
            if book_metadata:
                st.markdown("---")
                display_book_info(book_metadata, compact=True)

        st.markdown("---")
        st.header("🗺️ Your Travel Itinerary")

        itinerary_data = st.session_state.itinerary_data
        cities_list = itinerary_data.get("cities", [])

        if not cities_list:
            st.error("❌ No cities found in itinerary")
            return

        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cities", len(cities_list))
        with col2:
            total_days = sum(city.get("days_suggested", 0) for city in cities_list)
            st.metric("Total Days", total_days)
        with col3:
            total_stops = sum(len(city.get("stops", [])) for city in cities_list)
            st.metric("Total Stops", total_stops)

        st.markdown("---")

        # Display each city
        for i, city_plan in enumerate(cities_list, 1):
            display_itinerary_card(city_plan, i)

        # Trip summary
        if itinerary_data.get("summary_text"):
            st.markdown("---")
            st.header("📝 Trip Summary")
            st.markdown(itinerary_data.get("summary_text"))

    # Welcome screen (default)
    if st.session_state.current_step == 'input' and not start_button:
        st.markdown("""
        ### Welcome to StoryLand AI!

        Transform your favorite books into real travel experiences. Here's how it works:

        1. **📚 Enter a book title** - Tell us which book you'd like to explore
        2. **⚙️ Set your preferences** - Budget, pace, and travel style
        3. **🌍 Choose regions** - Select which geographic areas you want to visit
        4. **🗺️ Get your itinerary** - Receive a detailed, personalized travel plan

        ### Features:
        - **AI-powered research** - Discovers locations tied to book settings, author sites, and themes
        - **Geographic grouping** - Regions organized for practical travel planning
        - **Personalized itineraries** - Tailored to your budget and travel preferences
        - **Detailed stops** - Each location includes context, timing, and tips

        **Ready to start?** Enter a book title in the sidebar and click "Create Itinerary"!

        ### Example Books:
        - Pride and Prejudice by Jane Austen
        - The Great Gatsby by F. Scott Fitzgerald
        - 1984 by George Orwell
        - The Nightingale by Kristin Hannah
        """)


if __name__ == "__main__":
    main()
