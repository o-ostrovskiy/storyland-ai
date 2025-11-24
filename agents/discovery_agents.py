"""
Discovery agents for cities, landmarks, and author sites.

Three parallel two-stage pipelines that discover places to visit related to the book.
"""

from google.adk.agents import LlmAgent, SequentialAgent
from models.discovery import CityDiscovery, LandmarkDiscovery, AuthorSites


def create_city_pipeline(model, google_search_tool):
    """
    Create the city discovery pipeline.

    Args:
        model: The LLM model to use
        google_search_tool: The Google Search tool

    Returns:
        SequentialAgent that discovers and formats cities to visit
    """
    city_researcher = LlmAgent(
        name="city_researcher",
        model=model,
        tools=[google_search_tool],
        instruction="""You are a literary travel specialist. Find real cities readers can visit.

Based on the book's setting from the conversation history, use google_search to find:

1. SETTING CITIES: Actual cities where the story takes place
2. FILMING LOCATIONS: If adapted to film, where was it filmed?
3. AUTHOR'S CITIES: Where did the author live or get inspiration?

Search queries to try:
- "[book title] real locations to visit"
- "[book title] filming locations"
- "[book title] [author name] inspiration places"
- "[setting location] literary tourism"

For EACH city found, explain:
- What is the city name and country?
- How does it relate to the book? (setting, filming, author connection)

Find at least 2-3 cities.""",
    )

    city_formatter = LlmAgent(
        name="city_formatter",
        model=model,
        output_schema=CityDiscovery,
        output_key="city_discovery",
        instruction="""Format the cities into validated CityDiscovery.

IMPORTANT: If the research found no cities, return an empty list - do not hallucinate.

For each city found, create a CityInfo with:
- name: City name only (e.g., "Paris", not "Paris, France")
- country: Country name (e.g., "France")
- relevance: One sentence explaining the connection to the book

Return CityDiscovery with a list of cities. Include only cities actually mentioned in the research.""",
    )

    return SequentialAgent(
        name="city_pipeline", sub_agents=[city_researcher, city_formatter]
    )


def create_landmark_pipeline(model, google_search_tool):
    """
    Create the landmark discovery pipeline.

    Args:
        model: The LLM model to use
        google_search_tool: The Google Search tool

    Returns:
        SequentialAgent that discovers and formats landmarks to visit
    """
    landmark_researcher = LlmAgent(
        name="landmark_researcher",
        model=model,
        tools=[google_search_tool],
        instruction="""You are a landmark research specialist. Find specific places to visit.

Based on the book and cities from the conversation history, use google_search to find:

1. MENTIONED LANDMARKS: Specific buildings, museums, or places mentioned in the book
2. THEMED EXPERIENCES: Literary walks, museum exhibits, or tours related to the book
3. ATMOSPHERIC LOCATIONS: Places that capture the book's setting or mood

Search queries to try:
- "[book title] landmarks mentioned"
- "[city name] [book title] tour"
- "[book title] museum exhibit"
- "[setting] places from [book title]"

For EACH landmark, provide:
- Exact name of the place
- Which city it's in
- Specific connection to the book (Was it mentioned? Does it relate to a scene?)

Find at least 3-5 landmarks across the cities.""",
    )

    landmark_formatter = LlmAgent(
        name="landmark_formatter",
        model=model,
        output_schema=LandmarkDiscovery,
        output_key="landmark_discovery",
        instruction="""Format the landmarks into validated LandmarkDiscovery.

IMPORTANT: If the research found no landmarks, return an empty list - do not hallucinate.

For each landmark found, create a LandmarkInfo with:
- name: Exact name of the landmark or place
- city: City where it's located
- connection: One sentence explaining how it relates to the book

Return LandmarkDiscovery with a list of landmarks. Include only landmarks actually mentioned in the research.""",
    )

    return SequentialAgent(
        name="landmark_pipeline",
        sub_agents=[landmark_researcher, landmark_formatter],
    )


def create_author_pipeline(model, google_search_tool):
    """
    Create the author sites discovery pipeline.

    Args:
        model: The LLM model to use
        google_search_tool: The Google Search tool

    Returns:
        SequentialAgent that discovers and formats author-related sites
    """
    author_researcher = LlmAgent(
        name="author_researcher",
        model=model,
        tools=[google_search_tool],
        instruction="""You are an author biography specialist. Find places connected to the author.

Based on the author from the conversation history, use google_search to find:

1. AUTHOR'S HOMETOWN: Where was the author born or raised?
2. AUTHOR MUSEUMS: Are there museums or houses dedicated to the author?
3. WRITING LOCATIONS: Where did the author write this book?
4. MEMORIALS: Statues, plaques, or commemorative sites

Search queries to try:
- "[author name] birthplace"
- "[author name] museum"
- "[author name] house"
- "[author name] hometown"
- "where did [author name] write [book title]"

For EACH site, provide:
- Exact name
- Type (museum, birthplace, statue, etc.)
- Which city it's in

Find at least 2-3 author-related sites.""",
    )

    author_formatter = LlmAgent(
        name="author_formatter",
        model=model,
        output_schema=AuthorSites,
        output_key="author_sites",
        instruction="""Format the author sites into validated AuthorSites.

IMPORTANT: If the research found no author sites, return an empty list - do not hallucinate.

For each site found, create an AuthorSiteInfo with:
- name: Exact name of the site
- type: Type of site (e.g., "museum", "birthplace", "statue", "house")
- city: City where it's located

Return AuthorSites with a list of author_sites. Include only sites actually mentioned in the research.""",
    )

    return SequentialAgent(
        name="author_pipeline", sub_agents=[author_researcher, author_formatter]
    )
