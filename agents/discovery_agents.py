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
4. NON-URBAN SETTINGS: If the book is a memoir, trail narrative, or road trip, search for gateway towns and route stops: "[book title] route towns", "[book title] trail towns", "gateway towns [book setting]"

Search queries to try:
- "[book title] real locations to visit"
- "[book title] filming locations"
- "[book title] [author name] inspiration places"
- "[setting location] literary tourism"

For recent fiction (published after 2000) that may not yet have established literary tourism, also try:
- "[city] [book title] filming location"
- "[book title] real-life inspiration"
- "[author name] [city] residence"
Even a loose connection (author lived there, setting inspired by the city) is valid.

**FICTIONAL/SPACE/FUTURE WORLDS — MANDATORY REDIRECT:** If the book's primary_locations from context are clearly non-Earth (space stations, fictional planets, asteroid belts, alien worlds, invented continents), you MUST NOT search for those locations and you MUST NOT return empty-handed. Instead, call google_search immediately with these queries and return the results:
- "[author name] hometown city"
- "[author name] where did they live"
- "[book title] filming location"
- "[book title] real world inspiration"
You MUST call google_search at least once even for space/fictional books. Author hometowns and filming locations are always valid real-world cities.

For EACH city found, explain:
- What is the city name and country?
- How does it relate to the book? (setting, filming, author connection)

GOAL: Find at least 2-3 cities.

ERROR HANDLING & GRACEFUL DEGRADATION:
- If search returns limited results, report what you found (even if less than 3 cities)
- If search fails completely, explain the error clearly and suggest alternatives
- If rate limited, inform the user to retry later
- Always return SOME results if any information is available, rather than failing completely
- Clearly distinguish between verified facts and educated guesses based on the book's content""",
    )

    city_formatter = LlmAgent(
        name="city_formatter",
        model=model,
        output_schema=CityDiscovery,
        output_key="city_discovery",
        instruction="""Format the cities into validated CityDiscovery.

IMPORTANT:
- If the research found no cities, return an empty list - do not hallucinate
- Include only cities actually mentioned in the research results
- If the researcher encountered errors but found partial results, include those partial results

For each city found, create a CityInfo with:
- name: City name only (e.g., "Paris", not "Paris, France")
- country: Country name (e.g., "France")
- relevance: One sentence explaining the connection to the book

GRACEFUL DEGRADATION:
- Accept partial results (1-2 cities) if that's all that was found
- Validate that each city has at minimum a name and country
- Skip cities with insufficient information rather than guessing
- VALIDATION GATE: The `country` field MUST be a real, sovereign nation that exists on Earth today (e.g. "Romania", "USA", "Japan"). Reject ANY entry where the country is a planet ("Mars"), moon, asteroid, space station, fictional place, "Solar System", "Outer Space", or anything that is not a real Earth country. If no valid Earth-country cities were found, return an empty cities list.

Return CityDiscovery with a list of cities.""",
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
4. PLAQUES & CONTEMPORARY MARKERS: For recent or contemporary fiction, search for: "[book title] blue plaque", "[book title] commemorative site", "[setting neighbourhood] [book title]"

Search queries to try:
- "[book title] landmarks mentioned"
- "[city name] [book title] tour"
- "[book title] museum exhibit"
- "[setting] places from [book title]"

For EACH landmark, provide:
- Exact name of the place
- Which city it's in
- Specific connection to the book (Was it mentioned? Does it relate to a scene?)

For non-urban books (trails, wilderness, road trips), scenic and route-based stops are valid: trailheads, viewpoints, rest stops, and local cafés along the route. Label these with confidence level "scenic_stop" or "route_point".

GOAL: Find at least 3-5 landmarks across the cities.

ERROR HANDLING & GRACEFUL DEGRADATION:
- If search finds fewer landmarks, report what you discovered (quality over quantity)
- If certain cities have no specific landmarks, suggest general atmospheric locations
- If search fails, fall back to well-known locations mentioned in the book itself
- Report partial results rather than failing completely
- Clearly indicate confidence level: "mentioned in book" vs "related to setting" vs "suggested for atmosphere" vs "scenic_stop/route_point""",
    )

    landmark_formatter = LlmAgent(
        name="landmark_formatter",
        model=model,
        output_schema=LandmarkDiscovery,
        output_key="landmark_discovery",
        instruction="""Format the landmarks into validated LandmarkDiscovery.

IMPORTANT:
- If the research found no landmarks, return an empty list - do not hallucinate
- Include only landmarks actually mentioned in the research results
- Accept partial results if searches encountered issues

For each landmark found, create a LandmarkInfo with:
- name: Exact name of the landmark or place
- city: City where it's located
- connection: One sentence explaining how it relates to the book

GRACEFUL DEGRADATION:
- Accept any number of results (even 1-2 landmarks if that's all that was found)
- Validate that each landmark has a name, city, and connection
- Skip landmarks with insufficient information

Return LandmarkDiscovery with a list of landmarks.""",
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

GOAL: Find at least 2-3 author-related sites.

ERROR HANDLING & GRACEFUL DEGRADATION:
- If author information is limited, report what you can find (even if less than 3 sites)
- For contemporary or obscure authors, it's OK to find fewer sites
- If search fails, acknowledge this and explain that author sites may not be publicly available
- Return partial results rather than no results
- Clearly indicate which information is verified vs. inferred""",
    )

    author_formatter = LlmAgent(
        name="author_formatter",
        model=model,
        output_schema=AuthorSites,
        output_key="author_sites",
        instruction="""Format the author sites into validated AuthorSites.

IMPORTANT:
- If the research found no author sites, return an empty list - do not hallucinate
- Include only sites actually mentioned in the research results
- Accept partial results if the author has limited public sites

For each site found, create an AuthorSiteInfo with:
- name: Exact name of the site
- type: Type of site (e.g., "museum", "birthplace", "statue", "house")
- city: City where it's located

GRACEFUL DEGRADATION:
- Accept any number of results (0-3+ sites depending on author prominence)
- For contemporary or private authors, an empty list is acceptable
- Validate that each site has a name, type, and city

Return AuthorSites with a list of author_sites.""",
    )

    return SequentialAgent(
        name="author_pipeline", sub_agents=[author_researcher, author_formatter]
    )
