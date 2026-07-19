"""
Discovery agents for cities, landmarks, and author sites.

Three researcher/formatter agent pairs (run as parallel graph branches) that discover places to visit related to the book.
"""

from google.adk.agents import LlmAgent
from models.discovery import CityDiscovery, LandmarkDiscovery, AuthorSites
from agents.prompts import AgentPrompts, load_prompts


def create_city_agents(model, google_search_tool, prompts: AgentPrompts | None = None):
    """
    Create the city discovery pipeline.

    Args:
        model: The LLM model to use
        google_search_tool: The Google Search tool
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        (researcher, formatter) LlmAgent pair that discovers and formats cities to visit
    """
    if prompts is None:
        prompts = load_prompts()

    city_researcher = LlmAgent(
        name="city_researcher",
        model=model,
        tools=[google_search_tool],
        instruction=prompts.city_researcher,
    )

    city_formatter = LlmAgent(
        name="city_formatter",
        model=model,
        output_schema=CityDiscovery,
        output_key="city_discovery",
        instruction=prompts.city_formatter,
    )

    return city_researcher, city_formatter


def create_landmark_agents(model, google_search_tool, prompts: AgentPrompts | None = None):
    """
    Create the landmark discovery pipeline.

    Args:
        model: The LLM model to use
        google_search_tool: The Google Search tool
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        (researcher, formatter) LlmAgent pair that discovers and formats landmarks to visit
    """
    if prompts is None:
        prompts = load_prompts()

    landmark_researcher = LlmAgent(
        name="landmark_researcher",
        model=model,
        tools=[google_search_tool],
        instruction=prompts.landmark_researcher,
    )

    landmark_formatter = LlmAgent(
        name="landmark_formatter",
        model=model,
        output_schema=LandmarkDiscovery,
        output_key="landmark_discovery",
        instruction=prompts.landmark_formatter,
    )

    return landmark_researcher, landmark_formatter


def create_author_agents(model, google_search_tool, prompts: AgentPrompts | None = None):
    """
    Create the author sites discovery pipeline.

    Args:
        model: The LLM model to use
        google_search_tool: The Google Search tool
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        (researcher, formatter) LlmAgent pair that discovers and formats author-related sites
    """
    if prompts is None:
        prompts = load_prompts()

    author_researcher = LlmAgent(
        name="author_researcher",
        model=model,
        tools=[google_search_tool],
        instruction=prompts.author_researcher,
    )

    author_formatter = LlmAgent(
        name="author_formatter",
        model=model,
        output_schema=AuthorSites,
        output_key="author_sites",
        instruction=prompts.author_formatter,
    )

    return author_researcher, author_formatter
