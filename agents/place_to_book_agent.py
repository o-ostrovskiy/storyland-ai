"""
Place→book reverse-routing agent (AI candidate layer).

Two-stage pipeline mirroring book_recommendation_agent: a researcher uses
google_search to find books set in / evoking a destination, and a tool-less
formatter shapes the grounded candidates into PlaceToBookCandidates, labelled
literal (set there) vs. vibe (feels like it).

This is the reverse of the existing book→place discovery flow: the place is the
input and books are the output. No new model and no new paid API — it reuses the
same google_search researcher pass already used across discovery.
"""

from google.adk.agents import LlmAgent, SequentialAgent

from models.place_to_book import PlaceToBookCandidates
from agents.prompts import AgentPrompts, load_prompts


# Session-state output_key the formatter writes its result under.
PLACE_TO_BOOK_OUTPUT_KEY = "last_place_to_book"


def create_place_to_book_pipeline(
    model,
    google_search_tool,
    place: str,
    prompts: AgentPrompts | None = None,
) -> SequentialAgent:
    """Create the place→book reverse-routing pipeline.

    Researcher uses google_search to find books genuinely set in (or strongly
    associated with) the place plus thematically-fitting "vibe" books; the
    tool-less formatter shapes them into a PlaceToBookCandidates result with
    literal/vibe labels. An ungroundable place yields an empty candidate list.

    Args:
        model: The LLM model to use.
        google_search_tool: The Google Search tool.
        place: Free-text destination (e.g. "Lisbon", "the Scottish Highlands").
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        SequentialAgent that researches and formats place→book candidates.
    """
    if prompts is None:
        prompts = load_prompts()

    researcher_instruction = prompts.place_to_book_researcher.format(place=place)
    formatter_instruction = prompts.place_to_book_formatter.format(place=place)

    researcher = LlmAgent(
        name="place_to_book_researcher",
        model=model,
        tools=[google_search_tool],
        instruction=researcher_instruction,
    )

    formatter = LlmAgent(
        name="place_to_book_formatter",
        model=model,
        output_schema=PlaceToBookCandidates,
        output_key=PLACE_TO_BOOK_OUTPUT_KEY,
        instruction=formatter_instruction,
    )

    return SequentialAgent(
        name="place_to_book_pipeline",
        sub_agents=[researcher, formatter],
    )
