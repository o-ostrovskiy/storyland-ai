"""
Expansion agent.

Two-stage pipeline that finds new places to add to an existing literary travel
itinerary based on a user-selected suggestion chip. Mirrors the local_atmosphere_agent
pattern: researcher uses google_search, formatter emits structured output only.
"""

from google.adk.agents import LlmAgent

from models.itinerary import ExpansionResult
from agents.prompts import AgentPrompts, load_prompts


def create_expansion_agents(
    model,
    google_search_tool,
    book_title: str,
    author: str,
    parent_city: str,
    action_prompt: str,
    existing_places: str,
    prompts: AgentPrompts | None = None,
):
    """Create the expansion pipeline.

    Researcher uses google_search to find new places matching the action_prompt;
    formatter shapes the result into an ExpansionResult with new CityStop entries
    and follow-up SuggestionChips.

    Args:
        model: The LLM model to use.
        google_search_tool: The Google Search tool.
        book_title: Exact book title.
        author: Exact author name.
        parent_city: City where new places should be located.
        action_prompt: The expansion instruction from the suggestion chip.
        existing_places: Newline-separated list of "Name (City)" strings to avoid.
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        (researcher, formatter) LlmAgent pair that researches and formats an expansion result.
    """
    if prompts is None:
        prompts = load_prompts()

    researcher_instruction = prompts.expansion_researcher.format(
        book_title=book_title,
        author=author,
        parent_city=parent_city,
        action_prompt=action_prompt,
        existing_places=existing_places,
    )
    formatter_instruction = prompts.expansion_formatter.format(
        parent_city=parent_city,
        action_prompt=action_prompt,
    )

    researcher = LlmAgent(
        name="expansion_researcher",
        model=model,
        tools=[google_search_tool],
        instruction=researcher_instruction,
    )

    formatter = LlmAgent(
        name="expansion_formatter",
        model=model,
        output_schema=ExpansionResult,
        output_key="last_expansion",
        instruction=formatter_instruction,
    )

    return researcher, formatter
