"""
Region analyzer agent.

Analyzes discovered cities and groups them into practical travel regions
based on geographic proximity using LLM world knowledge.
"""

from google.adk.agents import LlmAgent
from models.discovery import RegionAnalysis
from agents.prompts import AgentPrompts, load_prompts


def create_region_analyzer_agent(model, prompts: AgentPrompts | None = None):
    """
    Create the region analyzer agent.

    This agent analyzes all discovered cities from the conversation history
    and groups them into practical travel regions that can be visited together.

    Args:
        model: The LLM model to use
        prompts: Optional AgentPrompts instance. Loads default version if not provided.

    Returns:
        LlmAgent that produces RegionAnalysis with grouped regions
    """
    if prompts is None:
        prompts = load_prompts()
    return LlmAgent(
        name="region_analyzer",
        model=model,
        output_schema=RegionAnalysis,
        output_key="region_analysis",
        instruction=prompts.region_analyzer,
    )
