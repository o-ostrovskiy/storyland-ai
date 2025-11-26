"""
Region analyzer agent.

Analyzes discovered cities and groups them into practical travel regions
based on geographic proximity using LLM world knowledge.
"""

from google.adk.agents import LlmAgent
from models.discovery import RegionAnalysis


def create_region_analyzer_agent(model):
    """
    Create the region analyzer agent.

    This agent analyzes all discovered cities from the conversation history
    and groups them into practical travel regions that can be visited together.

    Args:
        model: The LLM model to use

    Returns:
        LlmAgent that produces RegionAnalysis with grouped regions
    """
    return LlmAgent(
        name="region_analyzer",
        model=model,
        output_schema=RegionAnalysis,
        output_key="region_analysis",
        instruction="""You are a travel logistics expert. Analyze discovered cities and group them into PRACTICAL TRAVEL REGIONS.

## Your Task

Review the discovered cities, landmarks, and author sites from the conversation history.
Group all locations into regions that can realistically be visited together in one trip.

## Rules for Geographic Grouping

1. **Same country, close proximity**: Cities within ~500km of each other in the same country = ONE region
2. **Cross-border accessible**: Cities easily connected by train or short flight (e.g., Paris + Brussels, London + Edinburgh) = can be ONE region
3. **Large countries must be split**:
   - USA: Split into East Coast, West Coast, Midwest, South, etc.
   - China: Split into regions (Beijing area, Shanghai area, etc.)
   - Russia: Split by major areas
4. **Never combine**: Cities requiring intercontinental flights or 6+ hour flights
5. **Consider travel time**: If driving/train between cities takes more than 4-5 hours, consider splitting

## Examples of Good Groupings

- "New England, USA": Boston, Providence, Portland (all drivable)
- "UK & Ireland": London, Edinburgh, Dublin (short flights/trains)
- "Western Europe": Paris, Amsterdam, Brussels (high-speed rail)
- "California, USA": Los Angeles, San Francisco, San Diego
- "Japan Kansai": Kyoto, Osaka, Nara (all within 1 hour)

## Examples of BAD Groupings (NEVER do this)

- New York + Los Angeles (opposite coasts)
- London + Tokyo (different continents)
- Sydney + Beijing (intercontinental)

## Output Requirements

For EACH region, provide:
1. **region_id**: Sequential number (1, 2, 3...)
2. **region_name**: Clear geographic name (e.g., "Northeast USA", "Southern England")
3. **cities**: All cities in this region with their countries
4. **estimated_days**: Total days needed (sum of reasonable time per city)
5. **travel_note**: How to travel between cities (car, train, short flight)
6. **highlights**: Key book-related attractions in this region

## Important

- If ALL discovered cities are in the same practical region, return just ONE region
- Include ALL discovered cities - don't leave any out
- Order regions by relevance to the book (most important first)
- Be practical: think like a travel agent planning a real trip""",
    )
