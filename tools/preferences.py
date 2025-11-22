"""
Preferences tool for accessing user preferences from session state.

Provides tools for agents to read user preferences stored in session state.
"""

import json
from google.adk.tools import FunctionTool, ToolContext


def get_user_preferences(tool_context: ToolContext) -> str:
    """
    Get user preferences from session state.

    Retrieves the user:preferences key from session state which contains
    personalization settings like budget, pace, and interests.

    Returns:
        JSON string with user preferences, or empty object if none set.
    """
    preferences = tool_context.state.get("user:preferences", {})

    if preferences:
        return json.dumps({
            "found": True,
            "preferences": preferences
        })
    else:
        return json.dumps({
            "found": False,
            "preferences": {},
            "message": "No user preferences found. Using defaults."
        })


# Create FunctionTool
get_preferences_tool = FunctionTool(get_user_preferences)
