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
    try:
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
    except Exception as e:
        # Graceful degradation: return defaults on error. DELIBERATE fail-open:
        # ADK 2.x only auto-retries a tool when the exception PROPAGATES, so
        # swallowing here opts this tool out of framework retry — correct for a
        # read of session-local state, where "no preferences" is a valid answer
        # and a retry could never do better.
        return json.dumps({
            "found": False,
            "preferences": {},
            "error": f"Error reading preferences: {str(e)}",
            "message": "Using default preferences due to error."
        })


# Create FunctionTool
get_preferences_tool = FunctionTool(get_user_preferences)
