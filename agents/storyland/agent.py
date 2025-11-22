"""
StoryLand Agent for ADK Web UI.

This is the entry point for `adk web agents/`.
"""

import os
import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from google.genai import types
from google.adk.models.google_llm import Gemini

from agents.orchestrator import create_workflow
from tools.google_books import google_books_tool

# Load environment variables
load_dotenv()

# Create model
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

model = Gemini(
    model=os.getenv("MODEL_NAME", "gemini-2.0-flash-lite"),
    api_key=api_key,
    retry_options=types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    ),
)

# Create root agent for ADK web
root_agent = create_workflow(model, google_books_tool)
