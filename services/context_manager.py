"""
Context manager for managing conversation history and token limits.

Provides strategies for keeping conversations within token budgets through:
- Sliding window (keep recent N events)
- Event summarization
- Hybrid approach (recent events + relevant memories)
"""

from typing import List, Optional
from google.adk.sessions import Session
from google.genai import types


class ContextManager:
    """
    Manages conversation context to stay within token limits.

    Strategies:
    1. Simple: Keep only recent N events (sliding window)
    2. Advanced: Summarize older events + keep recent events
    3. Hybrid: Recent events + relevant memory search results
    """

    def __init__(
        self,
        max_events: int = 20,
        max_tokens: Optional[int] = None,
        preserve_system: bool = True,
    ):
        """
        Initialize the context manager.

        Args:
            max_events: Maximum number of events to keep (default: 20)
            max_tokens: Optional token limit (not yet implemented)
            preserve_system: Always preserve system prompts (default: True)
        """
        self.max_events = max_events
        self.max_tokens = max_tokens
        self.preserve_system = preserve_system

    def limit_events(self, events: List, num_recent: int = 20) -> List:
        """
        Simple sliding window: Keep only recent N events.

        Args:
            events: List of session events
            num_recent: Number of recent events to keep

        Returns:
            List of events limited to num_recent

        Example:
            >>> context_manager = ContextManager(max_events=20)
            >>> limited = context_manager.limit_events(session.events, num_recent=15)
        """
        if len(events) <= num_recent:
            return events

        # Keep the most recent events
        return events[-num_recent:]

    def get_context_stats(self, events: List) -> dict:
        """
        Get statistics about the current context.

        Args:
            events: List of session events

        Returns:
            Dictionary with context statistics

        Example:
            >>> stats = context_manager.get_context_stats(session.events)
            >>> print(f"Events: {stats['num_events']}")
        """
        num_events = len(events)

        # Estimate tokens (rough approximation: 4 chars per token)
        total_chars = 0
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        total_chars += len(part.text)

        estimated_tokens = total_chars // 4

        return {
            "num_events": num_events,
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
            "within_limit": (
                num_events <= self.max_events
                if self.max_events
                else True
            ),
        }

    def should_compact(self, events: List) -> bool:
        """
        Check if context should be compacted.

        Args:
            events: List of session events

        Returns:
            True if context exceeds limits

        Example:
            >>> if context_manager.should_compact(session.events):
            ...     session.events = context_manager.limit_events(session.events)
        """
        if self.max_events and len(events) > self.max_events:
            return True

        if self.max_tokens:
            stats = self.get_context_stats(events)
            return stats["estimated_tokens"] > self.max_tokens

        return False


# Example usage patterns:

# Pattern 1: Simple sliding window
# -------------------------------
# context_manager = ContextManager(max_events=20)
#
# # Before running agent
# if context_manager.should_compact(session.events):
#     session.events = context_manager.limit_events(session.events, num_recent=15)


# Pattern 2: Using GetSessionConfig (ADK built-in)
# ------------------------------------------------
# from google.adk.sessions import GetSessionConfig
#
# session = await session_service.get_session(
#     app_name="storyland",
#     user_id=user_id,
#     session_id=session_id,
#     config=GetSessionConfig(
#         num_recent_events=20  # Only get recent events
#     )
# )


# Pattern 3: Hybrid with memory (future implementation)
# -----------------------------------------------------
# class HybridContextManager(ContextManager):
#     def __init__(self, memory_service, recent_event_count=15):
#         super().__init__(max_events=recent_event_count)
#         self.memory_service = memory_service
#
#     async def build_context(self, session, current_query):
#         # Get recent events
#         recent = session.events[-self.max_events:]
#
#         # Search memory for relevant past context
#         memory_results = await self.memory_service.search_memory(
#             app_name=session.app_name,
#             user_id=session.user_id,
#             query=current_query
#         )
#
#         # Combine memories + recent events
#         return memory_results.memories[:5] + recent
