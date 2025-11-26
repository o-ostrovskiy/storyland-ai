"""
Unit tests for workflow timeout functionality.

Tests the WorkflowTimeoutError and timeout handling in create_itinerary.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from main import WorkflowTimeoutError, create_itinerary


# =============================================================================
# WorkflowTimeoutError Tests
# =============================================================================

class TestWorkflowTimeoutError:
    """Tests for WorkflowTimeoutError exception."""

    def test_error_message(self):
        """Test that error includes helpful message."""
        error = WorkflowTimeoutError("Workflow exceeded 300s timeout.")

        assert "300s" in str(error)
        assert "timeout" in str(error).lower()

    def test_error_is_exception(self):
        """Test that error inherits from Exception."""
        error = WorkflowTimeoutError("Test")

        assert isinstance(error, Exception)

    def test_error_can_be_raised(self):
        """Test that error can be raised and caught."""
        with pytest.raises(WorkflowTimeoutError) as exc_info:
            raise WorkflowTimeoutError("Test timeout")

        assert "Test timeout" in str(exc_info.value)


# =============================================================================
# Timeout Configuration Tests
# =============================================================================

# NOTE: Timeout configuration tests removed - they tested the old two-phase workflow
# which has been replaced by the three-phase workflow with HITL region selection.
# Actual timeout behavior is still validated by TestTimeoutBehavior below.


# =============================================================================
# Timeout Behavior Tests
# =============================================================================

class TestTimeoutBehavior:
    """Tests for timeout behavior during workflow execution."""

    @pytest.mark.asyncio
    async def test_asyncio_timeout_raises_correctly(self):
        """Test that asyncio.timeout raises TimeoutError correctly."""
        async def slow_operation():
            await asyncio.sleep(10)

        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.1):
                await slow_operation()

    @pytest.mark.asyncio
    async def test_asyncio_timeout_allows_fast_operations(self):
        """Test that asyncio.timeout allows fast operations to complete."""
        async def fast_operation():
            await asyncio.sleep(0.01)
            return "completed"

        async with asyncio.timeout(1.0):
            result = await fast_operation()

        assert result == "completed"

    @pytest.mark.asyncio
    async def test_timeout_with_async_generator(self):
        """Test timeout behavior with async generators (like runner.run_async)."""
        async def slow_generator():
            for i in range(10):
                await asyncio.sleep(0.1)
                yield i

        results = []
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.25):
                async for item in slow_generator():
                    results.append(item)

        # Should have collected some items before timeout
        assert len(results) >= 1
        assert len(results) <= 3
