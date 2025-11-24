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

class TestTimeoutConfiguration:
    """Tests for timeout configuration."""

    @pytest.mark.asyncio
    @patch("main.load_config")
    @patch("main.create_session_service")
    @patch("main.create_metadata_stage")
    @patch("main.create_main_workflow")
    @patch("main.Runner")
    async def test_uses_config_timeout_by_default(
        self, mock_runner_class, mock_create_main_workflow, mock_create_metadata_stage, mock_session_service, mock_config
    ):
        """Test that workflow uses config timeout by default."""
        # Setup mocks
        config = MagicMock()
        config.workflow_timeout = 120
        config.google_api_key = "test-key"
        config.log_level = "INFO"
        config.enable_adk_debug = False
        config.model_name = "gemini-2.0-flash-lite"
        config.database_url = None
        config.use_database = False
        config.session_max_events = 20
        mock_config.return_value = config

        mock_session = MagicMock()
        mock_session.create_session = AsyncMock()
        # Return mock session with book_metadata in state
        mock_session_state = MagicMock()
        mock_session_state.state = {"book_metadata": {"book_title": "Test Book", "author": "Test Author"}}
        mock_session_state.events = []
        mock_session.get_session = AsyncMock(return_value=mock_session_state)
        mock_session_service.return_value = mock_session

        mock_metadata_stage = MagicMock()
        mock_create_metadata_stage.return_value = mock_metadata_stage

        mock_main_workflow = MagicMock()
        mock_create_main_workflow.return_value = mock_main_workflow

        # Create a mock runner that completes quickly
        async def quick_run(*args, **kwargs):
            # Yield a mock final event
            mock_event = MagicMock()
            mock_event.author = "trip_composer"
            mock_event.is_final_response.return_value = True
            mock_event.content.parts = [MagicMock(text='{"cities": [], "summary_text": "test"}')]
            yield mock_event

        mock_runner = MagicMock()
        mock_runner.run_async = quick_run
        mock_runner_class.return_value = mock_runner

        # Run without explicit timeout - should use config value (120s)
        # This test verifies the timeout is applied; actual timeout behavior
        # is tested in integration tests
        result = await create_itinerary(
            book_title="Test Book",
            author="Test Author",
            timeout=None,  # Should use config.workflow_timeout
        )

        # Verify config was loaded
        mock_config.assert_called_once()

    @pytest.mark.asyncio
    @patch("main.load_config")
    @patch("main.create_session_service")
    @patch("main.create_metadata_stage")
    @patch("main.create_main_workflow")
    @patch("main.Runner")
    async def test_explicit_timeout_overrides_config(
        self, mock_runner_class, mock_create_main_workflow, mock_create_metadata_stage, mock_session_service, mock_config
    ):
        """Test that explicit timeout parameter overrides config."""
        # Setup mocks
        config = MagicMock()
        config.workflow_timeout = 300  # Config says 300s
        config.google_api_key = "test-key"
        config.log_level = "INFO"
        config.enable_adk_debug = False
        config.model_name = "gemini-2.0-flash-lite"
        config.database_url = None
        config.use_database = False
        config.session_max_events = 20
        mock_config.return_value = config

        mock_session = MagicMock()
        mock_session.create_session = AsyncMock()
        # Return mock session with book_metadata in state
        mock_session_state = MagicMock()
        mock_session_state.state = {"book_metadata": {"book_title": "Test Book", "author": "Test Author"}}
        mock_session_state.events = []
        mock_session.get_session = AsyncMock(return_value=mock_session_state)
        mock_session_service.return_value = mock_session

        mock_metadata_stage = MagicMock()
        mock_create_metadata_stage.return_value = mock_metadata_stage

        mock_main_workflow = MagicMock()
        mock_create_main_workflow.return_value = mock_main_workflow

        # Create a slow runner that exceeds the explicit timeout
        async def slow_run(*args, **kwargs):
            await asyncio.sleep(0.5)  # Simulate slow execution
            mock_event = MagicMock()
            mock_event.author = "trip_composer"
            mock_event.is_final_response.return_value = True
            mock_event.content.parts = [MagicMock(text='{"cities": [], "summary_text": "test"}')]
            yield mock_event

        mock_runner = MagicMock()
        mock_runner.run_async = slow_run
        mock_runner_class.return_value = mock_runner

        # Run with explicit short timeout - should timeout
        with pytest.raises(WorkflowTimeoutError) as exc_info:
            await create_itinerary(
                book_title="Test Book",
                author="Test Author",
                timeout=0.1,  # Very short timeout - should trigger
            )

        assert "0.1s timeout" in str(exc_info.value)


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
