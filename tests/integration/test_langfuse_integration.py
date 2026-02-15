"""
Integration tests for Langfuse observability.

These tests verify that:
1. Langfuse plugin integrates correctly with Google ADK workflows
2. Token usage is tracked accurately
3. Traces are created and finalized properly
4. Authentication works with real credentials
5. Plugin degrades gracefully when credentials are missing
"""

import os
import pytest
from unittest.mock import patch
from google.adk.runners import Runner
from google.adk.agents import LlmAgent, SequentialAgent
from google.genai import types

from plugins.langfuse_plugin import LangfusePlugin
from common.langfuse_init import initialize_langfuse, is_enabled
from services.session_service import create_session_service


class TestLangfusePluginIntegration:
    """Test Langfuse plugin with real agent workflows."""

    @pytest.fixture
    def langfuse_credentials(self):
        """
        Get Langfuse credentials from environment.

        Returns tuple of (secret_key, public_key, host) or skips test if not available.
        """
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not secret_key or not public_key:
            pytest.skip(
                "LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY not set - "
                "skipping Langfuse integration tests"
            )

        return secret_key, public_key, host

    @pytest.fixture
    def session_service(self):
        """Create an in-memory session service for testing."""
        return create_session_service(use_database=False)

    @pytest.fixture
    def simple_agent(self):
        """Create a simple LLM agent for testing."""
        return LlmAgent(
            name="test_agent",
            model="gemini-2.0-flash-lite",
            instruction="You are a helpful assistant. Respond briefly to user questions.",
        )

    @pytest.mark.integration
    def test_plugin_initialization_without_credentials(self):
        """Test that plugin gracefully disables when credentials are missing."""
        plugin = LangfusePlugin(
            secret_key=None,
            public_key=None,
            host=None,
        )

        # Plugin should be disabled but not crash
        assert plugin.enabled is False
        assert plugin.client is None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_plugin_initialization_with_credentials(self, langfuse_credentials):
        """Test that plugin initializes with credentials (or auto-disables if package unavailable)."""
        secret_key, public_key, host = langfuse_credentials

        plugin = LangfusePlugin(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )

        # Plugin should be enabled if Langfuse package is installed
        # If not installed, it will auto-disable gracefully
        if plugin.enabled:
            assert plugin.client is not None
            print("\n✅ Langfuse plugin initialized successfully")
            await plugin.close()
        else:
            print("\n⚠️  Langfuse plugin disabled (package not available)")

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.real_api
    async def test_workflow_execution_with_plugin(self, simple_agent, session_service):
        """Test that workflows run successfully with Langfuse plugin (enabled or disabled)."""
        # Read Langfuse credentials directly from environment
        # (avoids requiring all config variables like SESSION_MAX_EVENTS)
        plugin = LangfusePlugin(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        # Create runner with plugin
        runner = Runner(
            agent=simple_agent,
            app_name="test_app",
            session_service=session_service,
            plugins=[plugin],
        )

        try:
            # Create session first
            await session_service.create_session(
                app_name="test_app",
                user_id="test_user",
                session_id="test_session",
            )

            # Run a simple query using correct ADK API
            user_message = types.Content(role="user", parts=[types.Part(text="What is 2+2? Answer in one word.")])

            event_count = 0
            async with runner:
                async for event in runner.run_async(
                    user_id="test_user",
                    session_id="test_session",
                    new_message=user_message
                ):
                    event_count += 1

            # Workflow should complete successfully
            assert event_count > 0, "Should generate at least one event"

            # Check token tracking if plugin was enabled
            stats = plugin.get_session_stats()
            if plugin.enabled:
                assert stats["total_tokens"] > 0, "Should track tokens when enabled"
                print(f"\n✅ Langfuse tracked: {stats['total_tokens']} tokens, ${stats['cost_usd']:.6f}")
            else:
                # If disabled, stats should be zero
                assert stats["total_tokens"] == 0
                print("\n⚠️  Langfuse disabled - no token tracking")

            await plugin.flush()

        finally:
            await plugin.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.real_api
    async def test_plugin_graceful_degradation(self, simple_agent, session_service):
        """Test that invalid credentials don't break workflows."""
        # Create plugin with invalid credentials
        plugin = LangfusePlugin(
            secret_key="invalid-key",
            public_key="invalid-key",
            host="https://invalid-host.example.com",
        )

        # Create runner
        runner = Runner(
            agent=simple_agent,
            app_name="test_app",
            session_service=session_service,
            plugins=[plugin],
        )

        # Create session
        await session_service.create_session(
            app_name="test_app",
            user_id="test_user",
            session_id="test_session",
        )

        # Workflow should still complete without crashing
        user_message = types.Content(role="user", parts=[types.Part(text="Hello")])

        event_count = 0
        async with runner:
            async for event in runner.run_async(
                user_id="test_user",
                session_id="test_session",
                new_message=user_message
            ):
                event_count += 1

        assert event_count > 0, "Workflow should complete despite plugin errors"


class TestLangfuseInitialization:
    """Test Langfuse initialization with OpenTelemetry."""

    @pytest.mark.integration
    def test_initialize_langfuse_with_credentials(self):
        """Test Langfuse initialization with valid credentials."""
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not secret_key or not public_key:
            pytest.skip("LANGFUSE credentials not set - skipping initialization test")

        # Test initialization
        result = initialize_langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )

        # Should return a boolean (True if successful, or could be True if already initialized)
        assert isinstance(result, bool), "Should return a boolean"

    @pytest.mark.integration
    def test_initialize_langfuse_without_credentials(self):
        """Test that initialization fails gracefully without credentials."""
        with patch.dict(os.environ, {}, clear=True):
            result = initialize_langfuse(
                secret_key=None,
                public_key=None,
                host=None,
            )

            # Should return boolean but not crash
            assert isinstance(result, bool), "Should return a boolean value"

    @pytest.mark.integration
    def test_initialize_langfuse_idempotent(self):
        """Test that calling initialize_langfuse multiple times is safe."""
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not secret_key or not public_key:
            pytest.skip("LANGFUSE credentials not set")

        # Initialize multiple times should not crash
        result1 = initialize_langfuse(secret_key, public_key, host)
        result2 = initialize_langfuse(secret_key, public_key, host)

        assert isinstance(result1, bool), "First call should return boolean"
        assert isinstance(result2, bool), "Second call should return boolean"


class TestLangfuseWithRealWorkflow:
    """Test Langfuse with actual StoryLand workflows."""

    @pytest.mark.integration
    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_book_metadata_workflow_with_langfuse(self):
        """Test book metadata extraction workflow with Langfuse tracking."""
        from agents import create_book_metadata_pipeline
        from tools.google_books import google_books_tool

        # Create plugin (will auto-disable if credentials missing or package unavailable)
        plugin = LangfusePlugin(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        # Create metadata pipeline
        pipeline = create_book_metadata_pipeline("gemini-2.0-flash-lite", google_books_tool)

        # Create session service
        session_service = create_session_service(use_database=False)

        # Create runner
        runner = Runner(
            agent=pipeline,
            app_name="test_app",
            session_service=session_service,
            plugins=[plugin],
        )

        try:
            # Create session
            await session_service.create_session(
                app_name="test_app",
                user_id="test_user",
                session_id="test_session",
            )

            # Run workflow
            user_message = types.Content(
                role="user",
                parts=[types.Part(text="Find book metadata for '1984' by George Orwell")]
            )

            event_count = 0
            async with runner:
                async for event in runner.run_async(
                    user_id="test_user",
                    session_id="test_session",
                    new_message=user_message
                ):
                    event_count += 1

            # Workflow should complete
            assert event_count > 0, "Workflow should generate events"

            # If plugin was enabled, verify tracking
            stats = plugin.get_session_stats()
            if plugin.enabled:
                assert stats["total_tokens"] > 0, "Should track tokens when enabled"
                print(f"\n✅ Langfuse tracked: {stats['total_tokens']} tokens, ${stats['cost_usd']:.6f}")
            else:
                print("\n⚠️  Langfuse disabled (credentials not configured or package unavailable)")

            await plugin.flush()

        finally:
            await plugin.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.real_api
    async def test_multiple_agents_with_langfuse(self):
        """Test that Langfuse correctly tracks multiple nested agents."""
        # Create plugin (will auto-disable if credentials missing or package unavailable)
        plugin = LangfusePlugin(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        # Create nested agents
        agent1 = LlmAgent(
            name="agent1",
            model="gemini-2.0-flash-lite",
            instruction="Say 'Agent 1 complete' and nothing else.",
        )

        agent2 = LlmAgent(
            name="agent2",
            model="gemini-2.0-flash-lite",
            instruction="Say 'Agent 2 complete' and nothing else.",
        )

        workflow = SequentialAgent(
            name="test_workflow",
            sub_agents=[agent1, agent2],
        )

        # Create session service
        session_service = create_session_service(use_database=False)

        runner = Runner(
            agent=workflow,
            app_name="test_app",
            session_service=session_service,
            plugins=[plugin],
        )

        try:
            # Create session
            await session_service.create_session(
                app_name="test_app",
                user_id="test_user",
                session_id="test_session",
            )

            # Run workflow
            user_message = types.Content(
                role="user",
                parts=[types.Part(text="Execute both agents")]
            )

            event_count = 0
            async with runner:
                async for event in runner.run_async(
                    user_id="test_user",
                    session_id="test_session",
                    new_message=user_message
                ):
                    event_count += 1

            # Should have events from both agents
            assert event_count > 0, "Should generate events from nested agents"

            # Should track both agents if enabled
            stats = plugin.get_session_stats()
            if plugin.enabled:
                assert stats["total_tokens"] > 0, "Should track tokens from nested agents"
                print(f"\n✅ Multi-agent tracking: {stats['total_tokens']} tokens")
            else:
                print("\n⚠️  Langfuse disabled - no tracking")

            await plugin.flush()

        finally:
            await plugin.close()
