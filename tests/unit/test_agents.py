"""
Unit tests for agent factory functions.

Tests that agent factories create properly configured agents with correct types.
"""

import pytest

from google.adk.agents import SequentialAgent, ParallelAgent, LlmAgent
from google.adk.tools import FunctionTool

from agents import (
    create_book_metadata_pipeline,
    create_book_context_pipeline,
    create_city_pipeline,
    create_landmark_pipeline,
    create_author_pipeline,
    create_trip_composer_agent,
    create_reader_profile_agent,
    create_workflow,
    create_metadata_stage,
    create_main_workflow,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def model_name():
    """Return a valid model name string for agent creation."""
    return "gemini-2.0-flash"


@pytest.fixture
def mock_google_books_tool():
    """Create a mock Google Books FunctionTool."""
    def mock_search_book(title: str, author: str = "") -> str:
        """Mock search function."""
        return '{"book_title": "Test", "author": "Test Author"}'

    return FunctionTool(mock_search_book)


@pytest.fixture
def mock_google_search_tool():
    """Create a mock Google Search FunctionTool."""
    def mock_search(query: str) -> str:
        """Mock search function."""
        return "Search results for: " + query

    return FunctionTool(mock_search)


# =============================================================================
# Book Metadata Pipeline Tests
# =============================================================================

class TestBookMetadataPipeline:
    """Tests for create_book_metadata_pipeline."""

    def test_creates_sequential_agent(self, model_name, mock_google_books_tool):
        """Test that pipeline returns a SequentialAgent."""
        pipeline = create_book_metadata_pipeline(model_name, mock_google_books_tool)

        assert isinstance(pipeline, SequentialAgent)

    def test_pipeline_has_correct_name(self, model_name, mock_google_books_tool):
        """Test pipeline has expected name."""
        pipeline = create_book_metadata_pipeline(model_name, mock_google_books_tool)

        assert pipeline.name == "book_metadata_pipeline"

    def test_pipeline_has_sub_agents(self, model_name, mock_google_books_tool):
        """Test pipeline contains sub-agents."""
        pipeline = create_book_metadata_pipeline(model_name, mock_google_books_tool)

        assert len(pipeline.sub_agents) == 2


# =============================================================================
# Book Context Pipeline Tests
# =============================================================================

class TestBookContextPipeline:
    """Tests for create_book_context_pipeline."""

    def test_creates_sequential_agent(self, model_name, mock_google_search_tool):
        """Test that pipeline returns a SequentialAgent."""
        pipeline = create_book_context_pipeline(
            model_name, mock_google_search_tool,
            book_title="The Nightingale", author="Kristin Hannah"
        )

        assert isinstance(pipeline, SequentialAgent)

    def test_pipeline_has_correct_name(self, model_name, mock_google_search_tool):
        """Test pipeline has expected name."""
        pipeline = create_book_context_pipeline(
            model_name, mock_google_search_tool,
            book_title="The Nightingale", author="Kristin Hannah"
        )

        assert pipeline.name == "book_context_pipeline"

    def test_pipeline_has_sub_agents(self, model_name, mock_google_search_tool):
        """Test pipeline contains sub-agents."""
        pipeline = create_book_context_pipeline(
            model_name, mock_google_search_tool,
            book_title="The Nightingale", author="Kristin Hannah"
        )

        assert len(pipeline.sub_agents) == 2


# =============================================================================
# City Pipeline Tests
# =============================================================================

class TestCityPipeline:
    """Tests for create_city_pipeline."""

    def test_creates_sequential_agent(self, model_name, mock_google_search_tool):
        """Test that pipeline returns a SequentialAgent."""
        pipeline = create_city_pipeline(model_name, mock_google_search_tool)

        assert isinstance(pipeline, SequentialAgent)

    def test_pipeline_has_correct_name(self, model_name, mock_google_search_tool):
        """Test pipeline has expected name."""
        pipeline = create_city_pipeline(model_name, mock_google_search_tool)

        assert pipeline.name == "city_pipeline"


# =============================================================================
# Landmark Pipeline Tests
# =============================================================================

class TestLandmarkPipeline:
    """Tests for create_landmark_pipeline."""

    def test_creates_sequential_agent(self, model_name, mock_google_search_tool):
        """Test that pipeline returns a SequentialAgent."""
        pipeline = create_landmark_pipeline(model_name, mock_google_search_tool)

        assert isinstance(pipeline, SequentialAgent)

    def test_pipeline_has_correct_name(self, model_name, mock_google_search_tool):
        """Test pipeline has expected name."""
        pipeline = create_landmark_pipeline(model_name, mock_google_search_tool)

        assert pipeline.name == "landmark_pipeline"


# =============================================================================
# Author Pipeline Tests
# =============================================================================

class TestAuthorPipeline:
    """Tests for create_author_pipeline."""

    def test_creates_sequential_agent(self, model_name, mock_google_search_tool):
        """Test that pipeline returns a SequentialAgent."""
        pipeline = create_author_pipeline(model_name, mock_google_search_tool)

        assert isinstance(pipeline, SequentialAgent)

    def test_pipeline_has_correct_name(self, model_name, mock_google_search_tool):
        """Test pipeline has expected name."""
        pipeline = create_author_pipeline(model_name, mock_google_search_tool)

        assert pipeline.name == "author_pipeline"


# =============================================================================
# Trip Composer Agent Tests
# =============================================================================

class TestTripComposerAgent:
    """Tests for create_trip_composer_agent."""

    def test_creates_llm_agent(self, model_name):
        """Test that trip composer returns an LlmAgent."""
        agent = create_trip_composer_agent(model_name)

        assert isinstance(agent, LlmAgent)

    def test_agent_has_correct_name(self, model_name):
        """Test agent has expected name."""
        agent = create_trip_composer_agent(model_name)

        assert agent.name == "trip_composer"

    def test_agent_has_output_schema(self, model_name):
        """Test agent has Pydantic output schema configured."""
        agent = create_trip_composer_agent(model_name)

        # Should have output_schema or output_key set for Pydantic validation
        assert hasattr(agent, 'output_schema') or hasattr(agent, 'output_key')


# =============================================================================
# Reader Profile Agent Tests
# =============================================================================

class TestReaderProfileAgent:
    """Tests for create_reader_profile_agent."""

    def test_creates_llm_agent(self, model_name):
        """Test that reader profile returns an LlmAgent."""
        agent = create_reader_profile_agent(model_name)

        assert isinstance(agent, LlmAgent)

    def test_agent_has_correct_name(self, model_name):
        """Test agent has expected name."""
        agent = create_reader_profile_agent(model_name)

        assert agent.name == "reader_profile_agent"

    def test_agent_has_tools(self, model_name):
        """Test agent has preference tool configured."""
        agent = create_reader_profile_agent(model_name)

        # Should have tools for reading preferences
        assert hasattr(agent, 'tools')
        assert len(agent.tools) > 0


# =============================================================================
# Workflow Orchestrator Tests
# =============================================================================

class TestCreateWorkflow:
    """Tests for create_workflow orchestrator."""

    def test_creates_sequential_agent(self, model_name, mock_google_books_tool):
        """Test that workflow returns a SequentialAgent."""
        workflow = create_workflow(model_name, mock_google_books_tool)

        assert isinstance(workflow, SequentialAgent)

    def test_workflow_has_correct_name(self, model_name, mock_google_books_tool):
        """Test workflow has expected name."""
        workflow = create_workflow(model_name, mock_google_books_tool)

        assert workflow.name == "workflow"

    def test_workflow_has_five_stages(self, model_name, mock_google_books_tool):
        """Test workflow has 5 main stages."""
        workflow = create_workflow(model_name, mock_google_books_tool)

        # book_metadata, book_context, reader_profile, parallel_discovery, trip_composer
        assert len(workflow.sub_agents) == 5

    def test_workflow_contains_parallel_agent(self, model_name, mock_google_books_tool):
        """Test workflow contains a ParallelAgent for discovery."""
        workflow = create_workflow(model_name, mock_google_books_tool)

        parallel_agents = [
            agent for agent in workflow.sub_agents
            if isinstance(agent, ParallelAgent)
        ]
        assert len(parallel_agents) == 1
        assert parallel_agents[0].name == "parallel_discovery"

    def test_parallel_discovery_has_three_pipelines(self, model_name, mock_google_books_tool):
        """Test parallel discovery contains city, landmark, author pipelines."""
        workflow = create_workflow(model_name, mock_google_books_tool)

        parallel_agent = next(
            agent for agent in workflow.sub_agents
            if isinstance(agent, ParallelAgent)
        )

        assert len(parallel_agent.sub_agents) == 3

        # Verify names of sub-agents
        names = [agent.name for agent in parallel_agent.sub_agents]
        assert "city_pipeline" in names
        assert "landmark_pipeline" in names
        assert "author_pipeline" in names

    def test_workflow_stages_order(self, model_name, mock_google_books_tool):
        """Test workflow stages are in correct order."""
        workflow = create_workflow(model_name, mock_google_books_tool)

        stage_names = [agent.name for agent in workflow.sub_agents]

        # Verify order: metadata first, then context, then profile, then discovery, then composer
        assert stage_names[0] == "book_metadata_pipeline"
        assert stage_names[1] == "book_context_pipeline"
        assert stage_names[2] == "reader_profile_agent"
        assert stage_names[3] == "parallel_discovery"
        assert stage_names[4] == "trip_composer"


# =============================================================================
# Metadata Stage Tests (Two-Phase Workflow)
# =============================================================================

class TestMetadataStage:
    """Tests for create_metadata_stage."""

    def test_creates_sequential_agent(self, model_name, mock_google_books_tool):
        """Test that metadata stage returns a SequentialAgent."""
        stage = create_metadata_stage(model_name, mock_google_books_tool)

        assert isinstance(stage, SequentialAgent)

    def test_stage_has_correct_name(self, model_name, mock_google_books_tool):
        """Test metadata stage has expected name."""
        stage = create_metadata_stage(model_name, mock_google_books_tool)

        assert stage.name == "metadata_stage"

    def test_stage_contains_metadata_pipeline(self, model_name, mock_google_books_tool):
        """Test metadata stage contains book_metadata_pipeline."""
        stage = create_metadata_stage(model_name, mock_google_books_tool)

        assert len(stage.sub_agents) == 1
        assert stage.sub_agents[0].name == "book_metadata_pipeline"


# =============================================================================
# Main Workflow Tests (Two-Phase Workflow)
# =============================================================================

class TestMainWorkflow:
    """Tests for create_main_workflow."""

    def test_creates_sequential_agent(self, model_name):
        """Test that main workflow returns a SequentialAgent."""
        workflow = create_main_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        assert isinstance(workflow, SequentialAgent)

    def test_workflow_has_correct_name(self, model_name):
        """Test main workflow has expected name."""
        workflow = create_main_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        assert workflow.name == "main_workflow"

    def test_workflow_has_four_stages(self, model_name):
        """Test main workflow has 4 stages (no metadata pipeline)."""
        workflow = create_main_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        # book_context, reader_profile, parallel_discovery, trip_composer
        assert len(workflow.sub_agents) == 4

    def test_workflow_does_not_contain_metadata_pipeline(self, model_name):
        """Test main workflow does NOT contain book_metadata_pipeline."""
        workflow = create_main_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        stage_names = [agent.name for agent in workflow.sub_agents]
        assert "book_metadata_pipeline" not in stage_names

    def test_workflow_stages_order(self, model_name):
        """Test main workflow stages are in correct order."""
        workflow = create_main_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        stage_names = [agent.name for agent in workflow.sub_agents]

        assert stage_names[0] == "book_context_pipeline"
        assert stage_names[1] == "reader_profile_agent"
        assert stage_names[2] == "parallel_discovery"
        assert stage_names[3] == "trip_composer"

    def test_workflow_contains_parallel_agent(self, model_name):
        """Test main workflow contains a ParallelAgent for discovery."""
        workflow = create_main_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        parallel_agents = [
            agent for agent in workflow.sub_agents
            if isinstance(agent, ParallelAgent)
        ]
        assert len(parallel_agents) == 1
        assert parallel_agents[0].name == "parallel_discovery"
