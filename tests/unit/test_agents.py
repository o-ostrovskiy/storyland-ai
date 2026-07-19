"""
Unit tests for agent factory functions.

Tests that agent factories create properly configured agents with correct types.
"""

import pytest

from google.adk.agents import SequentialAgent, ParallelAgent, LlmAgent
from google.adk.tools import FunctionTool

from agents import (
    create_book_context_pipeline,
    create_city_pipeline,
    create_landmark_pipeline,
    create_author_pipeline,
    create_local_atmosphere_pipeline,
    create_trip_composer_agent,
    create_region_analyzer_agent,
    create_book_recommendation_pipeline,
    create_book_recommendation_workflow,
    create_discovery_workflow,
    create_composition_workflow,
    create_local_atmosphere_workflow,
)
from agents.prompts import load_prompts, AgentPrompts


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def model_name():
    """Return a valid model name string for agent creation."""
    return "gemini-2.0-flash"


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
# Local Atmosphere Pipeline Tests
# =============================================================================

class TestLocalAtmospherePipeline:
    """Tests for create_local_atmosphere_pipeline."""

    def test_creates_sequential_agent(self, model_name, mock_google_search_tool):
        pipeline = create_local_atmosphere_pipeline(
            model_name,
            mock_google_search_tool,
            location_label="New York, NY",
            radius_km=80,
        )
        assert isinstance(pipeline, SequentialAgent)
        assert pipeline.name == "local_atmosphere_pipeline"

    def test_has_two_sub_agents(self, model_name, mock_google_search_tool):
        pipeline = create_local_atmosphere_pipeline(
            model_name,
            mock_google_search_tool,
            location_label="New York, NY",
            radius_km=80,
        )
        assert len(pipeline.sub_agents) == 2
        assert pipeline.sub_agents[0].name == "local_atmosphere_researcher"
        assert pipeline.sub_agents[1].name == "local_atmosphere_formatter"

    def test_location_baked_into_instructions(self, model_name, mock_google_search_tool):
        pipeline = create_local_atmosphere_pipeline(
            model_name,
            mock_google_search_tool,
            location_label="Salem, MA",
            radius_km=60,
        )
        researcher, formatter = pipeline.sub_agents
        assert "Salem, MA" in researcher.instruction
        assert "60" in researcher.instruction
        assert "Salem, MA" in formatter.instruction
        assert "60" in formatter.instruction


class TestLocalAtmosphereWorkflow:
    """Tests for create_local_atmosphere_workflow."""

    def test_creates_sequential_agent(self, model_name):
        workflow = create_local_atmosphere_workflow(
            model_name,
            book_title="Wuthering Heights",
            author="Emily Brontë",
            location_label="New York, NY",
            radius_km=80,
        )
        assert isinstance(workflow, SequentialAgent)
        assert workflow.name == "local_atmosphere_workflow"

    def test_two_stage_pipeline(self, model_name):
        """book_context_pipeline -> local_atmosphere_pipeline (MYS-436: no
        reader_profile_agent hop -- see TestNoReaderProfileAgent)."""
        workflow = create_local_atmosphere_workflow(
            model_name,
            book_title="X",
            author="Y",
            location_label="Boston, MA",
            radius_km=80,
        )
        names = [a.name for a in workflow.sub_agents]
        assert names == [
            "book_context_pipeline",
            "local_atmosphere_pipeline",
        ]


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
# MYS-436: reader_profile_agent removed -- class-level guard, not a spot check
# =============================================================================

class TestNoReaderProfileAgent:
    """Pins the deletion: reader_profile_agent must not exist anywhere in the
    tree, and no SequentialAgent this module builds may contain an agent
    named "reader_profile_agent" -- a regression here would silently
    reintroduce a per-search LLM call that produces a constant (MYS-436)."""

    def test_factory_function_is_gone(self):
        import agents

        assert not hasattr(agents, "create_reader_profile_agent")

    def test_module_file_is_gone(self):
        import importlib.util

        assert importlib.util.find_spec("agents.reader_profile_agent") is None

    def test_discovery_workflow_has_no_reader_profile_agent(self, model_name):
        workflow = create_discovery_workflow(
            model_name, book_title="1984", author="George Orwell"
        )
        names = [a.name for a in workflow.sub_agents]
        assert "reader_profile_agent" not in names

    def test_local_atmosphere_workflow_has_no_reader_profile_agent(self, model_name):
        workflow = create_local_atmosphere_workflow(
            model_name,
            book_title="X",
            author="Y",
            location_label="Boston, MA",
            radius_km=80,
        )
        names = [a.name for a in workflow.sub_agents]
        assert "reader_profile_agent" not in names


# =============================================================================
# Workflow Orchestrator Tests
# =============================================================================


# =============================================================================
# Main Workflow Tests (Two-Phase Workflow)
# =============================================================================
# Region Analyzer Agent Tests
# =============================================================================

class TestRegionAnalyzerAgent:
    """Tests for create_region_analyzer_agent."""

    def test_creates_llm_agent(self, model_name):
        """Test that region analyzer returns an LlmAgent."""
        agent = create_region_analyzer_agent(model_name)

        assert isinstance(agent, LlmAgent)

    def test_agent_has_correct_name(self, model_name):
        """Test agent has expected name."""
        agent = create_region_analyzer_agent(model_name)

        assert agent.name == "region_analyzer"

    def test_agent_has_output_schema(self, model_name):
        """Test agent has Pydantic output schema configured."""
        agent = create_region_analyzer_agent(model_name)

        assert hasattr(agent, 'output_schema') or hasattr(agent, 'output_key')

    def test_agent_has_output_key(self, model_name):
        """Test agent stores output in region_analysis key."""
        agent = create_region_analyzer_agent(model_name)

        assert agent.output_key == "region_analysis"


# =============================================================================
# Discovery Workflow Tests (Three-Phase Workflow)
# =============================================================================

class TestDiscoveryWorkflow:
    """Tests for create_discovery_workflow."""

    def test_creates_sequential_agent(self, model_name):
        """Test that discovery workflow returns a SequentialAgent."""
        workflow = create_discovery_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        assert isinstance(workflow, SequentialAgent)

    def test_workflow_has_correct_name(self, model_name):
        """Test discovery workflow has expected name."""
        workflow = create_discovery_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        assert workflow.name == "discovery_workflow"

    def test_workflow_has_three_stages(self, model_name):
        """Test discovery workflow has 3 stages (MYS-436: reader_profile_agent
        removed -- was a dead hot-path LLM call producing a constant)."""
        workflow = create_discovery_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        # book_context, parallel_discovery, region_analyzer
        assert len(workflow.sub_agents) == 3

    def test_workflow_ends_with_region_analyzer(self, model_name):
        """Test discovery workflow ends with region_analyzer."""
        workflow = create_discovery_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        stage_names = [agent.name for agent in workflow.sub_agents]
        assert stage_names[-1] == "region_analyzer"

    def test_workflow_stages_order(self, model_name):
        """Test discovery workflow stages are in correct order."""
        workflow = create_discovery_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        stage_names = [agent.name for agent in workflow.sub_agents]

        assert stage_names[0] == "book_context_pipeline"
        assert stage_names[1] == "parallel_discovery"
        assert stage_names[2] == "region_analyzer"

    def test_workflow_contains_parallel_agent(self, model_name):
        """Test discovery workflow contains a ParallelAgent for discovery."""
        workflow = create_discovery_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

        parallel_agents = [
            agent for agent in workflow.sub_agents
            if isinstance(agent, ParallelAgent)
        ]
        assert len(parallel_agents) == 1
        assert parallel_agents[0].name == "parallel_discovery"


# =============================================================================
# Composition Workflow Tests (Three-Phase Workflow)
# =============================================================================

class TestCompositionWorkflow:
    """Tests for create_composition_workflow."""

    def test_creates_sequential_agent(self, model_name):
        """Test that composition workflow returns a SequentialAgent."""
        workflow = create_composition_workflow(model_name)

        assert isinstance(workflow, SequentialAgent)

    def test_workflow_has_correct_name(self, model_name):
        """Test composition workflow has expected name."""
        workflow = create_composition_workflow(model_name)

        assert workflow.name == "composition_workflow"

    def test_workflow_has_one_stage(self, model_name):
        """Test composition workflow has 1 stage (trip_composer only)."""
        workflow = create_composition_workflow(model_name)

        assert len(workflow.sub_agents) == 1

    def test_workflow_contains_trip_composer(self, model_name):
        """Test composition workflow contains trip_composer agent."""
        workflow = create_composition_workflow(model_name)

        assert workflow.sub_agents[0].name == "trip_composer"


# =============================================================================
# BookContext Pipeline Dynamic Instruction Tests
# =============================================================================

class TestBookContextDynamicInstruction:
    """Verify book_context_pipeline handles both known and unknown title/author."""

    def test_known_title_baked_into_instruction(self, model_name, mock_google_search_tool):
        """When title/author are provided, they should appear in the instruction."""
        pipeline = create_book_context_pipeline(
            model_name, mock_google_search_tool,
            book_title="1984", author="George Orwell"
        )
        researcher = pipeline.sub_agents[0]
        assert '"1984"' in researcher.instruction
        assert "George Orwell" in researcher.instruction

    def test_no_title_uses_dynamic_reference(self, model_name, mock_google_search_tool):
        """When no title/author provided, instruction should reference conversation history."""
        pipeline = create_book_context_pipeline(
            model_name, mock_google_search_tool
        )
        researcher = pipeline.sub_agents[0]
        assert "book_metadata" in researcher.instruction
        assert "[from conversation]" not in researcher.instruction

    def test_title_starting_with_bracket_is_treated_as_real_title(
        self, model_name, mock_google_search_tool
    ):
        """A real title like '[Pygmalion]' should still be baked into the instruction."""
        pipeline = create_book_context_pipeline(
            model_name,
            mock_google_search_tool,
            book_title="[Pygmalion]",
            author="George Bernard Shaw",
        )
        researcher = pipeline.sub_agents[0]
        assert '"[Pygmalion]" by George Bernard Shaw' in researcher.instruction


# =============================================================================
# Prompt Loader Tests
# =============================================================================

class TestLoadPrompts:
    """Tests for the versioned prompt loader."""

    def test_load_prompts_default_returns_agent_prompts(self):
        """load_prompts() with no args returns an AgentPrompts instance."""
        prompts = load_prompts()
        assert isinstance(prompts, AgentPrompts)
        assert prompts.trip_composer  # non-empty string

    def test_load_prompts_caches_same_object(self):
        """Repeated calls for the same version return the identical cached object."""
        assert load_prompts("v2") is load_prompts("v2")

    def test_load_prompts_missing_version_raises(self):
        """Requesting a non-existent version raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="v99"):
            load_prompts("v99")

    def test_book_recommendation_prompts_exist(self):
        """AgentPrompts includes both book_recommendation prompt fields."""
        prompts = load_prompts()
        for field in ("book_recommendation_researcher", "book_recommendation_formatter"):
            assert hasattr(prompts, field)
            template = getattr(prompts, field)
            assert "{book_title}" in template
            assert "{destinations}" in template
            assert "{themes}" in template


# =============================================================================
# BookRecommendationPipeline Tests
# =============================================================================


class TestBookRecommendationPipeline:
    """Tests for create_book_recommendation_pipeline."""

    def test_creates_sequential_agent(self, model_name, mock_google_search_tool):
        pipeline = create_book_recommendation_pipeline(
            model_name,
            mock_google_search_tool,
            book_title="The Da Vinci Code",
            author="Dan Brown",
            destinations="Paris, London",
            themes="mystery, art, religion",
        )
        assert isinstance(pipeline, SequentialAgent)
        assert pipeline.name == "book_recommendation_pipeline"

    def test_pipeline_has_researcher_and_formatter(self, model_name, mock_google_search_tool):
        pipeline = create_book_recommendation_pipeline(
            model_name,
            mock_google_search_tool,
            book_title="1984",
            author="George Orwell",
            destinations="London",
            themes="dystopia, surveillance",
        )
        assert len(pipeline.sub_agents) == 2
        researcher, formatter = pipeline.sub_agents
        assert isinstance(researcher, LlmAgent)
        assert isinstance(formatter, LlmAgent)
        assert researcher.name == "book_recommendation_researcher"
        assert formatter.name == "book_recommendation_formatter"

    def test_researcher_uses_search_tool_only(self, model_name, mock_google_search_tool):
        """ADK forbids tools + output_schema. Researcher gets the tool, formatter gets the schema."""
        pipeline = create_book_recommendation_pipeline(
            model_name,
            mock_google_search_tool,
            book_title="1984",
            author="George Orwell",
            destinations="London",
            themes="dystopia",
        )
        researcher, formatter = pipeline.sub_agents
        assert researcher.tools and len(researcher.tools) == 1
        assert researcher.output_schema is None
        assert not formatter.tools

    def test_formatter_has_output_schema(self, model_name, mock_google_search_tool):
        from models.book import BookRecommendationsResult
        pipeline = create_book_recommendation_pipeline(
            model_name,
            mock_google_search_tool,
            book_title="1984",
            author="George Orwell",
            destinations="London",
            themes="dystopia, surveillance",
        )
        formatter = pipeline.sub_agents[1]
        assert formatter.output_schema is BookRecommendationsResult
        assert formatter.output_key == "last_book_recommendations"

    def test_prompt_interpolation(self, model_name, mock_google_search_tool):
        pipeline = create_book_recommendation_pipeline(
            model_name,
            mock_google_search_tool,
            book_title="Middlemarch",
            author="George Eliot",
            destinations="Coventry, London",
            themes="social reform, marriage",
        )
        researcher, formatter = pipeline.sub_agents
        for agent in (researcher, formatter):
            assert "Middlemarch" in agent.instruction
            assert "George Eliot" in agent.instruction
            assert "Coventry, London" in agent.instruction
            assert "social reform, marriage" in agent.instruction


class TestBookRecommendationWorkflow:
    """Tests for create_book_recommendation_workflow."""

    def test_creates_sequential_agent(self, model_name, mock_google_search_tool):
        wf = create_book_recommendation_workflow(
            model_name,
            mock_google_search_tool,
            book_title="1984",
            author="George Orwell",
            destinations="London",
            themes="dystopia",
        )
        assert isinstance(wf, SequentialAgent)

    def test_workflow_name(self, model_name, mock_google_search_tool):
        wf = create_book_recommendation_workflow(
            model_name,
            mock_google_search_tool,
            book_title="1984",
            author="George Orwell",
            destinations="London",
            themes="dystopia",
        )
        assert wf.name == "book_recommendation_workflow"

    def test_workflow_wraps_pipeline(self, model_name, mock_google_search_tool):
        wf = create_book_recommendation_workflow(
            model_name,
            mock_google_search_tool,
            book_title="1984",
            author="George Orwell",
            destinations="London",
            themes="dystopia",
        )
        assert len(wf.sub_agents) == 1
        pipeline = wf.sub_agents[0]
        assert isinstance(pipeline, SequentialAgent)
        assert pipeline.name == "book_recommendation_pipeline"
