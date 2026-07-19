"""
Unit tests for agent factory functions and workflow graphs.

Agent factories return (researcher, formatter) LlmAgent pairs; workflow
factories return google.adk.workflow.Workflow graphs (ADK 2 graph rewrite,
ADR #24 — no Sequential/ParallelAgent templates). Structural assertions here
are graph-native: node names, edge pairs, join gating.
"""

import pytest

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.workflow import JoinNode, Workflow

from agents import (
    create_book_context_agents,
    create_city_agents,
    create_landmark_agents,
    create_author_agents,
    create_local_atmosphere_agents,
    create_trip_composer_agent,
    create_region_analyzer_agent,
    create_book_recommendation_agents,
    create_book_recommendation_workflow,
    create_book_to_place_discovery_workflow,
    create_book_to_place_composition_workflow,
    create_local_atmosphere_workflow,
)
from agents.prompts import load_prompts, AgentPrompts

START_NAME = "__START__"


# =============================================================================
# Fixtures / helpers
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


def edge_pairs(workflow: Workflow) -> set[tuple[str, str]]:
    """The workflow's edges as (from_name, to_name) pairs."""
    return {(e.from_node.name, e.to_node.name) for e in workflow.graph.edges}


def node_names(workflow: Workflow) -> set[str]:
    return {n.name for n in workflow.graph.nodes}


def successors(workflow: Workflow, name: str) -> set[str]:
    return {t for f, t in edge_pairs(workflow) if f == name}


def predecessors(workflow: Workflow, name: str) -> set[str]:
    return {f for f, t in edge_pairs(workflow) if t == name}


# =============================================================================
# Agent-pair factory tests
# =============================================================================

class TestBookContextAgents:
    """Tests for create_book_context_agents."""

    def test_returns_researcher_formatter_pair(self, model_name, mock_google_search_tool):
        researcher, formatter = create_book_context_agents(
            model_name, mock_google_search_tool,
            book_title="The Nightingale", author="Kristin Hannah"
        )
        assert researcher.name == "book_context_researcher"
        assert formatter.name == "book_context_formatter"

    def test_researcher_has_tool_formatter_has_schema(self, model_name, mock_google_search_tool):
        researcher, formatter = create_book_context_agents(
            model_name, mock_google_search_tool,
            book_title="The Nightingale", author="Kristin Hannah"
        )
        assert researcher.tools
        assert researcher.output_schema is None
        assert not formatter.tools
        assert formatter.output_schema is not None


class TestCityAgents:
    def test_researcher_carries_book_facts(self, model_name, mock_google_search_tool):
        """Graph-scoped context (ADR #24): the branch researchers never see the
        discovery user prompt, so title/author/vibe/taste must be baked into
        their instructions (Codex P1 on the first PR-3 head)."""
        researcher, _ = create_city_agents(
            model_name,
            mock_google_search_tool,
            book_title="Persuasion",
            author="Jane Austen",
            vibe="melancholic coastal",
            taste_context={"titles": ["Rebecca"], "moods": ["windswept"]},
        )
        for needle in ("Persuasion", "Jane Austen", "melancholic coastal", "Rebecca", "windswept"):
            assert needle in researcher.instruction, needle

    def test_pair_names(self, model_name, mock_google_search_tool):
        researcher, formatter = create_city_agents(model_name, mock_google_search_tool, book_title="1984", author="George Orwell")
        assert researcher.name == "city_researcher"
        assert formatter.name == "city_formatter"

    def test_formatter_output_key(self, model_name, mock_google_search_tool):
        _, formatter = create_city_agents(model_name, mock_google_search_tool, book_title="1984", author="George Orwell")
        assert formatter.output_key == "city_discovery"


class TestLandmarkAgents:
    def test_pair_names(self, model_name, mock_google_search_tool):
        researcher, formatter = create_landmark_agents(model_name, mock_google_search_tool, book_title="1984", author="George Orwell")
        assert researcher.name == "landmark_researcher"
        assert formatter.name == "landmark_formatter"

    def test_formatter_output_key(self, model_name, mock_google_search_tool):
        _, formatter = create_landmark_agents(model_name, mock_google_search_tool, book_title="1984", author="George Orwell")
        assert formatter.output_key == "landmark_discovery"


class TestAuthorAgents:
    def test_pair_names(self, model_name, mock_google_search_tool):
        researcher, formatter = create_author_agents(model_name, mock_google_search_tool, book_title="1984", author="George Orwell")
        assert researcher.name == "author_researcher"
        assert formatter.name == "author_formatter"

    def test_formatter_output_key(self, model_name, mock_google_search_tool):
        _, formatter = create_author_agents(model_name, mock_google_search_tool, book_title="1984", author="George Orwell")
        assert formatter.output_key == "author_sites"


class TestLocalAtmosphereAgents:
    def test_pair_names(self, model_name, mock_google_search_tool):
        researcher, formatter = create_local_atmosphere_agents(
            model_name,
            mock_google_search_tool,
            location_label="New York, NY",
            radius_km=80,
        )
        assert researcher.name == "local_atmosphere_researcher"
        assert formatter.name == "local_atmosphere_formatter"

    def test_location_baked_into_instructions(self, model_name, mock_google_search_tool):
        researcher, formatter = create_local_atmosphere_agents(
            model_name,
            mock_google_search_tool,
            location_label="Salem, MA",
            radius_km=60,
        )
        assert "Salem, MA" in researcher.instruction
        assert "60" in researcher.instruction
        assert "Salem, MA" in formatter.instruction
        assert "60" in formatter.instruction

    def test_preferences_baked_into_both_instructions(self, model_name, mock_google_search_tool):
        """Codex P2 (2026-07-19 21:10): preferences appended to the initial
        user prompt reach only the FIRST graph node; the formatter is 3-4
        nodes downstream. Both local agents must carry them in-instruction."""
        researcher, formatter = create_local_atmosphere_agents(
            model_name,
            mock_google_search_tool,
            location_label="Salem, MA",
            radius_km=60,
            preferences={"pace": "la-prefs-marker", "budget": "low"},
        )
        assert "la-prefs-marker" in researcher.instruction
        assert "la-prefs-marker" in formatter.instruction

    def test_no_preferences_leaves_instructions_unchanged(self, model_name, mock_google_search_tool):
        with_none = create_local_atmosphere_agents(
            model_name, mock_google_search_tool,
            location_label="Salem, MA", radius_km=60, preferences=None,
        )
        without = create_local_atmosphere_agents(
            model_name, mock_google_search_tool,
            location_label="Salem, MA", radius_km=60,
        )
        assert with_none[0].instruction == without[0].instruction
        assert with_none[1].instruction == without[1].instruction


class TestLocalAtmosphereWorkflow:
    """Tests for create_local_atmosphere_workflow."""

    def test_creates_workflow_graph(self, model_name):
        workflow = create_local_atmosphere_workflow(
            model_name,
            book_title="Wuthering Heights",
            author="Emily Brontë",
            location_label="New York, NY",
            radius_km=80,
        )
        assert isinstance(workflow, Workflow)
        assert workflow.name == "local_atmosphere_workflow"

    def test_linear_chain(self, model_name):
        """START → book_context researcher→formatter → local researcher→formatter.

        MYS-436: no reader_profile hop — see TestNoReaderProfileAgent."""
        workflow = create_local_atmosphere_workflow(
            model_name,
            book_title="X",
            author="Y",
            location_label="Boston, MA",
            radius_km=80,
        )
        assert edge_pairs(workflow) == {
            (START_NAME, "book_context_researcher"),
            ("book_context_researcher", "book_context_formatter"),
            ("book_context_formatter", "local_atmosphere_researcher"),
            ("local_atmosphere_researcher", "local_atmosphere_formatter"),
        }


# =============================================================================
# Trip Composer Agent Tests
# =============================================================================

class TestTripComposerAgent:
    """Tests for create_trip_composer_agent."""

    def test_creates_llm_agent(self, model_name):
        agent = create_trip_composer_agent(model_name)
        assert isinstance(agent, LlmAgent)

    def test_agent_has_correct_name(self, model_name):
        agent = create_trip_composer_agent(model_name)
        assert agent.name == "trip_composer"

    def test_agent_has_output_schema(self, model_name):
        agent = create_trip_composer_agent(model_name)
        assert hasattr(agent, 'output_schema') or hasattr(agent, 'output_key')


# =============================================================================
# MYS-436: reader_profile_agent removed -- class-level guard, not a spot check
# =============================================================================

class TestNoReaderProfileAgent:
    """Pins the deletion: reader_profile_agent must not exist anywhere in the
    tree, and no workflow graph this module builds may contain a node named
    "reader_profile_agent" -- a regression here would silently reintroduce a
    per-search LLM call that produces a constant (MYS-436)."""

    def test_factory_function_is_gone(self):
        import agents

        assert not hasattr(agents, "create_reader_profile_agent")

    def test_module_file_is_gone(self):
        import importlib.util

        assert importlib.util.find_spec("agents.reader_profile_agent") is None

    def test_discovery_workflow_has_no_reader_profile_agent(self, model_name):
        workflow = create_book_to_place_discovery_workflow(
            model_name, book_title="1984", author="George Orwell"
        )
        assert "reader_profile_agent" not in node_names(workflow)

    def test_local_atmosphere_workflow_has_no_reader_profile_agent(self, model_name):
        workflow = create_local_atmosphere_workflow(
            model_name,
            book_title="X",
            author="Y",
            location_label="Boston, MA",
            radius_km=80,
        )
        assert "reader_profile_agent" not in node_names(workflow)


# =============================================================================
# Region Analyzer Agent Tests
# =============================================================================

class TestRegionAnalyzerAgent:
    """Tests for create_region_analyzer_agent."""

    def test_creates_llm_agent(self, model_name):
        agent = create_region_analyzer_agent(model_name)
        assert isinstance(agent, LlmAgent)

    def test_agent_has_correct_name(self, model_name):
        agent = create_region_analyzer_agent(model_name)
        assert agent.name == "region_analyzer"

    def test_agent_has_output_schema(self, model_name):
        agent = create_region_analyzer_agent(model_name)
        assert hasattr(agent, 'output_schema') or hasattr(agent, 'output_key')

    def test_agent_has_output_key(self, model_name):
        agent = create_region_analyzer_agent(model_name)
        assert agent.output_key == "region_analysis"


# =============================================================================
# Book→Place Discovery Workflow (phase 1 of the primary flow)
# =============================================================================

class TestBookToPlaceDiscoveryWorkflow:
    """Tests for create_book_to_place_discovery_workflow's graph shape."""

    def _workflow(self, model_name):
        return create_book_to_place_discovery_workflow(
            model_name, book_title="1984", author="George Orwell"
        )

    def test_creates_workflow_graph(self, model_name):
        workflow = self._workflow(model_name)
        assert isinstance(workflow, Workflow)
        assert workflow.name == "book_to_place_discovery"

    def test_context_chain_precedes_fanout(self, model_name):
        """START → context researcher → formatter → 3-way fan-out."""
        workflow = self._workflow(model_name)
        assert successors(workflow, START_NAME) == {"book_context_researcher"}
        assert successors(workflow, "book_context_researcher") == {
            "book_context_formatter"
        }
        assert successors(workflow, "book_context_formatter") == {
            "city_researcher",
            "landmark_researcher",
            "author_researcher",
        }

    def test_each_branch_is_researcher_then_formatter(self, model_name):
        workflow = self._workflow(model_name)
        for key in ("city", "landmark", "author"):
            assert successors(workflow, f"{key}_researcher") == {f"{key}_formatter"}

    def test_join_gates_region_analyzer_on_all_branches(self, model_name):
        """The fan-in MUST be a JoinNode: a plain node fires on ANY
        predecessor, and region_analyzer needs ALL three formatter outputs
        (behavioral pin: tests/unit/test_graph_workflows.py)."""
        workflow = self._workflow(model_name)
        assert predecessors(workflow, "discovery_join") == {
            "city_formatter",
            "landmark_formatter",
            "author_formatter",
        }
        join_nodes = [
            n for n in workflow.graph.nodes if isinstance(n, JoinNode)
        ]
        assert [n.name for n in join_nodes] == ["discovery_join"]
        assert successors(workflow, "discovery_join") == {"region_analyzer"}

    def test_region_analyzer_is_terminal(self, model_name):
        workflow = self._workflow(model_name)
        assert successors(workflow, "region_analyzer") == set()


# =============================================================================
# Book→Place Composition Workflow (phase 2 of the primary flow)
# =============================================================================

class TestBookToPlaceCompositionWorkflow:
    """Tests for create_book_to_place_composition_workflow."""

    def test_creates_workflow_graph(self, model_name):
        workflow = create_book_to_place_composition_workflow(model_name)
        assert isinstance(workflow, Workflow)
        assert workflow.name == "book_to_place_composition"

    def test_single_composer_node(self, model_name):
        workflow = create_book_to_place_composition_workflow(model_name)
        assert edge_pairs(workflow) == {(START_NAME, "trip_composer")}


# =============================================================================
# BookContext Dynamic Instruction Tests
# =============================================================================

class TestBookContextDynamicInstruction:
    """Verify book_context agents handle both known and unknown title/author."""

    def test_known_title_baked_into_instruction(self, model_name, mock_google_search_tool):
        researcher, _ = create_book_context_agents(
            model_name, mock_google_search_tool,
            book_title="1984", author="George Orwell"
        )
        assert '"1984"' in researcher.instruction
        assert "George Orwell" in researcher.instruction

    def test_no_title_uses_dynamic_reference(self, model_name, mock_google_search_tool):
        researcher, _ = create_book_context_agents(
            model_name, mock_google_search_tool
        )
        assert "book_metadata" in researcher.instruction
        assert "[from conversation]" not in researcher.instruction

    def test_title_starting_with_bracket_is_treated_as_real_title(
        self, model_name, mock_google_search_tool
    ):
        researcher, _ = create_book_context_agents(
            model_name,
            mock_google_search_tool,
            book_title="[Pygmalion]",
            author="George Bernard Shaw",
        )
        assert '"[Pygmalion]" by George Bernard Shaw' in researcher.instruction


# =============================================================================
# Prompt Loader Tests
# =============================================================================

class TestLoadPrompts:
    """Tests for the versioned prompt loader."""

    def test_load_prompts_default_returns_agent_prompts(self):
        prompts = load_prompts()
        assert isinstance(prompts, AgentPrompts)
        assert prompts.trip_composer  # non-empty string

    def test_load_prompts_caches_same_object(self):
        assert load_prompts("v2") is load_prompts("v2")

    def test_load_prompts_missing_version_raises(self):
        with pytest.raises(FileNotFoundError, match="v99"):
            load_prompts("v99")

    def test_book_recommendation_prompts_exist(self):
        prompts = load_prompts()
        for field in ("book_recommendation_researcher", "book_recommendation_formatter"):
            assert hasattr(prompts, field)
            template = getattr(prompts, field)
            assert "{book_title}" in template
            assert "{destinations}" in template
            assert "{themes}" in template


# =============================================================================
# Book Recommendation agents + workflow
# =============================================================================


class TestBookRecommendationAgents:
    """Tests for create_book_recommendation_agents."""

    def _pair(self, model_name, tool, **overrides):
        kwargs = dict(
            book_title="1984",
            author="George Orwell",
            destinations="London",
            themes="dystopia, surveillance",
        )
        kwargs.update(overrides)
        return create_book_recommendation_agents(model_name, tool, **kwargs)

    def test_pair_names(self, model_name, mock_google_search_tool):
        researcher, formatter = self._pair(model_name, mock_google_search_tool)
        assert isinstance(researcher, LlmAgent)
        assert isinstance(formatter, LlmAgent)
        assert researcher.name == "book_recommendation_researcher"
        assert formatter.name == "book_recommendation_formatter"

    def test_researcher_uses_search_tool_only(self, model_name, mock_google_search_tool):
        """The researcher gets the tool, the formatter gets the schema. ADK 2.x
        would allow combining them; the separation is the ADR #2
        anti-hallucination contract, kept deliberately."""
        researcher, formatter = self._pair(model_name, mock_google_search_tool)
        assert researcher.tools and len(researcher.tools) == 1
        assert researcher.output_schema is None
        assert not formatter.tools

    def test_formatter_has_output_schema(self, model_name, mock_google_search_tool):
        from models.book import BookRecommendationsResult
        _, formatter = self._pair(model_name, mock_google_search_tool)
        assert formatter.output_schema is BookRecommendationsResult
        assert formatter.output_key == "last_book_recommendations"

    def test_prompt_interpolation(self, model_name, mock_google_search_tool):
        researcher, formatter = self._pair(
            model_name,
            mock_google_search_tool,
            book_title="Middlemarch",
            author="George Eliot",
            destinations="Coventry, London",
            themes="social reform, marriage",
        )
        for agent in (researcher, formatter):
            assert "Middlemarch" in agent.instruction
            assert "George Eliot" in agent.instruction
            assert "Coventry, London" in agent.instruction
            assert "social reform, marriage" in agent.instruction


class TestBookRecommendationWorkflow:
    """Tests for create_book_recommendation_workflow."""

    def _workflow(self, model_name, tool):
        return create_book_recommendation_workflow(
            model_name,
            tool,
            book_title="1984",
            author="George Orwell",
            destinations="London",
            themes="dystopia",
        )

    def test_creates_workflow_graph(self, model_name, mock_google_search_tool):
        wf = self._workflow(model_name, mock_google_search_tool)
        assert isinstance(wf, Workflow)
        assert wf.name == "book_recommendation_workflow"

    def test_researcher_to_formatter_chain(self, model_name, mock_google_search_tool):
        wf = self._workflow(model_name, mock_google_search_tool)
        assert edge_pairs(wf) == {
            (START_NAME, "book_recommendation_researcher"),
            ("book_recommendation_researcher", "book_recommendation_formatter"),
        }
