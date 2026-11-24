"""ai-private#67 Part 2/3: the film and series prompts and where they are wired.

The book path must not move by a byte (``for_kind("book")`` is the v4 set,
its workflows build the same instructions, its cache key is unchanged); a
film or series gets the screen overlay on the same graph, with no extra agents.
"""

import dataclasses
import json

import pytest
from google.adk.tools import google_search

from agents import work_prompts
from agents.book_context_agent import create_book_context_agents
from agents.orchestrator import (
    create_book_to_place_composition_workflow,
    create_book_to_place_discovery_workflow,
)
from agents.prompts import AgentPrompts, load_prompts
from agents.work_prompts import _SCREEN_OVERLAY, for_kind, is_screen_kind, screen_prompt_hash
from models.book import BookContext
from models.screen import ScreenContext, ScreenResearch

MODEL = "gemini-test"
OVERLAY = json.loads(_SCREEN_OVERLAY.read_text(encoding="utf-8"))
SCREEN = ("film", "series")


def _agents(workflow) -> dict:
    """name -> LlmAgent for every agent node in a Workflow graph."""
    found: dict = {}
    for edge in workflow.edges:
        for node in edge if isinstance(edge, tuple) else (edge,):
            for item in node if isinstance(node, tuple) else (node,):
                if hasattr(item, "instruction"):
                    found[item.name] = item
    return found


@pytest.fixture(autouse=True)
def _fresh_overlay_cache():
    work_prompts._cache.clear()
    yield
    work_prompts._cache.clear()


class TestForKind:
    def test_book_is_the_loaded_set_itself(self):
        assert for_kind("book") is load_prompts()

    @pytest.mark.parametrize("kind", SCREEN)
    def test_replaced_agents_carry_the_overlay_text(self, kind):
        prompts = for_kind(kind)
        for name, text in OVERLAY["replace"].items():
            assert getattr(prompts, name) == text

    @pytest.mark.parametrize("kind", SCREEN)
    def test_the_composer_keeps_the_book_text_and_adds_the_screen_rules(self, kind):
        book = load_prompts().trip_composer
        screen = for_kind(kind).trip_composer
        assert screen == book + OVERLAY["append"]["trip_composer"]
        for rule in ('"filmed_here"', '"set_here"', "filming_scene", "fictional"):
            assert rule in screen

    @pytest.mark.parametrize("kind", SCREEN)
    def test_every_other_agent_is_the_book_text(self, kind):
        touched = set(OVERLAY["replace"]) | set(OVERLAY["append"])
        book, screen = load_prompts(), for_kind(kind)
        untouched = [f.name for f in dataclasses.fields(AgentPrompts) if f.name not in touched]
        assert untouched  # region_analyzer, the formatters, place_to_book...
        for name in untouched:
            assert getattr(screen, name) == getattr(book, name), name

    def test_the_context_researcher_keeps_its_placeholders(self):
        text = for_kind("film").book_context_researcher
        assert text.format(book_ref="REF", search_hint="HINT").count("REF") == 1

    def test_an_overlay_naming_an_unknown_agent_is_refused(self, tmp_path, monkeypatch):
        bad = tmp_path / "screen_v1.json"
        bad.write_text(json.dumps({"replace": {"no_such_agent": "x"}}))
        monkeypatch.setattr(work_prompts, "_SCREEN_OVERLAY", bad)
        with pytest.raises(ValueError, match="no_such_agent"):
            for_kind("film")

    def test_is_screen_kind(self):
        assert [is_screen_kind(k) for k in ("book", "film", "series", "podcast")] == [
            False,
            True,
            True,
            False,
        ]


class TestScreenPromptHash:
    def test_hash_is_short_and_follows_the_file(self, tmp_path, monkeypatch):
        assert len(screen_prompt_hash()) == 12
        other = tmp_path / "screen_v1.json"
        other.write_text("{}")
        monkeypatch.setattr(work_prompts, "_SCREEN_OVERLAY", other)
        assert screen_prompt_hash() != ""
        assert screen_prompt_hash() != work_prompts.hashlib.sha1(_SCREEN_OVERLAY.read_bytes()).hexdigest()[:12]

    def test_unreadable_overlay_hashes_to_empty_never_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(work_prompts, "_SCREEN_OVERLAY", tmp_path / "missing.json")
        assert screen_prompt_hash() == ""


class TestScreenResearch:
    def test_the_formatter_schema_has_no_source_book_field(self):
        # The source book is the backend's (Story 6): the model must have no
        # field in which to invent one.
        assert "source_work" not in ScreenResearch.model_fields
        research = ScreenResearch.model_validate(
            {"filming_locations": ["Dubrovnik"], "source_work": {"kind": "book", "title": "Invented"}}
        )
        assert not hasattr(research, "source_work")

    def test_an_invented_world_keeps_no_story_locations(self):
        research = ScreenResearch(
            story_locations=["Westeros"], filming_locations=["Dubrovnik"], is_fictional_world=True
        )
        assert research.story_locations == []
        assert research.filming_locations == ["Dubrovnik"]

    def test_screen_context_still_carries_the_rules_and_the_source_work(self):
        ctx = ScreenContext(story_locations=[" Paris ", "paris"], time_period="2020s")
        assert ctx.story_locations == ["Paris"]
        assert ctx.time_period == "2020s"
        assert "source_work" in ScreenContext.model_fields


class TestContextAgents:
    def test_book_keeps_book_context_and_its_instruction(self):
        researcher, formatter = create_book_context_agents(
            MODEL, google_search, book_title="Persuasion", author="Jane Austen"
        )
        assert formatter.output_schema is BookContext
        assert 'BOOK: "Persuasion" by Jane Austen' in researcher.instruction
        assert "setting location" in researcher.instruction

    @pytest.mark.parametrize("kind", SCREEN)
    def test_screen_uses_screen_research_and_filming_hints(self, kind):
        researcher, formatter = create_book_context_agents(
            MODEL,
            google_search,
            book_title="Game of Thrones",
            author="David Benioff",
            prompts=for_kind(kind),
            work_kind=kind,
        )
        assert formatter.output_schema is ScreenResearch
        assert formatter.output_key == "book_context"  # composer + recommender read this key
        word = "series" if kind == "series" else "film"
        assert f'{word.upper()}: "Game of Thrones" (created by David Benioff)' in researcher.instruction
        assert f"Game of Thrones {word} filming locations" in researcher.instruction
        assert "BOOK: \"" not in researcher.instruction


class TestWorkflows:
    def test_book_discovery_is_byte_identical_to_the_explicit_v4_build(self):
        default = _agents(
            create_book_to_place_discovery_workflow(MODEL, book_title="Persuasion", author="Jane Austen")
        )
        explicit = _agents(
            create_book_to_place_discovery_workflow(
                MODEL, book_title="Persuasion", author="Jane Austen", prompts=load_prompts()
            )
        )
        assert default.keys() == explicit.keys()
        for name in default:
            assert default[name].instruction == explicit[name].instruction, name
        assert default["book_context_formatter"].output_schema is BookContext

    @pytest.mark.parametrize("kind", SCREEN)
    def test_screen_discovery_runs_the_same_graph_with_screen_instructions(self, kind):
        book = _agents(create_book_to_place_discovery_workflow(MODEL, book_title="X", author="Y"))
        screen = _agents(
            create_book_to_place_discovery_workflow(MODEL, book_title="X", author="Y", work_kind=kind)
        )
        assert screen.keys() == book.keys()  # no extra agents
        for name in ("city_researcher", "landmark_researcher", "author_researcher"):
            assert screen[name].instruction.endswith(OVERLAY["replace"][name])
            assert screen[name].instruction != book[name].instruction
        assert screen["region_analyzer"].instruction == book["region_analyzer"].instruction
        assert screen["book_context_formatter"].output_schema is ScreenResearch

    def test_book_composition_is_the_v4_composer(self):
        agents = _agents(create_book_to_place_composition_workflow(MODEL))
        assert agents["trip_composer"].instruction == load_prompts().trip_composer

    @pytest.mark.parametrize("kind", SCREEN)
    def test_screen_composition_gets_the_claim_rules(self, kind):
        agents = _agents(create_book_to_place_composition_workflow(MODEL, work_kind=kind))
        assert agents["trip_composer"].instruction == for_kind(kind).trip_composer
