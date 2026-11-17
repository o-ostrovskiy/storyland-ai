"""Unit tests for the one-way git -> Langfuse prompt sync.

Fake client only: sync() calls nothing but get_prompt/create_prompt, so no
Langfuse credentials or network are needed.
"""

from types import SimpleNamespace

from langfuse.api import NotFoundError

from agents.prompts import CURRENT_PROMPT_VERSION, AgentPrompts, prompt_content_hash
from evaluation.tools import sync_prompts_to_langfuse as sync_mod


class FakeClient:
    """Stores the latest created text per (name, label), like Langfuse labels."""

    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}
        self.created: list[dict] = []

    def get_prompt(self, name, *, label, cache_ttl_seconds):
        assert cache_ttl_seconds == 0, "a stale SDK cache would hide changes"
        try:
            return SimpleNamespace(prompt=self.store[(name, label)])
        except KeyError:
            raise NotFoundError(body="not found")

    def create_prompt(self, **payload):
        self.created.append(payload)
        for label in payload["labels"]:
            self.store[(payload["name"], label)] = payload["prompt"]


class TestTemplateConversion:
    def test_python_fields_become_langfuse_variables(self):
        text, variables = sync_mod.to_langfuse_template("Near {place} within {radius_km} km")
        assert text == "Near {{place}} within {{radius_km}} km"
        assert variables == ["place", "radius_km"]

    def test_text_without_fields_is_untouched(self):
        assert sync_mod.to_langfuse_template("No fields here.") == ("No fields here.", [])


class TestBuildPayloads:
    def test_one_payload_per_agent_labelled_with_the_version_never_production(self):
        payloads = sync_mod.build_payloads()
        names = {p["name"] for p in payloads}
        expected = {f"storyland-ai/{f.name}" for f in AgentPrompts.__dataclass_fields__.values()}
        assert names == expected
        for p in payloads:
            assert p["labels"] == [CURRENT_PROMPT_VERSION]
            assert "production" not in p["labels"]
            assert p["config"]["prompt_hash"] == prompt_content_hash()
            assert p["config"]["source"] == "git"

    def test_payload_is_the_effective_instruction_including_the_guardrail(self):
        from core.guardrails import READER_TONE_GUARDRAIL

        composer = next(p for p in sync_mod.build_payloads() if p["name"].endswith("/trip_composer"))
        assert READER_TONE_GUARDRAIL.strip() in composer["prompt"]

    def test_template_variables_are_recorded(self):
        place = next(p for p in sync_mod.build_payloads() if p["name"].endswith("/place_to_book_researcher"))
        assert place["config"]["variables"] == ["place"]
        assert "{{place}}" in place["prompt"]


class TestSync:
    def test_first_run_creates_every_prompt(self):
        client = FakeClient()
        result = sync_mod.sync(client)
        assert len(result["created"]) == len(AgentPrompts.__dataclass_fields__)
        assert result["unchanged"] == []

    def test_second_run_is_a_noop(self):
        client = FakeClient()
        sync_mod.sync(client)
        client.created.clear()

        result = sync_mod.sync(client)

        assert result["created"] == []
        assert client.created == [], "re-syncing identical prompts must not mint new Langfuse versions"

    def test_only_a_changed_prompt_is_recreated(self):
        client = FakeClient()
        sync_mod.sync(client)
        client.store[("storyland-ai/reader_profile", CURRENT_PROMPT_VERSION)] = "edited in the UI"
        client.created.clear()

        result = sync_mod.sync(client)

        assert result["created"] == ["storyland-ai/reader_profile"]

    def test_dry_run_writes_nothing(self):
        client = FakeClient()
        result = sync_mod.sync(client, dry_run=True)
        assert result["created"]
        assert client.created == []


class TestMain:
    def test_missing_credentials_exit_2_without_a_client(self, monkeypatch):
        monkeypatch.setattr(sync_mod, "load_dotenv", lambda: None)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        assert sync_mod.main([]) == 2
