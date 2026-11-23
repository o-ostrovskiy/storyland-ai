"""READER_CLARITY_RULES reach every agent that writes reader-facing text (2026-11-22)."""

from agents.prompts import (
    READER_CLARITY_RULES,
    _CONNECTION_AGENTS,
    _EXPLANATION_AGENTS,
    _inject_clarity_rules,
    load_prompts,
)


def test_rules_cover_the_link_and_the_honesty_limits():
    text = READER_CLARITY_RULES
    assert "WRITE FOR THE READER'S UNDERSTANDING" in text
    assert "Say the link first" in text
    assert "Add nothing the research" in text
    assert "length limit" in text
    # Several instructions are str.format templates: a brace would break them.
    assert "{" not in text and "}" not in text


def test_explanation_and_connection_agents_carry_the_rules():
    prompts = load_prompts()
    for name in _EXPLANATION_AGENTS + _CONNECTION_AGENTS:
        assert getattr(prompts, name).count(READER_CLARITY_RULES) == 1, name


def test_researchers_do_not_carry_the_rules():
    prompts = load_prompts()
    assert READER_CLARITY_RULES not in prompts.city_researcher
    assert READER_CLARITY_RULES not in prompts.place_to_book_researcher


def test_template_agents_still_format():
    prompts = load_prompts()
    prompts.place_to_book_formatter.format(place="Luga")
    prompts.local_atmosphere_formatter.format(location_label="Luga", radius_km=10)


def test_injection_is_idempotent_and_leaves_others_alone():
    base = {"city_formatter": "Format.", "city_researcher": "Search."}
    once = _inject_clarity_rules(base)
    twice = _inject_clarity_rules(once)
    assert once == twice
    assert twice["city_researcher"] == "Search."
