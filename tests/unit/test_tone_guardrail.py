"""Tests for the rec-explanation tone guardrail.

Covers:
  * core.guardrails.tone_guardrail.flag_reader_directed — catches reader-directed
    verdicts/labels/inferences, ignores legitimate fit explanations (incl. benign
    second person about the place).
  * sanitize_explanation — drops only the offending sentence, keeps fit-only text,
    falls back when everything is stripped, is fail-open on non-string input.
  * sanitize_{itinerary,expansion,book_recommendations} — walk the reader-facing
    fields and are fail-open on None/empty.
  * agents.prompts.load_prompts — appends READER_TONE_GUARDRAIL to the
    explanation-producing agents only, and is idempotent.

All deterministic and offline — no Gemini/model calls.
"""

import pytest

from core.guardrails.tone_guardrail import (
    READER_TONE_GUARDRAIL,
    flag_reader_directed,
    sanitize_explanation,
    sanitize_itinerary_explanations,
    sanitize_expansion_explanations,
    sanitize_book_recommendations,
    _FIT_FALLBACK,
)


READER_DIRECTED = [
    "Your taste is rather pedestrian, but this town still works.",
    "Your reading choices reveal a melancholic streak.",
    "You seem like the anxious type, so a quiet village suits you.",
    "You're the kind of reader who needs constant stimulation.",
    "Readers like you usually prefer something lighter.",
    "This says a lot about you and your shallow preferences.",
    "As a sophisticated reader, you will appreciate the nuance.",
    "You must be quite the romantic to pick this one.",
]

FIT_ONLY = [
    "You'll wander the same foggy quays the detective walked at midnight.",
    "Chosen for its gothic cathedrals and mist-laden moors that mirror the dread.",
    "The cafe's candle-lit interior echoes the book's slow-burn romance.",
    "Stand where the duel was fought on these very steps.",
    "As a reader, immerse yourself in the salt-air harbour the author describes.",
    "This neighbourhood reveals the city's literary past in every plaque.",
    "You can walk the trail that inspired the wilderness chapters.",
    "With a relaxed pace in mind, this itinerary lingers in two coastal towns.",
]


class TestFlagReaderDirected:
    @pytest.mark.parametrize("text", READER_DIRECTED)
    def test_flags_reader_directed(self, text):
        assert flag_reader_directed(text), f"should flag: {text!r}"

    @pytest.mark.parametrize("text", FIT_ONLY)
    def test_passes_fit_only(self, text):
        assert flag_reader_directed(text) == [], f"should NOT flag: {text!r}"

    def test_fail_open_on_non_string(self):
        assert flag_reader_directed(None) == []
        assert flag_reader_directed(123) == []
        assert flag_reader_directed("") == []
        assert flag_reader_directed("   ") == []


class TestSanitizeExplanation:
    def test_clean_text_unchanged(self):
        text = FIT_ONLY[0]
        clean, matched = sanitize_explanation(text)
        assert clean == text
        assert matched == []

    def test_drops_only_offending_sentence(self):
        text = (
            "Chosen for its mist-laden moors that mirror the novel's dread. "
            "Your taste is rather basic, frankly. "
            "Stand where the duel was fought."
        )
        clean, matched = sanitize_explanation(text)
        assert matched  # something was flagged
        assert "Your taste is" not in clean
        assert "mist-laden moors" in clean
        assert "duel was fought" in clean

    def test_full_strip_falls_back(self):
        text = "Your taste is shallow. You seem like a tourist."
        clean, matched = sanitize_explanation(text)
        assert matched
        assert clean == _FIT_FALLBACK

    def test_custom_fallback(self):
        clean, _ = sanitize_explanation("Readers like you are lazy.", fallback="X")
        assert clean == "X"

    def test_fail_open_on_non_string(self):
        assert sanitize_explanation(None) == ("", [])
        assert sanitize_explanation(42) == ("", [])


class TestSanitizeWalkers:
    def test_itinerary_sanitized(self):
        itin = {
            "summary_text": "A trip for you. This says a lot about you and your taste.",
            "cities": [
                {
                    "overview": "Your reading habits suggest you'd love this.",
                    "stops": [
                        {"name": "Quay", "reason": "Foggy quays echo the novel's gloom."},
                        {"name": "Bar", "reason": "You're the kind of reader who drinks alone."},
                    ],
                }
            ],
        }
        out = sanitize_itinerary_explanations(itin)
        assert "about you" not in out["summary_text"]
        assert "reading habits" not in out["cities"][0]["overview"]
        # clean stop untouched, dirty stop scrubbed
        assert out["cities"][0]["stops"][0]["reason"] == "Foggy quays echo the novel's gloom."
        assert "kind of reader" not in out["cities"][0]["stops"][1]["reason"]

    def test_itinerary_fail_open(self):
        assert sanitize_itinerary_explanations(None) is None
        assert sanitize_itinerary_explanations({}) == {}

    def test_expansion_sanitized(self):
        exp = {"places": [{"reason": "Readers like you love this."}, {"reason": "Quiet gothic chapel."}]}
        out = sanitize_expansion_explanations(exp)
        assert "like you" not in out["places"][0]["reason"]
        assert out["places"][1]["reason"] == "Quiet gothic chapel."

    def test_expansion_fail_open(self):
        assert sanitize_expansion_explanations(None) is None

    def test_book_recommendations_sanitized(self):
        rec = {"recommendations": [
            {"title": "A", "reason": "Your taste is basic, so try this."},
            {"title": "B", "reason": "A windswept saga set on the same coast."},
        ]}
        out = sanitize_book_recommendations(rec)
        assert "Your taste is" not in out["recommendations"][0]["reason"]
        assert out["recommendations"][1]["reason"] == "A windswept saga set on the same coast."

    def test_book_recommendations_fail_open(self):
        assert sanitize_book_recommendations(None) is None


class TestPromptInjection:
    def test_guardrail_constant_has_scope_and_prohibited_list(self):
        # AC1: the clause scopes output to the fit and carries a prohibited list.
        assert "READER-SAFETY GUARDRAIL" in READER_TONE_GUARDRAIL
        assert "Prohibited" in READER_TONE_GUARDRAIL
        assert "never" in READER_TONE_GUARDRAIL.lower()

    def test_explanation_agents_carry_guardrail(self):
        from agents.prompts import load_prompts, _EXPLANATION_AGENTS

        prompts = load_prompts()
        for name in _EXPLANATION_AGENTS:
            assert READER_TONE_GUARDRAIL in getattr(prompts, name), name

    def test_researcher_agents_do_not_carry_guardrail(self):
        from agents.prompts import load_prompts

        prompts = load_prompts()
        # Researchers only gather candidates; they don't author explanations.
        assert READER_TONE_GUARDRAIL not in prompts.book_recommendation_researcher
        assert READER_TONE_GUARDRAIL not in prompts.city_researcher

    def test_injection_idempotent(self):
        from agents.prompts import _inject_tone_guardrail

        base = {"trip_composer": "Plan a trip.", "city_researcher": "Search."}
        once = _inject_tone_guardrail(base)
        twice = _inject_tone_guardrail(once)
        assert once["trip_composer"] == twice["trip_composer"]
        assert once["trip_composer"].count("READER-SAFETY GUARDRAIL") == 1
        # non-explanation agent untouched
        assert twice["city_researcher"] == "Search."
