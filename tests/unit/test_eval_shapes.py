"""Unit tests for the both-shapes eval machinery (PR-4 step zero).

The books evalset carries two request shapes: preference-carrying cases (the
documented API-contract path) and preference-free cases (the shape 100% of
prod traffic takes since MYS-392 — issue #221). These pin: the 5-dimension
average for unscored preference_adherence, the per-shape aggregation the
gates read, and the evalset's shape split itself.
"""

import json
from pathlib import Path

import pytest

from evaluation.tools.llm_scorer import ItineraryScores
from evaluation.tools.run_scheduled_eval import _summarize_by_shape


class TestOptionalPreferenceDimension:
    def test_average_over_six_when_scored(self):
        scores = ItineraryScores(
            book_relevance=4, preference_adherence=2, completeness=4,
            actionability=4, geographical_accuracy=4, engagement=4,
        )
        assert scores.average_score() == pytest.approx(22 / 6)

    def test_average_over_five_when_not_scored(self):
        scores = ItineraryScores(
            book_relevance=4, preference_adherence=None, completeness=4,
            actionability=4, geographical_accuracy=4, engagement=4,
        )
        assert scores.average_score() == pytest.approx(4.0)

    def test_none_is_default(self):
        scores = ItineraryScores(
            book_relevance=3, completeness=3, actionability=3,
            geographical_accuracy=3, engagement=3,
        )
        assert scores.preference_adherence is None


class TestSummarizeByShape:
    def _case(self, has_prefs, avg, tokens=1000):
        return {
            "has_preferences": has_prefs,
            "scores": {"average": avg},
            "token_usage": {"total_tokens": tokens},
        }

    def test_segments_and_never_blends(self):
        results = [
            self._case(True, 4.0, 2000),
            self._case(True, 3.0, 1000),
            self._case(False, 2.0, 500),
            {"has_preferences": False},  # unscored (failed) case
        ]
        shapes = _summarize_by_shape(results)
        assert shapes["with_preferences"] == {
            "n": 2, "n_unscored": 0, "mean_average": 3.5, "mean_total_tokens": 1500,
        }
        assert shapes["without_preferences"]["n"] == 1
        assert shapes["without_preferences"]["mean_average"] == 2.0
        # The unscored case stays VISIBLE in its cell, never silently dropped.
        assert shapes["without_preferences"]["n_unscored"] == 1

    def test_empty_shape_reports_zero(self):
        shapes = _summarize_by_shape([self._case(True, 4.0)])
        assert shapes["without_preferences"] == {"n": 0, "n_unscored": 0}

    def test_all_unscored_shape_still_visible(self):
        """A shape where every case failed scoring reports n=0 but a nonzero
        n_unscored — the difference between 'no cases ran' and 'cases ran and
        none could be scored' must be readable from the artifact."""
        shapes = _summarize_by_shape([{"has_preferences": True}])
        assert shapes["with_preferences"] == {"n": 0, "n_unscored": 1}


class TestEvalsetShapeSplit:
    def test_books_v1_carries_both_shapes(self):
        """The gate's validity depends on this split existing — pin it."""
        d = json.loads(
            (Path("evaluation") / "books_v1.evalset.json").read_text()
        )
        shapes = {True: 0, False: 0}
        for case in d["eval_cases"]:
            state = case.get("session_input", {}).get("state", {}) or {}
            has = bool(state.get("user:preferences"))
            shapes[has] += 1
            # A preference-free case must not carry a preference criterion
            # (the judge would grade adherence to nothing).
            if not has:
                assert "preference_adherence" not in (
                    case.get("quality_criteria") or {}
                ), case["eval_id"]
        assert shapes[True] >= 3, "API-contract shape underrepresented"
        assert shapes[False] >= 3, "prod shape underrepresented"


class TestModelDefault:
    def _base_env(self, monkeypatch):
        # load_config's full required-env set (same as test_retry_backoff).
        for k, v in {
            "GOOGLE_API_KEY": "k",
            "USE_DATABASE": "false",
            "SESSION_MAX_EVENTS": "100",
            "MAX_CONTEXT_TOKENS": "1000",
            "WORKFLOW_TIMEOUT": "300",
            "AGENT_TIMEOUT": "60",
            "LOG_LEVEL": "INFO",
            "ENABLE_ADK_DEBUG": "false",
        }.items():
            monkeypatch.setenv(k, v)

    def test_model_name_defaults_without_env(self, monkeypatch):
        """A deploy missing MODEL_NAME boots on the code default (old footgun:
        _require_env crashed at startup)."""
        from common.config import DEFAULT_MODEL_NAME, load_config

        self._base_env(monkeypatch)
        monkeypatch.delenv("MODEL_NAME", raising=False)
        config = load_config()
        assert config.model_name == DEFAULT_MODEL_NAME
        assert DEFAULT_MODEL_NAME == "gemini-3.1-flash-lite"

    def test_model_name_env_wins(self, monkeypatch):
        from common.config import load_config

        self._base_env(monkeypatch)
        monkeypatch.setenv("MODEL_NAME", "gemini-3.5-flash")
        assert load_config().model_name == "gemini-3.5-flash"


class TestJudgeModel:
    """The judge is an instrument; it must not drift or grade its own work."""

    def test_judge_is_not_the_model_under_test(self):
        """Same model both sides = the family grading itself (MYS-825)."""
        from common.config import DEFAULT_MODEL_NAME
        from evaluation.tools.llm_scorer import _DEFAULT_JUDGE_MODEL

        assert _DEFAULT_JUDGE_MODEL != DEFAULT_MODEL_NAME

    def test_judge_is_pinned_not_an_alias(self):
        """A `-latest` alias would re-point the instrument under the numbers
        with nothing in the diff — the same silent change as the 404 that
        zeroed every score on 2026-08-09."""
        from evaluation.tools.llm_scorer import _DEFAULT_JUDGE_MODEL

        assert not _DEFAULT_JUDGE_MODEL.endswith("-latest")

    def test_scoring_functions_use_the_constant(self):
        """Both entry points default to the one constant, so re-pinning after
        a retirement is a single edit."""
        import inspect

        from evaluation.tools.llm_scorer import (
            _DEFAULT_JUDGE_MODEL,
            score_criteria_coverage,
            score_itinerary,
        )

        for fn in (score_itinerary, score_criteria_coverage):
            default = inspect.signature(fn).parameters["model_name"].default
            assert default == _DEFAULT_JUDGE_MODEL, fn.__name__


class TestCountItineraryCities:
    """The composer returns an envelope; the count must reach into it.

    Reading top-level ["cities"] reported 0 for every successful run — beside
    itineraries carrying up to 11 cities in the same result file (MYS-825).
    """

    def _envelope(self, cities):
        return {
            "itinerary": {
                "cities": [
                    {"name": n, "country": "X", "days_suggested": 1,
                     "overview": "o", "stops": []}
                    for n in cities
                ],
                "summary_text": "s",
            },
            "suggestions": [],
        }

    def test_counts_cities_inside_the_envelope(self):
        from evaluation.tools.run_scheduled_eval import count_itinerary_cities

        assert count_itinerary_cities(self._envelope(["Bath", "Winchester"])) == 2

    def test_top_level_shape_still_counts(self):
        from evaluation.tools.run_scheduled_eval import count_itinerary_cities

        assert count_itinerary_cities({"cities": [{"name": "Bath"}]}) == 1

    def test_schema_invalid_payload_falls_back_rather_than_zeroing(self):
        """A payload the validator rejects must not silently read as 0 —
        that is the failure being fixed, not a behaviour to preserve."""
        from evaluation.tools.run_scheduled_eval import count_itinerary_cities

        assert count_itinerary_cities({"itinerary": {"cities": [{"nope": 1}]}}) == 1

    def test_empty_and_missing_are_zero(self):
        from evaluation.tools.run_scheduled_eval import count_itinerary_cities

        for payload in (None, {}, {"itinerary": {}}, {"itinerary": {"cities": []}}):
            assert count_itinerary_cities(payload) == 0


class TestJudgeDimensionContract:
    """Codex P2 on PR #228: preference_adherence is Optional for the
    no-preference shape, so a judge omitting it on a preference-CARRYING case
    would silently average 5 dims — the API-contract shape passing without
    adherence measured. The scorer now raises into its scoring-failed path
    instead. (Audit of all four PR-4 eval runs on disk: zero occurrences —
    latent, not fired; this pins it shut.)"""

    def test_scores_model_allows_none(self):
        # The model itself stays permissive (no-pref shape needs None)...
        s = ItineraryScores(
            book_relevance=3, completeness=3, actionability=3,
            geographical_accuracy=3, engagement=3,
        )
        assert s.preference_adherence is None

    async def test_missing_demanded_dimension_fails_scoring(self, monkeypatch):
        """score_itinerary with preferences must FAIL (not 5-dim-average) when
        the judge response omits preference_adherence."""
        import json as _json
        from types import SimpleNamespace

        from evaluation.tools import llm_scorer

        class _FakeModels:
            def generate_content(self, **kwargs):
                return SimpleNamespace(
                    text=_json.dumps({
                        "book_relevance": 4, "completeness": 4,
                        "actionability": 4, "geographical_accuracy": 4,
                        "engagement": 4,
                    }),
                    usage_metadata=None,
                )

        monkeypatch.setattr(
            llm_scorer.genai, "Client",
            lambda **kw: SimpleNamespace(models=_FakeModels()),
        )
        # Narrow by review: pytest.raises(Exception) would pass on ANY failure
        # — a mis-wired fake raising AttributeError before the parse would go
        # green with the guard never exercised (the same
        # success-over-the-thing-it-checks class this PR closes). The match
        # pins that THE GUARD is what fired.
        with pytest.raises(ValueError, match="omitted preference_adherence"):
            await llm_scorer.score_itinerary(
                api_key="k", book_title="B", author="A", input_text="i",
                itinerary={"cities": []},
                preferences={"pace": "fast"},
            )
