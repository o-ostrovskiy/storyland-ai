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
            {"has_preferences": False},  # unscored (failed) case: excluded
        ]
        shapes = _summarize_by_shape(results)
        assert shapes["with_preferences"] == {
            "n": 2, "mean_average": 3.5, "mean_total_tokens": 1500,
        }
        assert shapes["without_preferences"]["n"] == 1
        assert shapes["without_preferences"]["mean_average"] == 2.0

    def test_empty_shape_reports_zero(self):
        shapes = _summarize_by_shape([self._case(True, 4.0)])
        assert shapes["without_preferences"] == {"n": 0}


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
        with pytest.raises(Exception):
            await llm_scorer.score_itinerary(
                api_key="k", book_title="B", author="A", input_text="i",
                itinerary={"cities": []},
                preferences={"pace": "fast"},
            )
