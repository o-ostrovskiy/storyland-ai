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

from core.session_state import SessionStateAccessor, SessionStateKeys
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

    def test_no_runner_hardcodes_a_judge_model(self):
        """The defaults above are not enough on their own.

        The first fix re-pinned the constant while both call sites still passed
        `model_name="gemini-2.5-flash-lite"` explicitly — the default-checking
        test above passed, and the next live run failed on the dead model
        exactly as before. Pin the call sites at source level too.
        """
        from pathlib import Path

        for name in (
            "run_scheduled_eval.py",
            "run_local_atmosphere_eval.py",
            "llm_scorer.py",
        ):
            source = (Path("evaluation") / "tools" / name).read_text()
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or "_DEFAULT_JUDGE_MODEL" in line:
                    continue
                assert "gemini-2.5" not in line, f"{name}: {stripped}"


class TestDatasetFailureReason:
    """A run that measured nothing must not report success (MYS-825).

    The 2026-08-09 failure: judge 404s, every case catches its own error and
    records scores=None, and the summary prints "All evaluations completed
    successfully" with exit 0. "No result" is the one outcome that looks
    identical to a pass, so it needs an explicit check.
    """

    def _result(self, cases, **overrides):
        base = {
            "dataset_name": "storyland_eval",
            "total_cases": len(cases),
            "evaluated_cases": len(cases),
            "failed_cases": 0,
            "skipped_cases": 0,
            "case_results": cases,
        }
        base.update(overrides)
        return base

    def _scored(self):
        return {"status": "evaluated", "scores": {"average": 4.0}}

    def _unscored(self):
        return {"status": "evaluated", "scores": None}

    def test_all_unscored_is_a_failure(self):
        from evaluation.tools.run_scheduled_eval import dataset_failure_reason

        reason = dataset_failure_reason(self._result([self._unscored()] * 3))
        assert reason is not None
        assert "scored 0 of 3" in reason
        assert "llm_scorer.py" in reason  # points at the fix

    def test_all_scored_passes(self):
        from evaluation.tools.run_scheduled_eval import dataset_failure_reason

        assert dataset_failure_reason(self._result([self._scored()] * 3)) is None

    def test_partial_scoring_is_not_a_failure(self):
        """One judge hiccup is noise; it surfaces as a WARNING, not a red run."""
        from evaluation.tools.run_scheduled_eval import dataset_failure_reason

        result = self._result([self._scored(), self._unscored()])
        assert dataset_failure_reason(result) is None

    def test_empty_dataset_is_not_a_failure(self):
        from evaluation.tools.run_scheduled_eval import dataset_failure_reason

        assert dataset_failure_reason(self._result([], total_cases=0)) is None

    def test_explicit_error_still_reported_first(self):
        from evaluation.tools.run_scheduled_eval import dataset_failure_reason

        result = self._result([self._unscored()], error="Failed to fetch dataset")
        assert "Failed to fetch dataset" in dataset_failure_reason(result)

    def test_failed_cases_still_reported(self):
        from evaluation.tools.run_scheduled_eval import dataset_failure_reason

        result = self._result([self._scored()], failed_cases=2)
        assert "2 case(s) failed" in dataset_failure_reason(result)

    def test_count_scored_cases_tolerates_missing_key(self):
        from evaluation.tools.run_scheduled_eval import count_scored_cases

        assert count_scored_cases({}) == 0
        assert count_scored_cases({"case_results": None}) == 0


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


class TestSearchGroundingReporting:
    """MYS-817: the eval reports whether researchers actually searched.

    Deterministic and reported BESIDE the judge scores, never folded into
    them. The judge grades itinerary quality and cannot tell a searched answer
    from a remembered one, which is exactly how MYS-816 went unnoticed — a
    researcher that skips google_search still returns a plausible itinerary
    that scores fine.

    Report-only by decision: skips are currently near-universal, so a hard
    gate would be red on every run and would get switched off.
    """

    @staticmethod
    def _ledger(searched=(), unsearched=()):
        """A plugin whose ledger observed exactly these agents.

        Deliberately stubs BOTH accessors. A stub carrying only
        ``unsearched_agents`` is what let the grounded count be derived from an
        absence in the first place — it could not express "observed and
        searched" at all, so the only number available was a subtraction.
        """
        from types import SimpleNamespace

        return SimpleNamespace(
            searched_agents=lambda candidates: frozenset(searched),
            unsearched_agents=lambda candidates: frozenset(unsearched),
        )

    def test_summarize_counts_unsearched_researchers(self):
        from evaluation.tools.run_scheduled_eval import summarize_search_grounding
        from core.executor import DISCOVERY_RESEARCHER_AUTHORS

        others = [a for a in DISCOVERY_RESEARCHER_AUTHORS if a != "author_researcher"]
        result = summarize_search_grounding(
            self._ledger(searched=others, unsearched=["author_researcher"])
        )
        assert result == {
            "researchers_total": 4,
            "researchers_grounded": 3,
            "unsearched": ["author_researcher"],
            "unobserved": [],
        }

    def test_summarize_all_grounded(self):
        from evaluation.tools.run_scheduled_eval import summarize_search_grounding
        from core.executor import DISCOVERY_RESEARCHER_AUTHORS

        result = summarize_search_grounding(
            self._ledger(searched=DISCOVERY_RESEARCHER_AUTHORS)
        )
        assert result["researchers_grounded"] == result["researchers_total"]
        assert result["unsearched"] == []
        assert result["unobserved"] == []

    def test_an_empty_ledger_grounds_nobody(self):
        """The row the previous derivation got exactly backwards.

        ``total - len(unsearched)`` reported a clean 4/4 here, because
        ``unsearched_agents`` correctly omits an agent it never observed. So
        the single state that means "the instrumentation broke" produced the
        best possible score — the MYS-492 shape, on the metric built to catch
        it: a confident claim from a computation that never ran.
        """
        from evaluation.tools.run_scheduled_eval import summarize_search_grounding
        from core.executor import DISCOVERY_RESEARCHER_AUTHORS

        result = summarize_search_grounding(self._ledger())
        assert result["researchers_grounded"] == 0
        assert result["unsearched"] == []
        assert result["unobserved"] == sorted(DISCOVERY_RESEARCHER_AUTHORS)

    def test_a_partial_ledger_does_not_inflate_the_count(self):
        """Less visible than the empty case and the same defect."""
        from evaluation.tools.run_scheduled_eval import summarize_search_grounding

        result = summarize_search_grounding(
            self._ledger(searched=["city_researcher"], unsearched=["author_researcher"])
        )
        assert result["researchers_grounded"] == 1
        assert result["unsearched"] == ["author_researcher"]
        assert len(result["unobserved"]) == 2

    async def test_the_summary_reads_a_REAL_plugin_through_the_seam(self):
        """The row that makes every other row in this class mean something.

        Every case above feeds ``summarize_search_grounding`` a
        ``SimpleNamespace`` carrying the two accessors independently — a
        fixture that can express arrangements the real ledger cannot produce,
        and one that stays green through a rename or signature change on
        ``LangfusePlugin.searched_agents``. That is the label-pinning shape:
        a test of the summary's arithmetic standing in for a test of the seam.

        So this drives ``after_model_callback`` on a real credential-less
        plugin and summarises THAT — no stub anywhere in the path. The
        disjointness the other rows assert is here a consequence of the real
        ledger's write order rather than of how the fixture was written.
        """
        from evaluation.tools.run_scheduled_eval import summarize_search_grounding
        from core.executor import DISCOVERY_RESEARCHER_AUTHORS
        from plugins.langfuse_plugin import LangfusePlugin
        from types import SimpleNamespace

        authors = list(DISCOVERY_RESEARCHER_AUTHORS)
        searched_name, skipped_name = authors[0], authors[1]

        def _ctx(agent_name):
            return SimpleNamespace(
                agent_name=agent_name,
                _invocation_context=SimpleNamespace(branch="b"),
            )

        grounded = SimpleNamespace(
            usage_metadata=None,
            grounding_metadata=SimpleNamespace(
                web_search_queries=["dublin literary sites"],
                grounding_chunks=[
                    SimpleNamespace(
                        web=SimpleNamespace(
                            uri="https://example.com/d", title="D", domain=None
                        )
                    )
                ],
            ),
        )
        toolless = SimpleNamespace(usage_metadata=None, grounding_metadata=None)

        plugin = LangfusePlugin(secret_key=None, public_key=None, host=None)
        await plugin.after_model_callback(
            callback_context=_ctx(searched_name), llm_response=grounded
        )
        await plugin.after_model_callback(
            callback_context=_ctx(skipped_name), llm_response=toolless
        )

        result = summarize_search_grounding(plugin)

        assert result["researchers_grounded"] == 1
        assert result["unsearched"] == [skipped_name]
        # The two never driven through the callback are neither grounded nor
        # skips — they are holes, and the real ledger is what says so.
        assert result["unobserved"] == sorted(authors[2:])
        assert (
            result["researchers_grounded"]
            + len(result["unsearched"])
            + len(result["unobserved"])
            == result["researchers_total"]
        )

    def test_the_three_states_partition_the_researchers(self):
        """grounded + unsearched + unobserved == total, in every arrangement.

        The invariant is what makes the number readable: any researcher missing
        from the grounded count is accounted for as either a skip or a hole,
        never dropped.
        """
        from evaluation.tools.run_scheduled_eval import summarize_search_grounding
        from core.executor import DISCOVERY_RESEARCHER_AUTHORS

        authors = list(DISCOVERY_RESEARCHER_AUTHORS)
        arrangements = [
            ((), ()),
            (authors, ()),
            ((), authors),
            (authors[:1], authors[1:2]),
            (authors[:2], authors[2:]),
        ]
        for searched, unsearched in arrangements:
            r = summarize_search_grounding(self._ledger(searched, unsearched))
            assert (
                r["researchers_grounded"] + len(r["unsearched"]) + len(r["unobserved"])
                == r["researchers_total"]
            ), (searched, unsearched)

    def test_aggregate_is_none_when_no_case_reported(self):
        """An all-zero block would read as 'nothing was grounded' — omit it."""
        from evaluation.tools.run_scheduled_eval import aggregate_search_grounding

        assert aggregate_search_grounding([]) is None
        assert aggregate_search_grounding([{"status": "evaluated"}]) is None

    def test_aggregate_ranks_repeat_offenders(self):
        """Which researcher skips MOST is the actionable number: the offender
        varies run to run rather than being one permanently broken agent."""
        from evaluation.tools.run_scheduled_eval import aggregate_search_grounding

        cases = [
            {"search_grounding": {
                "researchers_total": 4, "researchers_grounded": 3,
                "unsearched": ["author_researcher"]}},
            {"search_grounding": {
                "researchers_total": 4, "researchers_grounded": 2,
                "unsearched": ["author_researcher", "city_researcher"]}},
            {"search_grounding": {
                "researchers_total": 4, "researchers_grounded": 4,
                "unsearched": []}},
        ]
        result = aggregate_search_grounding(cases)
        assert result["cases"] == 3
        assert result["cases_fully_grounded"] == 1
        assert result["researchers_grounded"] == 9
        assert result["researchers_total"] == 12
        assert result["unobserved_by_agent"] == {}
        # Ordered most-frequent first.
        assert list(result["unsearched_by_agent"]) == [
            "author_researcher",
            "city_researcher",
        ]

    def test_an_unobserved_case_is_not_fully_grounded(self):
        """The roll-up carried the same inversion as the per-case metric.

        "nothing in `unsearched`" is true of a case whose ledger saw nobody, so
        the dataset line reported it as a fully grounded case. Reading
        `grounded == total` instead makes the two states differ.
        """
        from evaluation.tools.run_scheduled_eval import aggregate_search_grounding

        cases = [
            {"search_grounding": {
                "researchers_total": 4, "researchers_grounded": 0,
                "unsearched": [], "unobserved": [
                    "author_researcher", "city_researcher",
                    "book_context_researcher", "landmark_researcher"]}},
            {"search_grounding": {
                "researchers_total": 4, "researchers_grounded": 4,
                "unsearched": [], "unobserved": []}},
        ]
        result = aggregate_search_grounding(cases)
        # The control sits in the same row: the genuinely grounded case still
        # counts, so this is not a change that simply reports less.
        assert result["cases_fully_grounded"] == 1
        assert result["unobserved_by_agent"]["city_researcher"] == 1
        assert result["unsearched_by_agent"] == {}

    def test_older_results_files_without_unobserved_still_aggregate(self):
        """Both shapes carry the two counts, so the new rule reads either."""
        from evaluation.tools.run_scheduled_eval import aggregate_search_grounding

        cases = [
            {"search_grounding": {
                "researchers_total": 4, "researchers_grounded": 4,
                "unsearched": []}},
            {"search_grounding": {
                "researchers_total": 4, "researchers_grounded": 2,
                "unsearched": ["city_researcher", "author_researcher"]}},
        ]
        result = aggregate_search_grounding(cases)
        assert result["cases_fully_grounded"] == 1
        assert result["unobserved_by_agent"] == {}

    def test_reporting_never_fails_the_run(self):
        """No pass rule yet — a fully-unsearched run must not be a failure."""
        from evaluation.tools.run_scheduled_eval import (
            aggregate_search_grounding,
            dataset_failure_reason,
        )

        case = {
            "status": "evaluated",
            "scores": {"overall": 4.0},
            "search_grounding": {
                "researchers_total": 4, "researchers_grounded": 0,
                "unsearched": ["city_researcher"]},
        }
        result = {
            "dataset_name": "d", "total_cases": 1, "evaluated_cases": 1,
            "failed_cases": 0, "skipped_cases": 0, "placeholder_cases": 0,
            "case_results": [case],
            "search_grounding": aggregate_search_grounding([case]),
        }
        assert dataset_failure_reason(result) is None


# --------------------------------------------------------------------------
# r2 (Codex P2): the eval must score what production would SHIP.
#
# The scheduled eval extracts the composer's JSON by hand and never runs
# extract_itinerary_from_response, which is where production demotes claims no
# searched researcher supports. So the judge scored literal/historical stops
# the product would have downgraded — in exactly the unsearched-researcher
# scenario this eval exists to measure.
# --------------------------------------------------------------------------

class TestProductionGroundingDowngradeInEval:
    def _itinerary(self):
        return {
            "itinerary": {
                "cities": [
                    {
                        "name": "Dublin",
                        "stops": [
                            {
                                "name": "Personal Office",
                                "match_type": "literal",
                                "grounding_source": "invented",
                            }
                        ],
                    }
                ]
            },
            "suggestions": [],
        }

    def _state(self, unverified):
        state = {
            SessionStateKeys.CITY_DISCOVERY: {"cities": [{"name": "Dublin"}]},
            SessionStateKeys.AUTHOR_SITES: {"author_sites": [{"name": "Personal Office"}]},
            SessionStateKeys.UNVERIFIED_DISCOVERY: list(unverified),
        }
        return SessionStateAccessor(state)

    def test_unsupported_claim_is_demoted_before_scoring(self):
        from evaluation.tools.run_scheduled_eval import apply_production_grounding_downgrade

        data = apply_production_grounding_downgrade(
            self._itinerary(), self._state([SessionStateKeys.AUTHOR_SITES])
        )
        stop = data["itinerary"]["cities"][0]["stops"][0]
        assert stop["match_type"] == "vibe"
        assert stop["grounding_source"] is None

    def test_nothing_searched_still_fails_closed(self):
        from evaluation.tools.run_scheduled_eval import apply_production_grounding_downgrade

        data = apply_production_grounding_downgrade(
            self._itinerary(),
            self._state([SessionStateKeys.AUTHOR_SITES, SessionStateKeys.CITY_DISCOVERY]),
        )
        assert data["itinerary"]["cities"][0]["stops"][0]["match_type"] == "vibe"

    def test_a_supported_claim_survives(self):
        """Control: this must not blanket-demote, or the eval measures nothing."""
        from evaluation.tools.run_scheduled_eval import apply_production_grounding_downgrade

        data = self._itinerary()
        data["itinerary"]["cities"][0]["stops"][0]["name"] = "Dublin"
        data = apply_production_grounding_downgrade(data, self._state([]))
        assert data["itinerary"]["cities"][0]["stops"][0]["match_type"] == "literal"

    def test_bare_itinerary_shape_is_handled_too(self):
        from evaluation.tools.run_scheduled_eval import apply_production_grounding_downgrade

        bare = self._itinerary()["itinerary"]
        data = apply_production_grounding_downgrade(
            bare, self._state([SessionStateKeys.AUTHOR_SITES])
        )
        assert data["cities"][0]["stops"][0]["match_type"] == "vibe"

    def test_none_and_junk_are_returned_unchanged(self):
        from evaluation.tools.run_scheduled_eval import apply_production_grounding_downgrade

        assert apply_production_grounding_downgrade(None, self._state([])) is None
        assert apply_production_grounding_downgrade("nope", self._state([])) == "nope"

    def test_an_unrecognised_dict_shape_is_reported_not_silently_skipped(
        self, monkeypatch
    ):
        """The silent no-op Codex's finding named, surviving inside its own fix.

        A dict that is neither an envelope nor a bare itinerary reached the
        downgrade, iterated ``.get("cities") or []`` over nothing, and returned
        untouched — so the results file could not tell "gated, nothing to
        change" from "shape unrecognised, nothing gated". Exactly the class
        this function exists to close.
        """
        from evaluation.tools import run_scheduled_eval as mod

        seen = []
        monkeypatch.setattr(
            mod.logger, "warning", lambda event, **kw: seen.append(event)
        )
        weird = {"trip": {"days": []}}
        assert mod.apply_production_grounding_downgrade(weird, self._state([])) is weird
        assert seen == ["eval_grounding_downgrade_shape_unrecognised"]

    def test_a_recognised_shape_does_not_warn(self, monkeypatch):
        """Control. A warning that fires on a good shape misleads just as much."""
        from evaluation.tools import run_scheduled_eval as mod

        seen = []
        monkeypatch.setattr(
            mod.logger, "warning", lambda event, **kw: seen.append(event)
        )
        mod.apply_production_grounding_downgrade(
            self._itinerary(), self._state([SessionStateKeys.AUTHOR_SITES])
        )
        assert seen == []

    def test_a_session_with_no_verdict_fails_open_OUT_LOUD(self, monkeypatch):
        """Fail-open is right for the eval; silent fail-open is not.

        A state with no ``unverified_discovery`` key at all reads back as
        ``[]``, ``all_discovery_unverified`` is False, and every literal claim
        survives — indistinguishable from a run where all four researchers
        genuinely searched. The verdict's PRESENCE is the receipt.
        """
        from evaluation.tools import run_scheduled_eval as mod

        payloads = {
            SessionStateKeys.CITY_DISCOVERY: {"cities": [{"name": "Dublin"}]},
            SessionStateKeys.AUTHOR_SITES: {
                "author_sites": [{"name": "Personal Office"}]
            },
        }
        no_verdict = SessionStateAccessor(dict(payloads))
        assert no_verdict.discovery_verification_ran is False

        seen = []
        monkeypatch.setattr(
            mod.logger, "warning", lambda event, **kw: seen.append(event)
        )
        data = mod.apply_production_grounding_downgrade(self._itinerary(), no_verdict)

        assert seen == ["eval_grounding_downgrade_no_verdict"]
        # It really did fail open: with no verdict the author-sites payload
        # counts as evidence, so the claim resting on it survives at full
        # strength. That is the right default for an eval and the reason the
        # log has to exist.
        assert data["itinerary"]["cities"][0]["stops"][0]["match_type"] == "literal"

        # The discriminating half: the SAME payloads with a verdict naming
        # author_sites demote it. Identical state but for the receipt.
        with_verdict = dict(payloads)
        with_verdict[SessionStateKeys.UNVERIFIED_DISCOVERY] = [
            SessionStateKeys.AUTHOR_SITES
        ]
        seen.clear()
        demoted = mod.apply_production_grounding_downgrade(
            self._itinerary(), SessionStateAccessor(with_verdict)
        )
        assert seen == [], "a run carrying a verdict must not warn"
        assert demoted["itinerary"]["cities"][0]["stops"][0]["match_type"] == "vibe"

    def test_an_empty_verdict_is_a_verdict_and_stays_quiet(self):
        """Converse: `[]` means the pass ran and cleared everyone."""
        state = self._state([])
        assert state.discovery_verification_ran is True

    def test_unverified_keys_match_the_production_derivation(self):
        from core.executor import RESEARCHER_PAYLOAD_KEYS
        from evaluation.tools.run_scheduled_eval import unverified_payload_keys
        from plugins.langfuse_plugin import LangfusePlugin

        plugin = LangfusePlugin()
        # One researcher ran and searched; one ran and did not.
        searched, unsearched = list(RESEARCHER_PAYLOAD_KEYS)[:2]
        plugin._agents_seen.update({searched, unsearched})
        plugin._agents_searched.add(searched)

        assert unverified_payload_keys(plugin) == [RESEARCHER_PAYLOAD_KEYS[unsearched]]
        # An agent that never ran is not "unsearched" — same asymmetry as prod.
        assert RESEARCHER_PAYLOAD_KEYS[searched] not in unverified_payload_keys(plugin)
