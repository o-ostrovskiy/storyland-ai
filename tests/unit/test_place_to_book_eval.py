"""Unit tests for the place→book grounding eval scorer (run_place_to_book_eval.score_case)."""

from types import SimpleNamespace

from evaluation.tools.run_place_to_book_eval import score_case


def _cand(match_type, maps_to, title="T", author="A"):
    return SimpleNamespace(title=title, author=author, match_type=match_type, maps_to=maps_to)


def _result(found, candidates):
    return SimpleNamespace(found=found, candidates=candidates)


def test_found_passes_with_enough_grounded_literals():
    result = _result(
        True, [_cand("literal", "Baixa, Lisbon", title="T1"), _cand("vibe", None, title="T2")]
    )
    scored = score_case(result, expect="found", min_literal=1)
    assert scored["pass"] is True
    assert scored["n_literal_grounded"] == 1
    assert scored["found_classification"] == 1.0
    assert scored["grounding_clean"] is True


def test_found_fails_when_below_min_literal():
    # Only vibe candidates → zero grounded literals, fails min_literal=1.
    result = _result(True, [_cand("vibe", None), _cand("vibe", None)])
    scored = score_case(result, expect="found", min_literal=1)
    assert scored["pass"] is False
    assert scored["n_literal_grounded"] == 0


def test_literal_without_maps_to_does_not_count_and_flags_unclean():
    result = _result(True, [_cand("literal", None)])
    scored = score_case(result, expect="found", min_literal=1)
    assert scored["n_literal_grounded"] == 0
    assert scored["grounding_clean"] is False
    assert scored["pass"] is False


def test_not_found_passes_on_clean_empty():
    result = _result(False, [])
    scored = score_case(result, expect="not_found", min_literal=None)
    assert scored["pass"] is True
    assert scored["found_classification"] == 1.0


def test_not_found_fails_when_place_returns_candidates():
    # Wakanda-style: a fictional place that surfaces vibe candidates instead of
    # the clean not-found state. No fabrication (grounding_clean), but the
    # not-found classification is wrong → the case must fail.
    result = _result(True, [_cand("vibe", None), _cand("vibe", None)])
    scored = score_case(result, expect="not_found", min_literal=None)
    assert scored["pass"] is False
    assert scored["found_classification"] == 0.0
    assert scored["grounding_clean"] is True


# ai-private#71: repeated titles and wrong authors fail a case.

def test_repeated_title_fails_the_case():
    result = _result(True, [
        _cand("literal", "Baixa", title="The Book of Disquiet", author="Fernando Pessoa"),
        _cand("literal", "Chiado", title="Book of Disquiet", author="Fernando Pessoa"),
    ])
    scored = score_case(result, expect="found", min_literal=1)
    assert scored["n_duplicate_titles"] == 1
    assert scored["pass"] is False


def test_distinct_titles_pass():
    result = _result(True, [
        _cand("literal", "Baixa", title="The Book of Disquiet", author="Fernando Pessoa"),
        _cand("literal", "Chiado", title="Night Train to Lisbon", author="Pascal Mercier"),
    ])
    scored = score_case(result, expect="found", min_literal=1)
    assert scored["n_duplicate_titles"] == 0
    assert scored["pass"] is True


def test_same_title_different_authors_are_not_repeats():
    # Two real books called "Paris": not a duplicate, the case passes.
    result = _result(True, [
        _cand("literal", "Île de la Cité", title="Paris", author="Edward Rutherfurd"),
        _cand("literal", "Montmartre", title="Paris", author="Julien Green"),
    ])
    scored = score_case(result, expect="found", min_literal=1)
    assert scored["n_duplicate_titles"] == 0
    assert scored["pass"] is True


def test_known_book_with_wrong_author_fails():
    result = _result(True, [
        _cand("literal", "Baixa", title="The Book of Disquiet", author="José Saramago"),
    ])
    scored = score_case(
        result, expect="found", min_literal=1,
        expected_examples=["The Book of Disquiet (Fernando Pessoa)"],
    )
    assert scored["n_author_mismatches"] == 1
    assert scored["pass"] is False


def test_known_book_with_right_author_passes_despite_formatting():
    result = _result(True, [
        _cand("literal", "Kyoto", title="the old capital", author="Yasunari  Kawabata"),
    ])
    scored = score_case(
        result, expect="found", min_literal=1,
        expected_examples=["The Old Capital (Yasunari Kawabata)"],
    )
    assert scored["n_author_mismatches"] == 0
    assert scored["pass"] is True


def test_author_accents_do_not_count_as_a_mismatch():
    """Codex P2 on #302: "Jose Saramago" is the same author as "José Saramago",
    whether the accent is precomposed (NFC) or a combining mark (NFD)."""
    import unicodedata

    nfd = unicodedata.normalize("NFD", "José Saramago")
    for author in ("Jose Saramago", nfd):
        result = _result(True, [
            _cand("literal", "Chiado", title="The Year of the Death of Ricardo Reis", author=author),
        ])
        scored = score_case(
            result, expect="found", min_literal=1,
            expected_examples=["The Year of the Death of Ricardo Reis (José Saramago)"],
        )
        assert scored["n_author_mismatches"] == 0, author
        assert scored["pass"] is True, author


def test_accented_and_plain_author_are_one_book_for_duplicates():
    result = _result(True, [
        _cand("literal", "Chiado", title="Blindness", author="José Saramago"),
        _cand("literal", "Baixa", title="Blindness", author="Jose Saramago"),
    ])
    scored = score_case(result, expect="found", min_literal=1)
    assert scored["n_duplicate_titles"] == 1


def test_unknown_books_are_not_author_checked():
    result = _result(True, [_cand("literal", "X", title="Some Other Book", author="Anyone")])
    scored = score_case(
        result, expect="found", min_literal=1,
        expected_examples=["The Old Capital (Yasunari Kawabata)"],
    )
    assert scored["n_author_mismatches"] == 0


def test_aggregate_metrics_means():
    from evaluation.tools.run_place_to_book_eval import aggregate_metrics

    good = score_case(_result(True, [_cand("literal", "B", title="A")]), "found", 1)
    bad = score_case(_result(True, [_cand("vibe", None)]), "found", 1)
    metrics = aggregate_metrics([good, bad])
    assert metrics["case_pass"] == 0.5
    assert metrics["grounding_clean"] == 1.0
    assert metrics["literal_grounded_mean"] == 0.5
    assert aggregate_metrics([]) == {}


# Luga, 2026-11-21: a small real town must come back non-empty.

def test_min_literal_zero_passes_on_any_grounded_book():
    # A small town whose only book is atmosphere-level still beats "not mapped".
    result = _result(True, [_cand("vibe", None, title="The Siege")])
    scored = score_case(result, expect="found", min_literal=0)
    assert scored["pass"] is True


def test_min_literal_zero_still_fails_an_empty_answer():
    scored = score_case(_result(False, []), expect="found", min_literal=0)
    assert scored["pass"] is False
    assert scored["found_classification"] == 0.0


def test_unset_min_literal_still_needs_one_literal():
    result = _result(True, [_cand("vibe", None)])
    assert score_case(result, expect="found", min_literal=None)["pass"] is False


def test_evalset_has_small_towns_found_and_invented_places_not_found():
    import json
    from pathlib import Path

    cases = {
        c["eval_id"]: c
        for c in json.loads(
            Path("evaluation/place_to_book_v1.evalset.json").read_text(encoding="utf-8")
        )["eval_cases"]
    }
    assert cases["p2b_small_town_luga"]["expect"] == "found"
    assert cases["p2b_small_town_luga"]["place"].startswith("Luga")
    assert cases["p2b_small_town_holcomb"]["expect"] == "found"
    for invented in ("p2b_notfound_wakanda", "p2b_notfound_macondo", "p2b_notfound_gibberish"):
        assert cases[invented]["expect"] == "not_found", invented


# --- ai-private#81: the --variant switch -----------------------------------------


def test_variant_stats_skip_cache_hits_and_count_wikidata_gaps():
    from evaluation.tools.run_place_to_book_eval import variant_stats

    cases = [
        {"latency_s": 4.0, "cached": False, "search_queries": 3, "wikidata_status": "found",
         "token_usage": {"total_tokens": 100, "cost_usd": 0.01}},
        {"latency_s": 9.0, "cached": False, "search_queries": 2, "wikidata_status": "not_found",
         "token_usage": {"total_tokens": 50, "cost_usd": 0.02}},
        {"latency_s": 0.01, "cached": True, "search_queries": 0, "wikidata_status": None},
        {"latency_s": 6.0, "cached": False, "search_queries": 1, "wikidata_status": "unavailable",
         "token_usage": None},
        {"item_id": "skipped", "status": "skipped"},
    ]
    stats = variant_stats(cases)
    assert stats["cases_run"] == 3 and stats["cases_cached"] == 1
    assert stats["latency_p50_s"] == 6.0 and stats["latency_p95_s"] == 9.0
    assert stats["search_calls"] == 6
    assert stats["wikidata_asked"] == 3 and stats["wikidata_nothing"] == 2
    assert stats["total_tokens"] == 150 and stats["cost_usd"] == 0.03


def test_variant_stats_on_a_search_only_run():
    from evaluation.tools.run_place_to_book_eval import variant_stats

    stats = variant_stats([{"latency_s": 2.0, "cached": False, "search_queries": 4, "wikidata_status": None}])
    assert stats["wikidata_asked"] == 0 and stats["wikidata_nothing"] == 0
    assert stats["total_tokens"] is None and stats["cost_usd"] is None
    assert variant_stats([])["latency_p50_s"] is None


def test_cli_offers_the_three_variants_and_defaults_to_search(monkeypatch):
    import evaluation.tools.run_place_to_book_eval as tool

    seen = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        return {"evaluated": 0, "passed": 0, "metrics": {}}

    monkeypatch.setattr(tool, "run", fake_run)
    monkeypatch.setattr(tool, "exit_with_verdict", lambda *a, **k: 0)
    for argv, expected in ((["prog"], "search"), (["prog", "--variant", "both"], "both")):
        monkeypatch.setattr("sys.argv", argv)
        try:
            tool.main()
        except SystemExit:
            pass
        assert seen["variant"] == expected
    assert tool.VARIANTS == ("search", "wikidata", "both")
