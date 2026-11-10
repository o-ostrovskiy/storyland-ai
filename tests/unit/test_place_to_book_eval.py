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
