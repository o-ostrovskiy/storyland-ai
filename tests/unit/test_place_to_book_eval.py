"""Unit tests for the place→book grounding eval scorer (run_place_to_book_eval.score_case)."""

from types import SimpleNamespace

from evaluation.tools.run_place_to_book_eval import score_case


def _cand(match_type, maps_to, title="T", author="A"):
    return SimpleNamespace(title=title, author=author, match_type=match_type, maps_to=maps_to)


def _result(found, candidates):
    return SimpleNamespace(found=found, candidates=candidates)


def test_found_passes_with_enough_grounded_literals():
    result = _result(True, [_cand("literal", "Baixa, Lisbon"), _cand("vibe", None)])
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
