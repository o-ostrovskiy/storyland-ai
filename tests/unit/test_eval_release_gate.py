"""The paid eval runners must say whether a release is allowed, and exit non-zero when it is not (2026-11-19 audit:
a 0/14 place→book run exited 0)."""

import json
import sys

import pytest

from evaluation.tools import (
    run_expansion_eval,
    run_local_atmosphere_eval,
    run_place_to_book_eval,
)
from evaluation.tools.release_gate import exit_with_verdict, verdict


def _summary(total, evaluated, passed, **metrics):
    return {"total": total, "evaluated": evaluated, "passed": passed, "metrics": metrics}


def test_all_pass_is_allowed():
    assert verdict(_summary(14, 14, 14, grounding_clean=1.0), 0.85, ["grounding_clean"]) == (True, [])


def test_zero_of_fourteen_is_blocked():
    allowed, reasons = verdict(_summary(14, 14, 0, grounding_clean=1.0), 0.85, ["grounding_clean"])
    assert not allowed and "0/14" in reasons[0]


def test_nothing_evaluated_is_blocked():
    allowed, reasons = verdict(_summary(0, 0, 0), 0.85)
    assert not allowed and "no case evaluated" in reasons[0]


def test_skipped_case_is_blocked():
    allowed, reasons = verdict(_summary(14, 13, 13), 0.85)
    assert not allowed and "skipped" in reasons[0]


def test_hard_metric_below_one_is_blocked_even_at_high_pass_rate():
    allowed, reasons = verdict(_summary(14, 14, 13, grounding_clean=0.929), 0.85, ["grounding_clean"])
    assert not allowed and "grounding_clean" in reasons[0]


def test_one_known_miss_is_within_the_floor():
    assert verdict(_summary(11, 11, 10, grounding_clean=1.0), 0.85, ["grounding_clean"])[0]


def test_verdict_is_written_into_the_results_file(tmp_path, capsys):
    path = tmp_path / "r.json"
    s = _summary(14, 14, 0)
    path.write_text(json.dumps(s))
    assert exit_with_verdict(s, 0.85, results_path=path) == 1
    assert json.loads(path.read_text())["release_gate"]["allowed"] is False
    assert "RELEASE: BLOCKED" in capsys.readouterr().out


@pytest.mark.parametrize(
    "module", [run_place_to_book_eval, run_expansion_eval, run_local_atmosphere_eval]
)
@pytest.mark.parametrize("passed,code", [(0, 1), (14, 0)])
def test_runner_main_exit_code_follows_the_verdict(module, passed, code, monkeypatch, tmp_path):
    async def fake_run(**_kw):
        return {**_summary(14, 14, passed, grounding_clean=1.0), "_results_path": str(tmp_path / "x.json")}

    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["runner", "--no-register"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == code
