# Re-baseline under the merits-not-resemblance judge (MYS-586 items 2–3)

**Date:** 2026-07-21 · **Judge:** gemini-2.5-flash-lite (pinned), prompt as of this
branch (reference reframed as example + 3 criteria re-anchors) · **System under
test:** gemini-3.1-flash-lite (main's CI default) · **Runs:** CI dispatches
29792373877 / 29792389935, 18 cases each (storyland_eval 8 + books_v1 10),
same config, per the two-run noise protocol.

Label caveat (carried from the calibration): comparisons against "labels" below
are against **Claude Fable 5 model labels (Olga-approved)** — a second model's
careful reading, not human ground truth. Read "bias" as *divergence from a
second model*, not as error against truth.

## Run results

| Dataset | Run 1 avg | Run 2 avg | Pooled | maxΔ (this pair) | tokens/case |
|---|---:|---:|---:|---:|---:|
| storyland_eval | 4.362 | 4.379 | **4.37** | 0.017 | ~18.0k |
| books_v1 | 3.024 | 2.861 | **2.94** | 0.163 | ~17.1k |

Per-shape (books_v1): with_preferences 3.37 / 3.60, without_preferences 2.68 / 2.12
— the preference-free (prod-shape) cells run markedly lower and noisier; worth its
own look before any per-shape gate hardens.

## Gates (rule: pooled − 2×maxΔ)

maxΔ from a single run-pair is an optimistic noise estimate (storyland's 0.017
here vs 0.17 historically measured). Two derivations recorded; **recommend the
conservative one** until the weekly runs accumulate more same-config pairs:

| Dataset | Pooled | Gate (this pair's maxΔ) | Gate (conservative: historical maxΔ 0.17 / 0.40) |
|---|---:|---:|---:|
| storyland_eval | 4.37 | 4.34 | **≥ 4.03** |
| books_v1 | 2.94 | 2.62 | **≥ 2.14** |

These supersede the interim constants (storyland ≥ 4.10 on pooled 4.44 under the
old judge + old system model; books ≥ 2.08 on pooled 2.88). Both judge and
system model changed — no delta against pre-fix numbers is meaningful.

## mechanism:

- **Dimension mix (books_v1, run1→run2):** book_relevance 2.8→2.8,
  preference_adherence 3.4→3.4, completeness 2.9→2.8, actionability 2.9→2.7,
  geographical_accuracy 3.1→3.0, engagement 3.4→3.1. No single-dimension
  collapse; the run-to-run spread is level, not mix.
- **Token profile:** 16.9k–17.2k total/case books, 17.7k–18.3k storyland —
  consistent across runs, no truncation signature.
- **Level did NOT lift on books_v1** (2.94 vs the old 2.88 band) despite the
  prompt fix. The controlled experiment below explains why: the fix as shipped
  removes only the smaller of two similarity mechanisms.

## Controlled re-judge: same 30 labeled itineraries, judge variants

Isolates the judge change from the generation change (new CI runs judge *new*
3.1-flash-lite generations; this section re-judges the exact payloads the labels
were made on). storyland_eval under the new prompt: **+0.18 / MAD 0.32 —
unchanged**, the control holds and the criteria re-anchors did no damage.

books_v1 (20 items, 110 dimension-pairs per config):

| Judge config | Bias (judge − label) | MAD |
|---|---:|---:|
| Old prompt (reference-as-benchmark + quality_criteria) | −1.16 | 1.27 |
| New prompt (reference-as-example + quality_criteria) — **shipped in this PR** | −0.95 | 1.16 |
| Ablation A: reference removed, quality_criteria kept | −0.72 | 0.95 |
| Ablation B: quality_criteria removed, reference kept (example framing) | **+0.07** | **0.45** |

**Finding: the dominant similarity mechanism is the book-specific
`quality_criteria` injection, not the reference output.** The benchmark framing
was worth ~0.2 of the ~1.35 divergence gap; the reference's mere presence
another ~0.2; the per-dimension "Book-specific requirement:" blocks carry the
rest. With them removed, books_v1 agreement (+0.07 / 0.45) essentially matches
reference-free storyland_eval — the qc blocks are a reference in checklist
form, grading compliance with expected specifics rather than quality.

## Decision needed (not taken in this PR)

What should books_v1's judge measure? Options, in increasing order of change:

1. **Keep qc as-is** — books_v1 stays a compliance gate; gates above stand;
   never read its deltas as quality.
2. **Reframe qc** in `_criterion_block` as "aspects worth considering, not
   requirements" — partial, would need its own ablation.
3. **Split the quantities** — score quality without qc (ablation-B config) and
   report qc coverage as a separate score (e.g. `criteria_coverage`), so
   quality and compliance stop sharing one number. Cleanest; touches the
   dataset's intent, so it's a dataset-owner call.

Until that call, the standing caveat holds: books_v1 judge deltas are a
catastrophe detector, not quality signal.
