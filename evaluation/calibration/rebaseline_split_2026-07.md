# Re-baseline under the quality/criteria_coverage split (MYS-586, final)

**Date:** 2026-07-21 · **Judge:** gemini-2.5-flash-lite (pinned), quality prompt
WITHOUT book-specific criteria (ablation-B config); compliance scored
separately as `criteria_coverage` · **System under test:**
gemini-3.1-flash-lite · **Runs:** CI dispatches 29793963847 / 29793981815,
18 cases each, same config, two-run noise protocol.

## Results

| Dataset | Run 1 | Run 2 | Pooled | pairΔ | criteria_coverage |
|---|---:|---:|---:|---:|---:|
| storyland_eval | 4.357 | 4.317 | **4.34** | 0.040 | n/a (no criteria) |
| books_v1 | 3.897 | 3.911 | **3.90** | 0.014 | **2.4 / 2.3** |

The judge-history arc on books_v1: 2.88 ±0.40 (compliance-blended) → 2.94
±0.16 (reference reframed) → **3.90 ±0.014** (compliance split out). The
level landed inside the ablation's predicted range (labels averaged 3.7–3.9
on the older generations), and the run-to-run noise collapsed by ~30×:
**the compliance mechanism was not only depressing the level — it was the
dominant noise source.** Similarity-to-a-reference is brittle; merits
scoring is stable.

`criteria_coverage` now says its own thing in its own column: current
generations satisfy few of the dataset's book-specific criteria (~2.3–2.4).
That is a real, visible compliance statement — previously it was silently
averaged into "quality."

## Per-shape (books_v1) — revises the MYS-560 support

| Shape | Run 1 | Run 2 |
|---|---:|---:|
| with_preferences | 4.03 | 4.10 |
| without_preferences | 3.76 | 3.72 |

Under the compliance-blended judge the preference-free (prod-shape) cells ran
~1.2 lower (2.68/2.12 vs 3.37/3.60). Under the split judge the gap is
**~0.3**. Most of the "prod shape scores a point lower and twice as noisy"
signal was the compliance mechanism, not a quality gap. A ~0.3 deficit on
n=5/run remains worth watching, but MYS-560 should quote this number, not the
pre-split one.

## Gates

Tight derivation (this pair's maxΔ): storyland ≥ 4.26, books ≥ 3.88. A
single pair's Δ is an optimistic noise estimate, so interim conservative
constants, flagged as judgment: **storyland ≥ 4.00** (historical maxΔ 0.17),
**books_v1 ≥ 3.55** (borrowing 0.17 as a cross-dataset proxy — the old books
maxΔ 0.40 was similarity noise that no longer exists; no post-split history
of its own yet). Re-derive both after ~4 weekly same-config pairs accumulate.
Both gates remain catastrophe detectors — pass means "no catastrophe."

## mechanism:

- Dimension mix (books, run1→run2): br 3.5→3.7, pa 4.0→4.2, co 3.9→3.9,
  ac 4.0→3.9, ga 4.2→4.1, en 3.9→3.9 — level shift is broad-based vs the
  pre-split runs (every dimension up 0.7–1.2), which is the expected
  signature of removing a global penalty, not of one dimension moving.
- Token profile: 16.6k–17.4k/case books, 16.9k–19.7k storyland — consistent
  with prior runs; no truncation signature. criteria_coverage adds one small
  judge call per books case (~1–2k tokens).
- Trust status: the quality judge for books_v1 now runs the exact config that
  agreed with the second-model reading at +0.07 bias / 0.45 MAD on the 30
  labeled itineraries. books_v1 quality deltas are now as believable as
  storyland's — same caveat as always that the anchor is a second model's
  reading, with human anchors accumulating via the weekly spot-checks.
