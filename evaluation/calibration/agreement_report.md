# Judge vs human agreement

**Generated:** 2026-07-20 17:06
**Manifest:** evaluation/calibration/queue_manifest_2026-07.json (30 items, 30 labeled)

Bias = mean(judge − human): positive means the judge scores HIGHER than the human on that dimension.

| Dimension | n | Mean abs diff | Bias | \|Δ\|≥2 |
|-----------|---|---------------|------|--------|
| book_relevance | 29 | 1.172 | -0.897 | 12 |
| preference_adherence | 16 | 0.875 | -0.750 | 3 |
| completeness | 29 | 1.0 | -0.862 | 11 |
| actionability | 29 | 0.759 | -0.690 | 7 |
| geographical_accuracy | 29 | 1.241 | -0.759 | 10 |
| engagement | 30 | 0.7 | -0.500 | 6 |

## Large disagreements (|Δ| ≥ 2)

| Dimension | Case | Book | Judge | Human |
|-----------|------|------|-------|-------|
| book_relevance | storyland_eval/d48fbd3c | Under the Tuscan Sun | 5 | 3 |
| book_relevance | books_v1/query_011 | The Lord of the Rings | 3 | 5 |
| book_relevance | books_v1/query_012 | The Long Walk | 1 | 3 |
| book_relevance | books_v1/query_014 | Red, White & Royal Blue | 2 | 4 |
| book_relevance | books_v1/query_016 | The Kite Runner | 2 | 4 |
| book_relevance | books_v1/query_017 | Crime and Punishment | 2 | 4 |
| book_relevance | books_v1/query_020 | Shogun | 2 | 4 |
| book_relevance | books_v1/query_014 | Red, White & Royal Blue | 2 | 4 |
| book_relevance | books_v1/query_017 | Crime and Punishment | 2 | 5 |
| book_relevance | books_v1/query_018 | Wild | 2 | 4 |
| book_relevance | books_v1/query_020 | Shogun | 1 | 3 |
| book_relevance | books_v1/query_012 | The Long Walk | 2 | 5 |
| preference_adherence | books_v1/query_017 | Crime and Punishment | 1 | 4 |
| preference_adherence | books_v1/query_017 | Crime and Punishment | 2 | 4 |
| preference_adherence | books_v1/query_019 | Dracula | 2 | 4 |
| completeness | books_v1/query_012 | The Long Walk | 1 | 3 |
| completeness | books_v1/query_014 | Red, White & Royal Blue | 2 | 4 |
| completeness | books_v1/query_016 | The Kite Runner | 2 | 4 |
| completeness | books_v1/query_018 | Wild | 2 | 4 |
| completeness | books_v1/query_020 | Shogun | 2 | 4 |
| completeness | books_v1/query_014 | Red, White & Royal Blue | 2 | 4 |
| completeness | books_v1/query_017 | Crime and Punishment | 2 | 4 |
| completeness | books_v1/query_018 | Wild | 2 | 4 |
| completeness | books_v1/query_019 | Dracula | 2 | 4 |
| completeness | books_v1/query_020 | Shogun | 1 | 3 |
| completeness | books_v1/query_012 | The Long Walk | 2 | 4 |
| actionability | books_v1/query_014 | Red, White & Royal Blue | 2 | 4 |
| actionability | books_v1/query_018 | Wild | 2 | 4 |
| actionability | books_v1/query_014 | Red, White & Royal Blue | 2 | 4 |
| actionability | books_v1/query_017 | Crime and Punishment | 2 | 4 |
| actionability | books_v1/query_018 | Wild | 2 | 4 |
| actionability | books_v1/query_019 | Dracula | 2 | 4 |
| actionability | books_v1/query_020 | Shogun | 1 | 3 |
| geographical_accuracy | books_v1/query_011 | The Lord of the Rings | 3 | 5 |
| geographical_accuracy | books_v1/query_012 | The Long Walk | 1 | 3 |
| geographical_accuracy | books_v1/query_013 | Leviathan Wakes | 5 | 2 |
| geographical_accuracy | books_v1/query_016 | The Kite Runner | 2 | 5 |
| geographical_accuracy | books_v1/query_017 | Crime and Punishment | 1 | 4 |
| geographical_accuracy | books_v1/query_016 | The Kite Runner | 3 | 5 |
| geographical_accuracy | books_v1/query_017 | Crime and Punishment | 2 | 5 |
| geographical_accuracy | books_v1/query_019 | Dracula | 3 | 5 |
| geographical_accuracy | books_v1/query_020 | Shogun | 1 | 4 |
| geographical_accuracy | books_v1/query_012 | The Long Walk | 1 | 4 |
| engagement | books_v1/query_014 | Red, White & Royal Blue | 2 | 4 |
| engagement | books_v1/query_017 | Crime and Punishment | 2 | 4 |
| engagement | books_v1/query_020 | Shogun | 2 | 4 |
| engagement | books_v1/query_017 | Crime and Punishment | 2 | 4 |
| engagement | books_v1/query_020 | Shogun | 1 | 3 |
| engagement | books_v1/query_012 | The Long Walk | 2 | 4 |

---

## Addendum: mechanism + re-anchoring proposals (2026-07-20)

**Label provenance:** these are MODEL labels (Claude Fable 5, Olga-approved in lieu
of hand labels) — the analysis measures judge-vs-Claude, not judge-vs-human.

### Mechanism: the harshness is entirely the books_v1 reference comparison

| Dataset | Bias (judge−label) | MAD | Judge prompt difference |
|---------|-------------------:|----:|-------------------------|
| storyland_eval | **+0.18** | 0.32 | no expected_output, no quality_criteria |
| books_v1 | **−1.16** | 1.27 | REFERENCE OUTPUT "use as benchmark" + book-specific criteria |

The judge agrees with a careful reading almost perfectly when it scores the
itinerary on its own merits, and turns into a reference-similarity metric when
`_build_scoring_prompt` injects the expected_output as a "benchmark". This one
prompt difference explains (a) the storyland 4.4 vs books 2.9 dataset gap and
(b) much of books_v1's ±0.40 run noise (similarity-to-reference is brittle).

Secondary finding: gemini-2.5-flash-lite cannot fact-check geography — it gave
5 to a Leviathan Wakes generation that puts "Portland Observatory" in Oregon
(it is in Portland, Maine; label 2), while giving 1–2 to Crime and Punishment
generations whose canonical St. Petersburg addresses all check out.
geographical_accuracy is the least trustworthy dimension (MAD 1.24).

### Proposed prompt/rubric edits (NOT applied — changes the pinned judge, needs re-baseline)

1. `_build_scoring_prompt` reference section: replace "**REFERENCE OUTPUT**
   (use this as benchmark when scoring)" with "**REFERENCE EXAMPLE** — one
   valid solution, for context only. Score the generated itinerary on its own
   merits against the criteria; do NOT penalize valid choices that differ from
   this example."
2. `geographical_accuracy` criterion: append "Judge the itinerary's own
   locations for real-world validity. A location is not wrong merely because
   it differs from a reference. Invented stop labels that are not real
   visitable places should lower the score."
3. `book_relevance` criterion: append "Evaluate connections against your
   knowledge of the book itself, not against any reference list."
4. `completeness` criterion: append "Completeness = the itinerary's own
   components and plannable detail, not coverage of every reference location."

### Gate arithmetic anchored to the labels (this 30-item sample)

Agreement weights (1/MAD normalized): engagement .218, actionability .201,
preference_adherence .175, completeness .153, book_relevance .130,
geographical_accuracy .123.

| Dataset | Judge-space mean | Agreement-weighted | Bias-corrected (label scale) | Gate (pooled−2×maxΔ) |
|---------|-----------------:|-------------------:|-----------------------------:|---------------------:|
| storyland_eval | 4.41 | 4.34 | 4.23 | 4.07 flat / 4.00 weighted |
| books_v1 | 2.64 | 2.64 | **3.80** | 1.84 flat / 1.84 weighted |

Reading: the storyland gate is measuring what it claims (keep ≈4.1).
Re-weighting does NOT fix books_v1 (2.64 either way) because the harshness is
uniform across dimensions — the mechanism is the reference comparison, not
dimension mix. A books_v1 judge score of 2.6 corresponds to ≈3.8 actual
quality; its gate is a reference-similarity floor, not a quality floor.
Recommendation: apply edit #1, re-baseline books_v1 (2 runs per the noise
protocol), then re-derive its gate — fold into PR4 step zero, which re-derives
gates anyway. Until then treat books_v1 deltas as similarity drift.
