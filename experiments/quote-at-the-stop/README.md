# Quote at the stop: whole novel vs. RAG vs. an agent with search

An experiment, not product code. It backs one My Storyland idea ([storyland-web#673](https://github.com/o-ostrovskiy/storyland-web/issues/673)):
at each stop of a trip, show the passage of the novel where that place is named. The question is where the
passage should come from: the whole novel in the model's context, a RAG index over chunks, or an agent that
searches the text and reads around the hit. Run on 2026-10-30; the write-up is
[I tried RAG on ten novels](https://olgaostrovskiy.com/notes/2026-10-30-do-we-still-need-rag/).
Nothing here is imported by the service; it shares only the repo's Python environment (`google-genai`, `numpy`).

## Setup

- **Books** (Project Gutenberg, public domain): every public-domain novel behind a live My Storyland trip, ten in
  all, 1.1M words: *The Legend of Sleepy Hollow* (12k words), *The Hound of the Baskervilles* (59k), *Eugénie
  Grandet* (66k), *Frankenstein* (75k), *The Enchanted April* (78k), *Madame Bovary* (116k), *Wuthering Heights*
  (116k), *The Odyssey* (130k, Butler's translation), *Jane Eyre* (185k), *Ulysses* (265k). The first run used only
  the three in bold sizes (Hound, Jane Eyre, Ulysses) and 20 questions; the numbers below are the full set.
- **Questions** (`questions.json`): 68 places, one question each — *does this place appear by name in the book;
  if yes, quote one verbatim passage and its chapter; if not, say so.* 50 places are in the text (Baskerville Hall,
  Thornfield, Davy Byrne's, Yonville, Ithaca…), 18 are real stops from the trips that the novel never names (Hound
  Tor, Haworth, Villa Diodati, Portofino, Top Withens, the James Joyce Centre…).
- **Model:** `gemini-3.1-flash-lite`, temperature 0, the same for every arm. Prices from the Gemini Developer API
  page on the run date: $0.25 / 1M input, $1.50 / 1M output, $0.15 / 1M embedding tokens.
- **Arms**
  - `run_full.py` — the whole novel in the prompt, one call per question (16k to 369k tokens, mean 148k).
  - `run_rag.py` — paragraphs merged into ~250-word chunks (5,053 chunks), embedded once with
    `gemini-embedding-001` (768 dims), cosine top-5 into the prompt.
  - `run_hybrid.py` — the same chunks and embeddings plus BM25 keyword scoring over the chunks; the two rankings
    are merged with reciprocal rank fusion (k=60), top-5 into the prompt. Nothing is paid twice: the embedding
    index is shared with `run_rag.py`, BM25 is computed locally.
  - `run_agent.py` — the model gets two tools, `search_book(query)` (case-insensitive substring over paragraphs,
    up to 8 hits) and `read_passage(paragraph_id)` (the paragraph and its neighbours), at most 8 tool calls.
- **Grading** (`grade.py`): the book is the ground truth, no judge model. A "present" answer is correct when it
  contains a passage of 15+ characters that appears verbatim in the novel (compared on letters and digits only)
  and names the place. An "absent" answer is correct when it says the book does not mention the place and quotes
  nothing that names it. Every verdict was read by eye after grading; the grader was loosened twice (short quotes,
  nested quote marks, `...` between two fragments, "(Book 4)" trailers, Butler's inline footnote numbers) and no
  model answer was touched.

## Results (10 books, 68 questions)

| arm | correct / 68 | wrong on "absent" (18) | mean input tokens | median latency | tool calls | cost, 68 q |
|---|---|---|---|---|---|---|
| whole novel | 61 | 0 | 147,557 | 1.5 s (max 39.7 s) | — | $2.52 |
| RAG, embeddings only, top-5 | 64 | 0 | 1,583 | 1.0 s | — | $0.03 (+22¢ once, the index) |
| hybrid RAG, BM25 + embeddings, top-5 | 66 | 0 | 1,555 | 1.1 s | — | $0.03 (same index) |
| agent, search + read | 67 | 0 | 1,646 | 1.7 s | 1.2 | $0.04 |

Per book, every answer and every verdict: `results/summary.md`, `results/graded.json`. The first run on three
books and 20 questions gave the same order: 16 · 17 · 19 · 20.

## What the failures were

- **Whole novel (7 wrong).** Never on the small books; five of the seven on the three biggest (*Jane Eyre*,
  *Ulysses*, *Madame Bovary*). Four times it quoted a real passage that does not name the place; twice it quoted a
  sentence that is not in the book (one was a real sentence with the pronouns changed: "He told me" became "I told
  him"); once, on *Ulysses*, it said "the text does not contain the word Sandycove" — the word is there seven
  times. Also the only slow calls: 10 and 40 seconds on *Ulysses*. And 75× the cost.
- **RAG, embeddings only (4 wrong).** Three times the five closest chunks did not contain the word (Sandycove and
  Glasnevin in *Ulysses*, Nantes in *Eugénie Grandet*), and the model honestly answered "not mentioned" — a false
  absence that a reader would never catch. Once it quoted the right house without its name. Embedding search alone
  is weak on proper nouns; the article's "use keyword search for exact names" is exactly this.
- **Hybrid RAG (2 wrong).** Adding BM25 keyword scores fixed all three false absences: every retrieval now contained
  the place name (68/68 retrieval hits). Both remaining misses are the same shape — the chunk with the name was in
  the prompt, and the model quoted the neighbouring sentence instead of the one that names it. A retrieval problem
  became a prompting problem.
- **Agent (1 wrong).** 1.2 tool calls on average: one exact-match search, then it quoted from the hit. Its win is
  mostly the search tool, not the autonomy: exact match finds a proper noun that embeddings miss, and reading the
  paragraph around the hit gives a quotable sentence. Its one miss is instructive: *Madame Bovary*'s inn is
  "Lion d’Or" with a typographic apostrophe; the agent searched `Lion d'Or` with a straight one, got zero hits twice,
  spent six calls on "Lion", "inn" and "Golden Lion", and quoted a paragraph that never names it. Exact match is
  exact. (A product search tool should fold quotes, apostrophes and accents before matching.) It was also the arm
  that found "Mer de Glace" is only ever called "the sea of ice" in *Frankenstein* — six calls, correct "absent".

## Three knobs, turned one at a time (hybrid RAG, same 68 questions)

`POC_VARIANT=c100|c500|k3|k10|rerank run_hybrid.py`; results in `results/hybrid_<variant>.jsonl`.

| variant | correct / 68 | mean input tokens | median latency | cost, 68 q | the miss(es) |
|---|---|---|---|---|---|
| baseline: 250-word chunks, top-5 | 66 | 1,555 | 1.1 s | $0.032 | h3, a5 |
| 100-word chunks (10,362 chunks) | 67 | 972 | 1.1 s | $0.022 | g5 |
| 500-word chunks (2,462 chunks) | 67 | 3,047 | 1.1 s | $0.057 | o4 |
| top-3 instead of top-5 | 67 | 990 | 1.1 s | $0.022 | s2 |
| top-10 | 66 | 2,989 | 1.1 s | $0.056 | h3, h5 |
| reranker: the model orders the fused top-20, top-5 go in | 67 | 6,682 | 2.0 s | $0.128 | h5 |

In every variant the place name was inside the retrieved chunks for all 50 "present" questions. Every miss, in
every variant, has the same shape: the right chunk was in the prompt and the model quoted the sentence next to
the one that names the place; and it is a different question each time. ±1 on 68 questions is noise. The knobs
moved the bill by 3–6× and the accuracy not at all; what is left is a prompting problem (ask for the sentence that
contains the name, then check it does), not a retrieval one.

## Caveats

- 68 questions, one model, one run each. Enough to see the shape, not to claim percentages.
- The questions are easy for keyword search by design: a place name either is in the text or is not. A question
  about an unnamed place ("the tower by the sea") would favour embeddings.
- Only public-domain books. Most trips on the site are for books still in copyright, where the feature cannot ship.

## Reproduce

From this folder, with the repo's virtualenv and `GOOGLE_API_KEY` in the environment:

```
./fetch_books.sh                      # ten Gutenberg texts into ./books (not committed)
python -I prep.py books corpus        # paragraphs, chunks and plain text per book (not committed)
python run_full.py && python run_rag.py && python run_hybrid.py && python run_agent.py
POC_VARIANT=c100 python run_hybrid.py # and c500 / k3 / k10 / rerank (c100 and c500 need POC_CHUNK_WORDS=100|500 prep.py first)
python grade.py                       # results/summary.md and results/graded.json
```

Runs are resumable: a question already in `results/<arm>.jsonl` is skipped, so `results/` as committed reproduces the
tables with `grade.py` alone. The embedding index (`emb/`, ~50 MB) is rebuilt on first use, about a minute and 22¢.
