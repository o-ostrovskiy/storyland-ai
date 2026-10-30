"""Grade every arm against the book itself, and write results/summary.md.

    python grade.py        # needs corpus/ (prep.py) and results/*.jsonl; no API calls

Why the book is the judge. The usual way to grade LLM answers is a second model with a rubric. That judge
can be wrong too, and it would need its own checking. Here the question was chosen so that no judge is
needed: the model was asked for a *verbatim* quote that *names the place*. A quote is either in the novel
word for word or it is not, and it either contains the name or it does not. Both are string checks.

What counts as correct:

    the place IS in the book     the answer contains a passage of at least 15 characters that appears
                                 verbatim in the novel and contains the place name
    the place is NOT in the book the answer says the book does not mention it, and offers no quote that
                                 names it

What counts as wrong, with the reason recorded next to it:

    "says the place is absent; it is in the text"            a false "not mentioned" — the worst kind for a
                                                              reader, because nothing looks wrong
    "verbatim quote, but it does not name the place"         a real passage, the wrong one
    "no verbatim quote found in the text"                    a quote that is not in the book: made up, or a
                                                              real sentence with words changed
    "claims the place appears ..." (absent questions only)   a quote for a place the book never names

The comparison is on letters and spaces only (see `loose`): quote marks, dashes, ellipses and digits are
dropped from both sides before matching, so a curly quote inside a quoted passage, a "..." between two
fragments, or Butler's inline footnote numbers in The Odyssey ("Menelaus36") cannot fail a true quote. The
grader was loosened twice during the experiment for exactly those cases; every verdict was then read by
eye, and no model answer was ever edited.
"""
import json
import re
import statistics
import unicodedata

from common import CORPUS, RESULTS, load_questions

# Arms are discovered from the files present, in this order.
ORDER = ["full", "rag", "hybrid", "hybrid_c100", "hybrid_c500", "hybrid_k3", "hybrid_k10", "hybrid_rerank", "agent"]
ARMS = [a for a in ORDER if (RESULTS / f"{a}.jsonl").exists()]

# Gemini Developer API, standard tier, as published on 2026-10-30 (ai.google.dev/gemini-api/docs/pricing),
# USD per million tokens. Used only to turn token counts into the cost column.
PRICE_IN, PRICE_OUT, PRICE_EMBED = 0.25, 1.50, 0.15

# How an answer says "the book does not mention it". Judged as a move, not a string: any of these shapes.
ABSENT = re.compile(r"(does not|doesn't|never|not)\s+(appear|mention|name|contain|feature|reference)|no (mention|reference)|is not (mentioned|named)|isn't (mentioned|named)|not mentioned|not named", re.I)
# Labels the model puts around a quote ("Quote: ...", "Passage: ...", "Chapter 3: ...") and trailers after
# it ("(Chapter 7)", "(Book 4)"). Both are stripped before matching so the candidate span is the quote alone.
LABEL = re.compile(r"^\s*(\*\*)?(quote|passage|chapter\s+\w+|ch\.\s*\w+)\s*(\*\*)?\s*:\s*", re.I)
TRAIL = re.compile(r"\s*\(\s*(chapter|ch\.|episode|book|part|letter)[^)]*\)\s*[.]?\s*$", re.I)
MIN_QUOTE = 15   # characters; shorter spans ("said he.") match by accident


def cost(r):
    return (r["prompt_tokens"] * PRICE_IN + (r["output_tokens"] + r.get("thought_tokens", 0)) * PRICE_OUT) / 1e6


def norm(s):
    """Unicode-normalise and lowercase; straighten curly quotes and dashes; collapse whitespace."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", "-").replace("–", "-")
    s = re.sub(r"_", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def loose(s):
    """Letters and spaces only. This is what both the answer and the book are compared on."""
    return re.sub(r"[^a-z ]", "", norm(s)).strip()  # digits too: Butler's Odyssey carries footnote numbers inline


def spans(answer):
    """Candidate quoted spans: each line and each sentence of the answer, with labels, trailers and outer
    quotes stripped; a span joined with '...' is split into its pieces, each checked on its own."""
    out = []
    for raw in re.split(r"\n+", answer) + re.split(r"(?<=[.!?])\s+", answer):
        s = TRAIL.sub("", LABEL.sub("", raw)).strip().strip('"“”').strip()
        for piece in re.split(r"\s*(?:\.\.\.|…)\s*", s):
            piece = piece.strip().strip('"“”').strip()
            if len(piece) >= MIN_QUOTE:
                out.append(piece)
    return out


def grade(q, answer, book_text):
    """(verdict, reason) for one answer. `book_text` is the whole novel passed through `loose`."""
    answer = answer or ""
    place = norm(q["place"])
    quoted = [sp for sp in spans(answer) if loose(sp) in book_text]
    quoted_with_place = [sp for sp in quoted if place in norm(sp)]
    claims_absent = bool(ABSENT.search(answer))
    if q["present"]:
        if quoted_with_place:
            return "correct", "verbatim quote names the place"
        if claims_absent:
            return "wrong", "says the place is absent; it is in the text"
        if quoted:
            return "wrong", "verbatim quote, but it does not name the place"
        return "wrong", "no verbatim quote found in the text (made up or paraphrased)"
    if claims_absent and not quoted_with_place:
        return "correct", "says absent, no quote"
    return "wrong", "claims the place appears" + (" with a verbatim quote" if quoted else " with a quote not in the text")


def main():
    qs = {q["id"]: q for q in load_questions()}
    texts = {k: loose((CORPUS / f"{k}.txt").read_text()) for k in {q["book"] for q in qs.values()}}
    rows = {}
    for arm in ARMS:
        for line in (RESULTS / f"{arm}.jsonl").read_text().splitlines():
            r = json.loads(line)
            q = qs[r["id"]]
            r["verdict"], r["why"] = grade(q, r["answer"], texts[q["book"]])
            rows[(arm, r["id"])] = r

    books = list(dict.fromkeys(q["book"] for q in qs.values()))
    nq = len(qs)
    lines = ["# RAG vs. reading vs. agent — results", "",
             f"Model: `{next(iter(rows.values()))['model']}` for every arm. {nq} questions over {len(books)} books, "
             f"{sum(q['present'] for q in qs.values())} places named in the text, {sum(not q['present'] for q in qs.values())} not.", ""]
    lines += ["| arm | correct / " + str(nq) + " | " + " | ".join(f"{b} ({sum(q['book']==b for q in qs.values())})" for b in books)
              + f" | absent ({sum(not q['present'] for q in qs.values())}) | mean input tokens | median latency | mean tool calls | cost, {nq} questions |",
              "|---|---|" + "---|" * len(books) + "---|---|---|---|---|"]
    for arm in ARMS:
        rs = [rows[(arm, i)] for i in qs if (arm, i) in rows]
        if not rs:
            continue
        ok = lambda f: sum(r["verdict"] == "correct" for r in rs if f(qs[r["id"]]))  # noqa: E731
        n = lambda f: sum(1 for r in rs if f(qs[r["id"]]))  # noqa: E731
        calls = [r.get("n_calls", 0) for r in rs]
        per_book = " | ".join(f"{ok(lambda q, b=b: q['book']==b)}/{n(lambda q, b=b: q['book']==b)}" for b in books)
        lines.append(f"| {arm} | {ok(lambda q: True)} / {len(rs)} | {per_book} | {ok(lambda q: not q['present'])}/{n(lambda q: not q['present'])} | "
                     f"{statistics.mean(r['prompt_tokens'] for r in rs):,.0f} | {statistics.median(r['latency_s'] for r in rs):.1f}s | "
                     f"{statistics.mean(calls):.1f} | ${sum(cost(r) for r in rs):.3f} |")
    words = {k: len((CORPUS / f"{k}.txt").read_text().split()) for k in texts}
    lines += ["", f"RAG also pays once to embed the books: about {sum(words.values()):,} words ≈ {sum(words.values())*1.3/1e6*PRICE_EMBED*100:.0f}¢ at ${PRICE_EMBED}/1M tokens; after that the index is reused for every question.",
              f"Prices: ${PRICE_IN}/1M input, ${PRICE_OUT}/1M output (Gemini Developer API, standard tier, 2026-10-30)."]
    lines += ["", "## Per question", "", "| id | place | in text | " + " | ".join(ARMS) + " |", "|---|---|---|" + "---|" * len(ARMS)]
    for i, q in qs.items():
        cells = []
        for arm in ARMS:
            r = rows.get((arm, i))
            cells.append("—" if not r else ("✅" if r["verdict"] == "correct" else "❌ " + r["why"]))
        lines.append(f"| {i} | {q['place']} | {'yes' if q['present'] else 'no'} | " + " | ".join(cells) + " |")
    lines += ["", "## Tokens and time per question", "", "| id | " + " | ".join(f"{a} tokens / s" for a in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for i in qs:
        lines.append(f"| {i} | " + " | ".join(f"{rows[(a,i)]['prompt_tokens']:,} / {rows[(a,i)]['latency_s']}" if (a, i) in rows else "—" for a in ARMS) + " |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "graded.json").write_text(json.dumps(list(rows.values()), ensure_ascii=False, indent=1))
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
