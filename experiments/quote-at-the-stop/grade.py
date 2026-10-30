"""Grade the three arms against the text itself, and write results/summary.md.

A "present" question is correct when the answer contains a passage of at least 15 characters that appears
verbatim in the novel (compared on letters and digits only, so punctuation and quote marks cannot break a
match) and names the place. An "absent" question is correct when the answer says the book
does not mention the place and offers no quote that names it. No judge model: the book is the ground truth.
"""
import json
import re
import statistics
import unicodedata
from collections import defaultdict

from common import CORPUS, HERE, RESULTS, load_questions

ORDER = ["full", "rag", "hybrid", "hybrid_c100", "hybrid_c500", "hybrid_k3", "hybrid_k10", "hybrid_rerank", "agent"]
ARMS = [a for a in ORDER if (RESULTS / f"{a}.jsonl").exists()]
# Gemini Developer API, standard tier, 2026-10-30 (ai.google.dev/gemini-api/docs/pricing), USD per 1M tokens.
PRICE_IN, PRICE_OUT, PRICE_EMBED = 0.25, 1.50, 0.15


def cost(r):
    return (r["prompt_tokens"] * PRICE_IN + (r["output_tokens"] + r.get("thought_tokens", 0)) * PRICE_OUT) / 1e6
ABSENT = re.compile(r"(does not|doesn't|never|not)\s+(appear|mention|name|contain|feature|reference)|no (mention|reference)|is not (mentioned|named)|isn't (mentioned|named)|not mentioned|not named", re.I)


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", "-").replace("–", "-")
    s = re.sub(r"_", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


LABEL = re.compile(r"^\s*(\*\*)?(quote|passage|chapter\s+\w+|ch\.\s*\w+)\s*(\*\*)?\s*:\s*", re.I)
TRAIL = re.compile(r"\s*\(\s*(chapter|ch\.|episode|book|part|letter)[^)]*\)\s*[.]?\s*$", re.I)
MIN_QUOTE = 15


def loose(s):
    """Letters and spaces only: nested curly quotes, dashes and ellipses can't break a match."""
    return re.sub(r"[^a-z ]", "", norm(s)).strip()  # digits too: Butler's Odyssey carries footnote numbers inline


def spans(answer):
    """Candidate quoted spans: each line and each sentence of the answer, with 'Quote:' labels, '(Chapter 7)'
    trailers and outer quotes stripped; a span joined with '...' is split into its pieces."""
    out = []
    for raw in re.split(r"\n+", answer) + re.split(r"(?<=[.!?])\s+", answer):
        s = TRAIL.sub("", LABEL.sub("", raw)).strip().strip('"“”').strip()
        for piece in re.split(r"\s*(?:\.\.\.|…)\s*", s):
            piece = piece.strip().strip('"“”').strip()
            if len(piece) >= MIN_QUOTE:
                out.append(piece)
    return out


def grade(q, answer, book_text):
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
        p = RESULTS / f"{arm}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            q = qs[r["id"]]
            r["verdict"], r["why"] = grade(q, r["answer"], texts[q["book"]])
            rows[(arm, r["id"])] = r
    lines = ["# RAG vs. reading vs. agent — results", "", f"Model: `{next(iter(rows.values()))['model']}` for every arm. {len(qs)} questions over {len({q['book'] for q in qs.values()})} books, {sum(q['present'] for q in qs.values())} places named in the text, {sum(not q['present'] for q in qs.values())} not.", ""]
    books = sorted({q["book"] for q in qs.values()}, key=lambda b: list(dict.fromkeys(q["book"] for q in qs.values())).index(b))
    nq = len(qs)
    lines += ["| arm | correct / " + str(nq) + " | " + " | ".join(f"{b} ({sum(q['book']==b for q in qs.values())})" for b in books) + f" | absent ({sum(not q['present'] for q in qs.values())}) | mean input tokens | median latency | mean tool calls | cost, {nq} questions |",
              "|---|---|" + "---|" * len(books) + "---|---|---|---|---|"]
    per_arm = {}
    for arm in ARMS:
        rs = [rows[(arm, i)] for i in qs if (arm, i) in rows]
        if not rs:
            continue
        ok = lambda f: sum(r["verdict"] == "correct" for r in rs if f(qs[r["id"]]))  # noqa: E731
        n = lambda f: sum(1 for r in rs if f(qs[r["id"]]))  # noqa: E731
        per_arm[arm] = dict(correct=ok(lambda q: True), n=len(rs))
        calls = [r.get("n_calls", 0) for r in rs]
        per_book = " | ".join(f"{ok(lambda q, b=b: q['book']==b)}/{n(lambda q, b=b: q['book']==b)}" for b in books)
        lines.append(f"| {arm} | {ok(lambda q: True)} / {len(rs)} | {per_book} | {ok(lambda q: not q['present'])}/{n(lambda q: not q['present'])} | {statistics.mean(r['prompt_tokens'] for r in rs):,.0f} | {statistics.median(r['latency_s'] for r in rs):.1f}s | {statistics.mean(calls):.1f} | ${sum(cost(r) for r in rs):.3f} |")
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
