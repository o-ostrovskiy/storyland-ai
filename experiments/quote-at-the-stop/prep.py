"""Turn the Gutenberg texts into the experiment's corpus.

    python -I prep.py books corpus            # 250-word chunks (the baseline)
    POC_CHUNK_WORDS=100 python -I prep.py books corpus   # also 100- and 500-word chunks, for the variants

For every book it writes three files into corpus/:

    <key>.txt          the plain text with the Gutenberg header and footer removed, and a
                       "[Title, Chapter N]" line wherever a chapter starts. run_full.py sends this whole
                       file as the prompt; grade.py uses it as the ground truth.
    <key>.json         the same text as a list of paragraphs, each with an id and its chapter.
                       run_agent.py searches and reads these paragraphs.
    <key>.chunks.json  paragraphs merged into chunks of about 250 words. The two RAG arms embed these.

Why chunks at all. RAG does not search whole books; it searches pieces, and the model only sees the pieces
that were found. How you cut the text is the first of the three choices every RAG system makes (the other
two: which embedding model, and how retrieval is configured). Cutting on paragraph boundaries keeps
sentences whole, and 250 words is roughly a paragraph or two: small enough that a chunk is about one thing,
big enough to keep some surrounding context. The variants (100 and 500 words) test whether that choice
matters here. There is no overlap between chunks in this experiment.

Chapter labels only exist so the model can answer "which chapter"; they take no part in retrieval.
"""
import json
import os
import re
import sys
from pathlib import Path

# Each book lists its heading patterns: (kind, label, regex with the number in group 1), kind "part" or
# "chapter". Headings are accepted as sequential runs (1, 2, 3, ...). When a pattern has several runs, the
# run that spans the most lines is the real one: Frankenstein and The Enchanted April list every chapter on
# a contents page before the text begins, and the first version of this script put the whole novel into
# the last chapter because of it. A "part" heading lets the chapter numbers restart (Madame Bovary).
BOOKS = {
    "hound": dict(file="pg2852.txt", title="The Hound of the Baskervilles", author="Arthur Conan Doyle",
                  headings=[("chapter", "Chapter", r"^Chapter (\d+)\.$")]),
    "jane": dict(file="pg1260.txt", title="Jane Eyre", author="Charlotte Brontë",
                 headings=[("chapter", "Chapter", r"^CHAPTER ([IVXL]+)$")]),
    "ulysses": dict(file="pg4300.txt", title="Ulysses", author="James Joyce",
                    headings=[("chapter", "Episode", r"^\[ (\d+) \]$")]),
    "frankenstein": dict(file="pg84.txt", title="Frankenstein", author="Mary Shelley",
                         headings=[("chapter", "Letter", r"^Letter (\d+)$"), ("chapter", "Chapter", r"^Chapter (\d+)$")]),
    "wuthering": dict(file="pg768.txt", title="Wuthering Heights", author="Emily Brontë",
                      headings=[("chapter", "Chapter", r"^CHAPTER ([IVXL]+)$")]),
    "bovary": dict(file="pg2413.txt", title="Madame Bovary", author="Gustave Flaubert",
                   headings=[("part", "Part", r"^Part ([IVX]+)$"), ("chapter", "Chapter", r"^Chapter (\w+)$")]),
    "april": dict(file="pg16389.txt", title="The Enchanted April", author="Elizabeth von Arnim",
                  headings=[("chapter", "Chapter", r"^Chapter (\d+)$")]),
    "sleepy": dict(file="pg41.txt", title="The Legend of Sleepy Hollow", author="Washington Irving",
                   headings=[]),  # no chapters: a 12,000-word story
    "odyssey": dict(file="pg1727.txt", title="The Odyssey", author="Homer (Samuel Butler's translation)",
                    headings=[("chapter", "Book", r"^BOOK ([IVXL]+)$")]),
    "grandet": dict(file="pg1715.txt", title="Eugénie Grandet", author="Honoré de Balzac",
                    headings=[("chapter", "Chapter", r"^([IVXL]+)$")]),
}

# Gutenberg wraps every text in a licence header and footer between these two markers.
START = re.compile(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK")
END = re.compile(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK")

CHUNK_WORDS = int(os.environ.get("POC_CHUNK_WORDS", "250"))
CHUNK_SUFFIX = "" if CHUNK_WORDS == 250 else f"{CHUNK_WORDS}"   # corpus/<key>.chunks100.json etc.

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
WORDS = {w: i + 1 for i, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}


def to_int(s):
    """Chapter numbers come as digits (Chapter 7), roman numerals (CHAPTER XII) or words (Chapter Nine)."""
    if s.isdigit():
        return int(s)
    if s.lower() in WORDS:
        return WORDS[s.lower()]
    if all(ch in ROMAN for ch in s):
        total, prev = 0, 0
        for ch in reversed(s):
            v = ROMAN[ch]
            total += -v if v < prev else v
            prev = max(prev, v)
        return total
    return -1


def strip_gutenberg(lines):
    """Keep only the lines between the START and END markers."""
    out, inside = [], False
    for line in lines:
        if not inside:
            if START.match(line):
                inside = True
            continue
        if END.match(line):
            break
        out.append(line.rstrip("\r\n"))
    return out


def accepted_headings(lines, headings):
    """Return {line_no: (kind, label, number)} for the headings that belong to each pattern's longest run."""
    pats = [(kind, label, re.compile(rx)) for kind, label, rx in headings]
    has_parts = any(kind == "part" for kind, _, _ in headings)
    found = {i: [] for i in range(len(pats))}
    for ln, line in enumerate(lines):
        s = line.strip()
        for i, (kind, label, rx) in enumerate(pats):
            m = rx.match(s)
            if m and to_int(m.group(1)) > 0:
                found[i].append((ln, to_int(m.group(1))))
                break
    out = {}
    for i, (kind, label, rx) in enumerate(pats):
        runs, run = [], []
        for ln, n in found[i]:
            restart = n == 1 and has_parts and kind == "chapter"
            if run and (n == run[-1][1] + 1 or restart):
                run.append((ln, n))
            else:
                if run:
                    runs.append(run)
                run = [(ln, n)] if n == 1 else []
        if run:
            runs.append(run)
        if not runs:
            continue
        best = max(runs, key=lambda r: r[-1][0] - r[0][0])   # the run that spans the most lines
        for ln, n in best:
            out[ln] = (kind, label, n)
    return out


def paragraphs(lines, headings):
    """Split the text into paragraphs (blank-line separated) and label each with its chapter."""
    marks = accepted_headings(lines, headings)
    paras, buf = [], []
    part, chapter = "", "front matter"

    def flush():
        if buf:
            text = " ".join(l.strip() for l in buf).strip()
            if text:
                label = f"{part}, {chapter}" if part and chapter != "front matter" else (part or chapter)
                paras.append({"id": len(paras), "chapter": label, "text": text})
            buf.clear()

    for ln, line in enumerate(lines):
        if ln in marks:
            flush()
            kind, label, n = marks[ln]
            if kind == "part":
                part, chapter = f"{label} {n}", "front matter"
            else:
                chapter = f"{label} {n}"
            continue
        if line.strip() == "":
            flush()
        else:
            buf.append(line)
    flush()
    return paras


def chunks(paras):
    """Merge consecutive paragraphs into chunks of up to CHUNK_WORDS words, never splitting a paragraph.
    A single paragraph longer than the limit becomes a chunk on its own."""
    out, cur, words = [], [], 0
    for p in paras:
        n = len(p["text"].split())
        if cur and words + n > CHUNK_WORDS:
            out.append(cur); cur, words = [], 0
        cur.append(p); words += n
    if cur:
        out.append(cur)
    return [{"id": i, "chapter": g[0]["chapter"], "para_ids": [p["id"] for p in g],
             "text": "\n\n".join(p["text"] for p in g)} for i, g in enumerate(out)]


def main(books_dir, out_dir):
    books_dir, out_dir = Path(books_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, meta in BOOKS.items():
        src = books_dir / meta["file"]
        if not src.exists():
            print(f"{key}: missing {src.name}, skipped")
            continue
        lines = strip_gutenberg(src.read_text(encoding="utf-8").splitlines(True))
        paras = paragraphs(lines, meta["headings"])
        ch = chunks(paras)
        words = sum(len(p["text"].split()) for p in paras)
        (out_dir / f"{key}.json").write_text(json.dumps(
            {"key": key, "title": meta["title"], "author": meta["author"], "paragraphs": paras}, ensure_ascii=False))
        (out_dir / f"{key}.txt").write_text("\n\n".join(
            (f"[{meta['title']}, {p['chapter']}]\n" if i == 0 or paras[i - 1]["chapter"] != p["chapter"] else "") + p["text"]
            for i, p in enumerate(paras)))
        (out_dir / f"{key}.chunks{CHUNK_SUFFIX}.json").write_text(json.dumps(ch, ensure_ascii=False))
        labels = [p["chapter"] for p in paras]
        front = labels.count("front matter")
        print(f"{key}: {words} words, {len(paras)} paragraphs ({front} in front matter), {len(ch)} chunks, "
              f"{len(set(labels)) - 1} chapter labels, last: {labels[-1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
