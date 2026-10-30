"""Turn Gutenberg texts into the experiment corpus.

Usage: python -I prep.py <books_dir> <out_dir>

Writes, per book, <out_dir>/<key>.json with the paragraphs (id, chapter, text),
<out_dir>/<key>.txt (the stripped full text) and <out_dir>/<key>.chunks.json
(paragraphs merged into ~250-word chunks for the RAG arm).
"""
import json
import os
import re
import sys
from pathlib import Path

# Each book lists its heading patterns: (kind, label, regex with the number in group 1), kind "part" or "chapter".
# Headings are accepted as sequential runs (1, 2, 3 …); when a pattern has several runs (a contents page lists every
# chapter before the text does) the run that spans the most lines is the real one. A "part" heading lets the
# chapter numbers restart.
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
                   headings=[]),
    "odyssey": dict(file="pg1727.txt", title="The Odyssey", author="Homer (Samuel Butler's translation)",
                    headings=[("chapter", "Book", r"^BOOK ([IVXL]+)$")]),
    "grandet": dict(file="pg1715.txt", title="Eugénie Grandet", author="Honoré de Balzac",
                    headings=[("chapter", "Chapter", r"^([IVXL]+)$")]),
}
START = re.compile(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK")
END = re.compile(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK")
CHUNK_WORDS = int(os.environ.get("POC_CHUNK_WORDS", "250"))
CHUNK_SUFFIX = "" if CHUNK_WORDS == 250 else f"{CHUNK_WORDS}"
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
WORDS = {w: i + 1 for i, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}


def to_int(s):
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
    part_lines = set()
    for ln, line in enumerate(lines):
        s = line.strip()
        for i, (kind, label, rx) in enumerate(pats):
            m = rx.match(s)
            if m and to_int(m.group(1)) > 0:
                found[i].append((ln, to_int(m.group(1))))
                if kind == "part":
                    part_lines.add(ln)
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
        best = max(runs, key=lambda r: r[-1][0] - r[0][0])
        for ln, n in best:
            out[ln] = (kind, label, n)
    return out


def paragraphs(lines, headings):
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
