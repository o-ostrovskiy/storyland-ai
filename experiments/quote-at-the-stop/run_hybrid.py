"""Arm D: hybrid RAG. The same chunks and embeddings as run_rag.py, plus keyword scores; the two rankings are
merged, and the top chunks go into the prompt.

Why two kinds of search. Embeddings find meaning: "a tower by the sea" will find the Martello tower even
if the word "tower" never appears in the chunk. Keyword search finds exact words: "Glasnevin" is in the
chunk or it is not. Each is blind where the other sees, and the place names in this experiment are the
worst case for embeddings (run_rag.py missed three of them) and the best case for keywords. A hybrid runs
both and merges the lists, so a chunk that ranks high on either gets in.

    BM25   the keyword score, the one Elasticsearch and Lucene use by default: a chunk scores higher the
           more often it contains the query words, the rarer those words are across all chunks, and the
           shorter the chunk is. Computed locally in `BM25`, no API, no index file.
    RRF    reciprocal rank fusion, the simplest way to merge two rankings: each chunk gets
           1/(60 + its rank) from every list it appears in, and the sums are sorted. No weights to tune.

Variants, chosen with POC_VARIANT (results land in results/hybrid_<variant>.jsonl). Each turns one of the
knobs every RAG system has, to see whether it matters here:

    c100 / c500   chunks of ~100 / ~500 words instead of 250 (run `POC_CHUNK_WORDS=100 prep.py` first).
                  Smaller chunks are more precise and cheaper to send; bigger ones keep more context.
    k3 / k10      3 or 10 chunks in the prompt instead of 5. More chunks, more chances to find it, more
                  noise and tokens.
    rerank        a RERANKER: retrieval takes the 20 best chunks, then a second model call reads them all
                  with the question and reorders them; the top 5 of that order go in. Cheap on 20 chunks,
                  impossible on 5,000, which is why it sits after retrieval, not instead of it. Here the
                  same Gemini model is the reranker (a dedicated cross-encoder would be the usual choice).

What the results showed: hybrid fixed all three false "not mentioned" answers; the name was in the
retrieved chunks for all 68 questions, at every chunk size and every K. The knobs moved the bill 3–6×
(the reranker 4×) and the accuracy not at all: 66–67 of 68 in every variant, and the remaining miss was
always the same shape, the right chunk in the prompt and the neighbouring sentence quoted. That is a
prompting problem, not a retrieval one.
"""
import json
import math
import os
import re
import time
from collections import Counter

import numpy as np

from common import CORPUS, MODEL, TASK, client, done_ids, gen_config, load_book, load_questions, usage, with_retry, write_row
from run_rag import embed, index_for

VARIANT = os.environ.get("POC_VARIANT", "")
ARM = "hybrid" + (f"_{VARIANT}" if VARIANT else "")
CHUNK_SUFFIX = {"c100": "100", "c500": "500"}.get(VARIANT, "")
TOP_K = {"k3": 3, "k10": 10}.get(VARIANT, 5)
RERANK = VARIANT == "rerank"
RERANK_POOL = 20   # how many fused candidates the reranker reads
RRF_K = 60         # the usual constant in reciprocal rank fusion; damps the advantage of rank 1 over rank 2


def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    """Keyword scoring over the chunks of one book. Built once per book, in memory, in a fraction of a second."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b                       # standard BM25 constants: term-frequency saturation, length normalisation
        self.tf = [Counter(tokens(d)) for d in docs]  # term counts per chunk
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=np.float32)
        self.avg = float(self.len.mean())
        df = Counter()                                # in how many chunks each term appears
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}  # rare terms weigh more

    def scores(self, query):
        out = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(tokens(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def rrf(*rankings):
    """Merge rankings (lists of chunk ids, best first) with reciprocal rank fusion."""
    score = Counter()
    for r in rankings:
        for rank, i in enumerate(r):
            score[int(i)] += 1 / (RRF_K + rank + 1)
    return [i for i, _ in score.most_common()]


def rerank(c, q, book, chunks, pool):
    """Ask the model to order the pool by how likely each passage names the place. Returns (ids, usage).
    The tokens this call costs are added to the question's total, so the cost column stays honest."""
    listing = "\n\n".join(f"[{i}] {chunks[i]['text'][:1200]}" for i in pool)
    prompt = (f"Below are {len(pool)} passages from {book['title']}. Rank them by how useful each is for quoting a sentence "
              f"that names the place \"{q['place']}\". Return only the passage numbers in square brackets, best first, "
              f"comma-separated, e.g. [12], [3], [7]. Include every number exactly once.\n\n{listing}")
    resp = with_retry(lambda: c.models.generate_content(model=MODEL, contents=prompt, config=gen_config()))
    order = [int(x) for x in re.findall(r"\[(\d+)\]", resp.text or "")]
    seen, ranked = set(), []
    for i in order + list(pool):   # whatever the model forgot keeps its original order at the end
        if i in pool and i not in seen:
            seen.add(i); ranked.append(i)
    return ranked, usage(resp)


def main():
    c = client()
    done = done_ids(ARM)
    idx, bm = {}, {}
    for q in load_questions():
        if q["id"] in done:
            continue
        book = load_book(q["book"])
        if q["book"] not in idx:
            idx[q["book"]] = index_for(c, q["book"], CHUNK_SUFFIX)       # embeddings, shared with run_rag.py
            bm[q["book"]] = BM25([ch["text"] for ch in idx[q["book"]][0]])  # keywords, built here
        chunks, vecs = idx[q["book"]]
        t0 = time.time()
        # Two rankings: by meaning (the question sentence) and by words (just the place name, the exact thing we need).
        query = f"the place {q['place']} in {book['title']}"
        qv = embed(c, [query], "RETRIEVAL_QUERY")[0]
        dense = np.argsort(-(vecs @ qv))[:20]
        sparse = np.argsort(-bm[q["book"]].scores(q["place"]))[:20]
        fused = rrf(dense, sparse)
        extra = {"prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
        if RERANK:
            fused, extra = rerank(c, q, book, chunks, fused[:RERANK_POOL])
        top = fused[:TOP_K]
        passages = "\n\n".join(f"[chunk {i}, chapter {chunks[i]['chapter']}]\n{chunks[i]['text']}" for i in top)
        prompt = (
            f"Here are {TOP_K} passages retrieved from {book['title']} by {book['author']}.\n\n<passages>\n{passages}\n</passages>\n\n"
            + TASK.format(place=q["place"], title=book["title"], author=book["author"])
        )
        resp = with_retry(lambda: c.models.generate_content(model=MODEL, contents=prompt, config=gen_config()))
        dt = time.time() - t0
        u = usage(resp)
        for k in u:
            u[k] += extra[k]
        hit = any(q["place"].lower() in chunks[i]["text"].lower() for i in top)
        row = {"id": q["id"], "arm": ARM, "model": MODEL, "answer": resp.text, "latency_s": round(dt, 2),
               "retrieved": top, "retrieval_hit": hit, "variant": VARIANT, "top_k": TOP_K, "chunk_words": int(CHUNK_SUFFIX or 250), **u}
        write_row(ARM, row)
        print(f"{q['id']} hit={hit} {row['prompt_tokens']} tok {dt:.1f}s :: {(resp.text or '')[:90]!r}")


if __name__ == "__main__":
    main()
