"""Arm D: hybrid RAG. The same chunks and embeddings as run_rag.py, plus BM25 keyword scoring;
the two rankings are merged with reciprocal rank fusion and the top K go into the prompt.

Variants, chosen with POC_VARIANT (results land in results/hybrid_<variant>.jsonl):
  c100 / c500  — chunks of ~100 / ~500 words instead of 250 (needs `POC_CHUNK_WORDS=100 prep.py` first)
  k3 / k10     — 3 or 10 chunks in the prompt instead of 5
  rerank       — the fused top 20 are re-ranked by the model itself, then the top 5 go into the prompt
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
RERANK_POOL = 20
RRF_K = 60


def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=np.float32)
        self.avg = float(self.len.mean())
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

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
    score = Counter()
    for r in rankings:
        for rank, i in enumerate(r):
            score[int(i)] += 1 / (RRF_K + rank + 1)
    return [i for i, _ in score.most_common()]


def rerank(c, q, book, chunks, pool):
    """Ask the model to order the pool by how likely each passage names the place. Returns (ids, usage)."""
    listing = "\n\n".join(f"[{i}] {chunks[i]['text'][:1200]}" for i in pool)
    prompt = (f"Below are {len(pool)} passages from {book['title']}. Rank them by how useful each is for quoting a sentence "
              f"that names the place \"{q['place']}\". Return only the passage numbers in square brackets, best first, "
              f"comma-separated, e.g. [12], [3], [7]. Include every number exactly once.\n\n{listing}")
    resp = with_retry(lambda: c.models.generate_content(model=MODEL, contents=prompt, config=gen_config()))
    order = [int(x) for x in re.findall(r"\[(\d+)\]", resp.text or "")]
    seen, ranked = set(), []
    for i in order + list(pool):
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
            idx[q["book"]] = index_for(c, q["book"], CHUNK_SUFFIX)
            bm[q["book"]] = BM25([ch["text"] for ch in idx[q["book"]][0]])
        chunks, vecs = idx[q["book"]]
        t0 = time.time()
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
