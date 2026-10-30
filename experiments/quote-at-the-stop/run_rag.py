"""Arm B: classic RAG. Chunks embedded once; per question the 5 closest chunks go into the prompt."""
import json
import time

import numpy as np
from google.genai import types

from common import CORPUS, EMBED_MODEL, HERE, MODEL, TASK, client, done_ids, gen_config, load_book, load_questions, usage, with_retry, write_row

ARM = "rag"
TOP_K = 5
EMB = HERE / "emb"
DIM = 768


def embed(c, texts, task_type):
    out = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        resp = with_retry(lambda: c.models.embed_content(
            model=EMBED_MODEL, contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=DIM)))
        out.extend(e.values for e in resp.embeddings)
    arr = np.array(out, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def index_for(c, key, suffix=""):
    EMB.mkdir(exist_ok=True)
    chunks = json.loads((CORPUS / f"{key}.chunks{suffix}.json").read_text())
    path = EMB / f"{key}{suffix}.npy"
    if path.exists():
        return chunks, np.load(path)
    t0 = time.time()
    vecs = embed(c, [ch["text"] for ch in chunks], "RETRIEVAL_DOCUMENT")
    np.save(path, vecs)
    print(f"indexed {key}: {len(chunks)} chunks in {time.time()-t0:.0f}s")
    (EMB / f"{key}{suffix}.index_time.json").write_text(json.dumps({"chunks": len(chunks), "seconds": round(time.time() - t0)}))
    return chunks, vecs


def main():
    c = client()
    done = done_ids(ARM)
    idx = {}
    for q in load_questions():
        if q["id"] in done:
            continue
        book = load_book(q["book"])
        if q["book"] not in idx:
            idx[q["book"]] = index_for(c, q["book"])
        chunks, vecs = idx[q["book"]]
        t0 = time.time()
        qv = embed(c, [f"the place {q['place']} in {book['title']}"], "RETRIEVAL_QUERY")[0]
        top = np.argsort(-(vecs @ qv))[:TOP_K]
        passages = "\n\n".join(f"[chunk {int(i)}, chapter {chunks[i]['chapter']}]\n{chunks[i]['text']}" for i in top)
        prompt = (
            f"Here are {TOP_K} passages retrieved from {book['title']} by {book['author']}.\n\n<passages>\n{passages}\n</passages>\n\n"
            + TASK.format(place=q["place"], title=book["title"], author=book["author"])
        )
        resp = with_retry(lambda: c.models.generate_content(model=MODEL, contents=prompt, config=gen_config()))
        dt = time.time() - t0
        hit = any(q["place"].lower() in chunks[i]["text"].lower() for i in top)
        row = {"id": q["id"], "arm": ARM, "model": MODEL, "answer": resp.text, "latency_s": round(dt, 2),
               "retrieved": [int(i) for i in top], "retrieval_hit": hit, **usage(resp)}
        write_row(ARM, row)
        print(f"{q['id']} hit={hit} {row['prompt_tokens']} tok {dt:.1f}s :: {(resp.text or '')[:90]!r}")


if __name__ == "__main__":
    main()
