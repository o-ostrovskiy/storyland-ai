"""Arm B: classic RAG, embeddings only. The five chunks closest in meaning to the question go into the prompt.

RAG (retrieval-augmented generation) means: find the relevant pieces of text first, then let the model
answer from those pieces instead of from the whole collection. Three steps, R-A-G:

    Retrieve   find the chunks closest to the question
    Augment    put them into the prompt next to the question
    Generate   let the model answer from them

Before any question can be asked, there is preparation (`index_for`):

    1. the book is cut into chunks (prep.py; about 250 words each);
    2. every chunk is turned into an EMBEDDING: a list of 768 numbers produced by a small separate model
       (gemini-embedding-001). The numbers encode meaning: two chunks about the same thing get vectors that
       point in nearly the same direction, whatever words they use. "Vector" and "embedding" are the same
       thing here;
    3. the vectors are stored. This experiment needs no vector database: 5,053 vectors × 768 numbers is a
       15 MB numpy array per chunk size, saved to emb/<key>.npy, and comparing a question against all of
       them (`vecs @ qv`, one matrix multiplication) takes well under a millisecond. A vector database
       (pgvector, Elasticsearch, Qdrant, Chroma...) becomes worth it with millions of chunks, documents that
       change, or access rules, none of which apply to ten public-domain novels. The index is built once
       (22¢ for 1.1M words) and reused for every question; that is RAG's whole economic argument.

Then, per question: embed the question with the same model (a different model's vectors are not
comparable), take the TOP_K chunks with the highest cosine similarity (the vectors are normalised, so the
dot product IS the cosine), and build the prompt from them.

What the results showed: embeddings are weak on proper nouns. Three times (Sandycove and Glasnevin in
Ulysses, Nantes in Eugénie Grandet) none of the five closest chunks contained the name, and the model
honestly answered "not mentioned". To an embedding, "Glasnevin" and "Sandymount" are both just "somewhere
in Dublin". run_hybrid.py fixes this by adding plain keyword search.
"""
import json
import time

import numpy as np
from google.genai import types

from common import CORPUS, EMBED_MODEL, HERE, MODEL, TASK, client, done_ids, gen_config, load_book, load_questions, usage, with_retry, write_row

ARM = "rag"
TOP_K = 5          # how many chunks go into the prompt
EMB = HERE / "emb" # the "vector store": one .npy file per book and chunk size (not committed, rebuilt on demand)
DIM = 768          # embedding size; the model offers up to 3072, 768 is plenty for this corpus and 4× smaller


def embed(c, texts, task_type):
    """Embed a list of texts, 50 per API call. task_type tells the model whether it is embedding a document
    to be searched or a query to search with; the two are tuned slightly differently. Returns unit vectors."""
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
    """The index for one book: (chunks, vectors). Built on first use and cached in emb/; `suffix` picks the
    chunk size ("" = 250 words, "100", "500"). Changing the embedding model or the chunking means deleting
    emb/ and rebuilding: there is no partial reindex for those."""
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
        # Retrieve: embed the question, rank every chunk by cosine similarity, keep the top K.
        qv = embed(c, [f"the place {q['place']} in {book['title']}"], "RETRIEVAL_QUERY")[0]
        top = np.argsort(-(vecs @ qv))[:TOP_K]
        # Augment: the chunks, with their chapter, go into the prompt in front of the task.
        passages = "\n\n".join(f"[chunk {int(i)}, chapter {chunks[i]['chapter']}]\n{chunks[i]['text']}" for i in top)
        prompt = (
            f"Here are {TOP_K} passages retrieved from {book['title']} by {book['author']}.\n\n<passages>\n{passages}\n</passages>\n\n"
            + TASK.format(place=q["place"], title=book["title"], author=book["author"])
        )
        # Generate.
        resp = with_retry(lambda: c.models.generate_content(model=MODEL, contents=prompt, config=gen_config()))
        dt = time.time() - t0
        # retrieval_hit: did the place name make it into the prompt at all? Recorded separately from the
        # verdict, so a wrong answer can be blamed on retrieval or on the model, not on "RAG".
        hit = any(q["place"].lower() in chunks[i]["text"].lower() for i in top)
        row = {"id": q["id"], "arm": ARM, "model": MODEL, "answer": resp.text, "latency_s": round(dt, 2),
               "retrieved": [int(i) for i in top], "retrieval_hit": hit, **usage(resp)}
        write_row(ARM, row)
        print(f"{q['id']} hit={hit} {row['prompt_tokens']} tok {dt:.1f}s :: {(resp.text or '')[:90]!r}")


if __name__ == "__main__":
    main()
