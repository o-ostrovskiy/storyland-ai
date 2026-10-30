"""Arm A: the whole novel goes into the prompt. One call per question, no retrieval at all.

This is the "just give the model the files" option. It is possible because the model takes up to a million
tokens of input, and the biggest book here (Ulysses) is about 370k. Anthropic's rule of thumb is that a
knowledge base under about 200k tokens (about 500 pages) can simply go into the prompt, with no RAG.

What to watch for, and what the results showed:

    - cost: the whole book is paid for on every question (74k to 369k tokens each; 75× the RAG arms).
    - latency: the only slow calls of the experiment, 10 and 40 seconds, both on Ulysses.
    - accuracy: all seven misses were on the three biggest books. Twice the model quoted a sentence that is
      not in the book (one was a real sentence with "He told me" changed to "I told him"); once, on
      Ulysses, it said the text does not contain "Sandycove", which is there seven times. The "Lost in the
      Middle" paper describes the shape: long inputs are used unevenly, and the middle suffers most.
"""
import time

from common import CORPUS, MODEL, TASK, client, done_ids, gen_config, load_book, load_questions, usage, with_retry, write_row

ARM = "full"


def main():
    c = client()
    done = done_ids(ARM)
    texts = {}
    for q in load_questions():
        if q["id"] in done:
            continue
        book = load_book(q["book"])
        # corpus/<key>.txt: the stripped text with "[Title, Chapter N]" markers, so the model can name the chapter.
        texts.setdefault(q["book"], (CORPUS / f"{q['book']}.txt").read_text())
        prompt = (
            f"Here is the full text of {book['title']} by {book['author']}.\n\n<book>\n{texts[q['book']]}\n</book>\n\n"
            + TASK.format(place=q["place"], title=book["title"], author=book["author"])
        )
        t0 = time.time()
        resp = with_retry(lambda: c.models.generate_content(model=MODEL, contents=prompt, config=gen_config()))
        dt = time.time() - t0
        row = {"id": q["id"], "arm": ARM, "model": MODEL, "answer": resp.text, "latency_s": round(dt, 2), **usage(resp)}
        write_row(ARM, row)
        print(f"{q['id']} {row['prompt_tokens']} tok {dt:.1f}s :: {(resp.text or '')[:90]!r}")


if __name__ == "__main__":
    main()
