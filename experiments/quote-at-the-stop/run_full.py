"""Arm A: the whole novel goes into the prompt, one call per question."""
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
