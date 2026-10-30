"""Arm C: an agent with two tools over the novel: search the text, read a passage. At most MAX_CALLS tool calls."""
import json
import re
import time

from google.genai import types

from common import MODEL, TASK, client, done_ids, gen_config, load_book, load_questions, usage, with_retry, write_row

ARM = "agent"
MAX_CALLS = 8
MAX_HITS = 8

TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_book",
        description="Case-insensitive search of the novel's text. Returns up to 8 paragraphs containing the query, each with its paragraph id and chapter.",
        parameters={"type": "object", "properties": {"query": {"type": "string", "description": "a word or short phrase to look for"}}, "required": ["query"]},
    ),
    types.FunctionDeclaration(
        name="read_passage",
        description="Returns the full text of a paragraph by id, plus the paragraph before and after it.",
        parameters={"type": "object", "properties": {"paragraph_id": {"type": "integer"}}, "required": ["paragraph_id"]},
    ),
])


def search_book(paras, query):
    pat = re.compile(re.escape(query.strip()), re.I)
    hits = []
    for p in paras:
        if pat.search(p["text"]):
            snippet = p["text"] if len(p["text"]) <= 500 else p["text"][:500] + " …"
            hits.append({"paragraph_id": p["id"], "chapter": p["chapter"], "text": snippet})
            if len(hits) >= MAX_HITS:
                break
    return {"hits": hits, "note": "" if hits else "no paragraph contains this query"}


def read_passage(paras, pid):
    lo, hi = max(0, pid - 1), min(len(paras) - 1, pid + 1)
    return {"paragraphs": [{"paragraph_id": p["id"], "chapter": p["chapter"], "text": p["text"]} for p in paras[lo:hi + 1]]}


def run_one(c, book, q):
    paras = book["paragraphs"]
    intro = (f"You can search and read the text of {book['title']} by {book['author']} with the tools provided. "
             f"Use at most {MAX_CALLS} tool calls, then answer.\n\n")
    contents = [types.Content(role="user", parts=[types.Part(text=intro + TASK.format(place=q["place"], title=book["title"], author=book["author"]))])]
    cfg = gen_config(tools=[TOOLS], automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    calls, tokens = [], {"prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    t0 = time.time()
    for _ in range(MAX_CALLS + 1):
        resp = with_retry(lambda: c.models.generate_content(model=MODEL, contents=contents, config=cfg))
        u = usage(resp)
        for k in tokens:
            tokens[k] += u[k]
        parts = resp.candidates[0].content.parts or []
        fcs = [p.function_call for p in parts if p.function_call]
        if not fcs or len(calls) >= MAX_CALLS:
            text = "".join(p.text for p in parts if p.text) or ""
            if fcs:  # out of budget: ask once more for a plain answer
                contents.append(resp.candidates[0].content)
                contents.append(types.Content(role="user", parts=[types.Part(text="You are out of tool calls. Answer now from what you have read.")]))
                resp = with_retry(lambda: c.models.generate_content(model=MODEL, contents=contents, config=gen_config()))
                u = usage(resp)
                for k in tokens:
                    tokens[k] += u[k]
                text = resp.text or ""
            return text, calls, tokens, time.time() - t0
        contents.append(resp.candidates[0].content)
        responses = []
        for fc in fcs:
            args = dict(fc.args or {})
            if fc.name == "search_book":
                result = search_book(paras, str(args.get("query", "")))
            elif fc.name == "read_passage":
                result = read_passage(paras, int(args.get("paragraph_id", 0)))
            else:
                result = {"error": f"unknown tool {fc.name}"}
            calls.append({"tool": fc.name, "args": args, "hits": len(result.get("hits", result.get("paragraphs", [])))})
            responses.append(types.Part.from_function_response(name=fc.name, response=result))
        contents.append(types.Content(role="user", parts=responses))
    return "", calls, tokens, time.time() - t0


def main():
    c = client()
    done = done_ids(ARM)
    for q in load_questions():
        if q["id"] in done:
            continue
        book = load_book(q["book"])
        text, calls, tokens, dt = run_one(c, book, q)
        row = {"id": q["id"], "arm": ARM, "model": MODEL, "answer": text, "latency_s": round(dt, 2),
               "tool_calls": calls, "n_calls": len(calls), **tokens}
        write_row(ARM, row)
        print(f"{q['id']} calls={len(calls)} {tokens['prompt_tokens']} tok {dt:.1f}s :: {text[:90]!r}")


if __name__ == "__main__":
    main()
