"""Shared bits for the three arms: client, model, prompt, timing, result rows."""
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

MODEL = os.environ.get("POC_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = "gemini-embedding-001"
HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
RESULTS = HERE / "results"

TASK = (
    "You are helping a literary travel app show, at each stop of a trip, the passage of the novel "
    "where that place is named.\n\n"
    "Question: does the place \"{place}\" appear by name in {title} by {author}?\n"
    "- If it does: quote ONE passage verbatim from the text, at least one full sentence, that names the place, "
    "and say which chapter it is in. Put the quote in double quotes and change nothing inside it.\n"
    "- If the text never names it: say plainly that the book does not mention \"{place}\" and quote nothing.\n"
    "Answer only from the text you were given."
)


def client():
    return genai.Client()


def load_book(key):
    return json.loads((CORPUS / f"{key}.json").read_text())


def load_questions():
    return json.loads((HERE / "questions.json").read_text())


def gen_config(**kw):
    return types.GenerateContentConfig(temperature=0, **kw)


def with_retry(fn, tries=4):
    delay = 5
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - we want to retry on anything transient
            if i == tries - 1:
                raise
            print(f"  retry {i+1} after error: {str(e)[:120]}")
            time.sleep(delay)
            delay *= 2


def usage(resp):
    u = resp.usage_metadata
    return {
        "prompt_tokens": u.prompt_token_count or 0,
        "output_tokens": u.candidates_token_count or 0,
        "thought_tokens": getattr(u, "thoughts_token_count", 0) or 0,
    }


def write_row(arm, row):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{arm}.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_ids(arm):
    p = RESULTS / f"{arm}.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["id"] for l in p.read_text().splitlines() if l.strip()}
