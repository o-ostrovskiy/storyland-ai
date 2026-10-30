"""Shared pieces for the four arms of the experiment.

The experiment asks one question 68 times, over ten novels: *does this place appear by name in the book,
and if so, which sentence names it?* Four arms try to answer it in four different ways (see README.md):

    run_full.py    the whole novel goes into the prompt
    run_rag.py     classic RAG: the five chunks whose embeddings are closest to the question
    run_hybrid.py  RAG with both embeddings and keyword scores (BM25), merged
    run_agent.py   the model gets two tools, search_book and read_passage, and decides what to read

Everything they share lives here: the model, the task prompt, the API client, retries, token accounting
and the results file format. Keeping these identical is what makes the arms comparable: the only thing
that differs between them is how the text reaches the model.
"""
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

# One model for every arm, so the comparison is about retrieval, not about model quality.
# gemini-3.1-flash-lite is the small, cheap tier, and the model My Storyland itself runs on.
# It takes up to 1M input tokens, which is why "the whole novel in the prompt" is even possible:
# Ulysses is about 370k tokens.
MODEL = os.environ.get("POC_MODEL", "gemini-3.1-flash-lite")

# The embedding model is a separate, much smaller model. It does not write text; it turns a piece of text
# into a list of numbers (a vector) such that pieces with similar meaning get similar numbers. Only the two
# RAG arms use it. Queries and documents must be embedded by the same model: vectors from different models
# are not comparable, so changing this means rebuilding the index.
EMBED_MODEL = "gemini-embedding-001"

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"      # built by prep.py from the Gutenberg texts (not committed)
RESULTS = HERE / "results"    # one JSONL file per arm, one row per question (committed)

# The task, identical for every arm. It asks for a verbatim quote so that the answer can be checked
# against the book itself (grade.py): a quoted sentence is either in the text or it is not.
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
    """The Gemini client. Reads GOOGLE_API_KEY from the environment."""
    return genai.Client()


def load_book(key):
    """The prepared book: {"key", "title", "author", "paragraphs": [{"id", "chapter", "text"}, ...]}."""
    return json.loads((CORPUS / f"{key}.json").read_text())


def load_questions():
    """questions.json: [{"id", "book", "place", "present"}, ...]. `present` is the ground truth, established
    with a plain text search before the experiment: the place name either occurs in the book or it does not."""
    return json.loads((HERE / "questions.json").read_text())


def gen_config(**kw):
    """temperature=0 makes the model as deterministic as the API allows, so a rerun is comparable."""
    return types.GenerateContentConfig(temperature=0, **kw)


def with_retry(fn, tries=4):
    """Retry an API call on transient errors (rate limits, 5xx), with exponential backoff."""
    delay = 5
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - any error from the API client is worth one more try
            if i == tries - 1:
                raise
            print(f"  retry {i+1} after error: {str(e)[:120]}")
            time.sleep(delay)
            delay *= 2


def usage(resp):
    """Token counts from the response. Input tokens are what the arms differ on by two orders of magnitude;
    output tokens are small and similar everywhere. grade.py turns these into dollars."""
    u = resp.usage_metadata
    return {
        "prompt_tokens": u.prompt_token_count or 0,
        "output_tokens": u.candidates_token_count or 0,
        "thought_tokens": getattr(u, "thoughts_token_count", 0) or 0,
    }


def write_row(arm, row):
    """Append one question's result to results/<arm>.jsonl. Appending as we go makes a run resumable."""
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{arm}.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_ids(arm):
    """Question ids already answered by this arm, so a rerun skips them (and the committed results
    reproduce the tables without any API call)."""
    p = RESULTS / f"{arm}.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["id"] for l in p.read_text().splitlines() if l.strip()}
