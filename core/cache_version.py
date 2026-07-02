"""Content fingerprint used to auto-invalidate the Discovery result cache.

The Discovery cache (``core.cache`` / ``core.disk_cache``) now persists across
process restarts and redeploys (SQLite on a docker volume). A persistent store
introduces a new hazard the old process-local dict never had: after a model or
prompt change, stale-but-still-live entries computed by the *previous* logic
would keep being served.

To make that impossible, every cache key is namespaced by a **version hash**
derived from the inputs that determine the recommendation output:

* the active model name, and
* the source text of the prompt modules that build the discovery request and
  the agent instructions.

Any change to the model or those prompts changes the hash, so all prior entries
simply stop matching (they are ignored and eventually TTL-evicted) and fresh
results are computed under the new key. This is what lets us drop post-deploy
cache warmup entirely: a deploy that changes nothing recomputes nothing, and a
deploy that changes the model/prompts self-invalidates.

The fingerprint is best-effort: if a prompt module's source can't be read
(e.g. a frozen/stripped build), that input contributes an empty string rather
than raising, so cache versioning degrades to "keyed by model name only" and
never breaks discovery.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect

# Prompt modules whose text determines the discovered region set. A change to
# any of these should invalidate cached discovery results.
_PROMPT_MODULES = ("core.prompts", "agents.prompts")


def _module_source(module_name: str) -> str:
    """Return a module's source text, or '' if it can't be read."""
    try:
        module = importlib.import_module(module_name)
        return inspect.getsource(module)
    except Exception:
        return ""


def compute_cache_version(model_name: str, prompt_modules=_PROMPT_MODULES) -> str:
    """A short, stable hash over the model name + prompt module sources.

    Deterministic for a given (model, prompt-source) pair and 12 hex chars long
    so it stays readable in logs and keys. Changing the model name or editing
    any prompt module changes the result.
    """
    parts = [model_name or ""]
    parts.extend(_module_source(name) for name in prompt_modules)
    digest = hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]
