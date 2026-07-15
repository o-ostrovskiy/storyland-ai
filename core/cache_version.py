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

# Modules whose text determines the discovered region set. A change to any of
# these must invalidate cached discovery results.
#
# ``models.discovery`` is in this list on purpose (MYS-460). The cached value IS
# a discovery payload, so the payload's SHAPE is an input to it: when we added
# ``place_key`` to RegionOption, every pre-existing entry became a region with no
# identity — and a persistent, cross-user cache would have gone on serving those
# keyless regions forever, hardest on the popular repeated titles the combined
# readaway is for. Hashing the schema module makes that self-invalidating instead
# of a thing someone has to remember. ``agents.prompts`` also carries
# CURRENT_PROMPT_VERSION, so a prompt-version bump changes this hash too — but the
# v2/v3 JSON *content* does not (it is data, not source: that gap is MYS-462).
#
# ``models.place_key`` and ``core.regions`` are in this list for the same reason,
# one seam deeper (MYS-460 review round 7). The cached payload doesn't just have
# the *shape* ``place_key`` — every ``place_key`` value in it was MINTED by
# ``mint_checked_place_key()`` (``models/place_key.py``) and applied at the
# ``enrich_region_analysis`` seam (``core/regions.py``). That mint logic is an
# INPUT to the cached value, not just its schema, and it has been corrected six
# times in this PR alone (hostile incoming key, ISO membership, locality-vs-cities,
# the country pair, the full ISO name table, admin_area). Without these two
# modules in the fingerprint, the next correction to how a key is minted would be
# invisible on every cache hit — hardest on the popular, repeated titles the
# combined readaway exists for, which is the MYS-222 trap: a cache that turns a
# real fix into a permanent false green. (``core.regions`` also carries
# ``validate_region_selection``, which does not shape the cached payload — an
# edit to it flushes the cache spuriously. That is an unnecessary re-warm, not a
# wrong combine, so it is the safe side to be wrong on.)
_PROMPT_MODULES = (
    "core.prompts",
    "agents.prompts",
    "models.discovery",
    "models.place_key",
    "core.regions",
)


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
