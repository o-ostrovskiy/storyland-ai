"""Search-grounding receipts from a grounded model response.

Every response produced with the built-in ``google_search`` tool carries a
``grounding_metadata`` block: the queries the model actually ran
(``web_search_queries``) and the pages it drew on (``grounding_chunks[].web``).
Nothing read that block before this module existed, which left one question
unanswerable: the researcher prompts *instruct* the model to search, but
nothing verified it ever did. A researcher that answered from memory produced
output indistinguishable from a searched one.

An absent block is therefore the load-bearing signal, not an edge case:
``extract_search_metadata`` returns None precisely when a response involved no
search, and the caller logs that.

Deliberately duck-typed (no ``google.genai`` / ADK import): the shapes here are
plain attribute reads, so this is unit-testable against SimpleNamespace fakes
and stays usable from either the plugin layer or the executor. Every read goes
through ``getattr``-with-default because these are pydantic *declared* fields —
the attribute always exists and the VALUE is what's missing, the same trap
documented on ``LangfusePlugin._extract_token_usage``.

Lives in ``common/`` rather than ``core/`` because ``plugins.langfuse_plugin``
is its first caller and ``core/__init__`` imports the executor, which imports
the plugin back — a ``core`` home makes that a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class SearchSource:
    """One web page the model cited as evidence."""

    title: str
    uri: str

    @property
    def host(self) -> str:
        """Hostname of ``uri`` ('' when unparseable).

        Note: Gemini returns redirect URIs under a vertexaisearch host rather
        than the publisher's own domain, so this is a coarse signal for logs —
        useful for "are the sources varied", not for "which publisher".
        """
        try:
            return urlparse(self.uri).hostname or ""
        except ValueError:
            return ""


@dataclass(frozen=True)
class SearchMetadata:
    """The search receipts for a single model response."""

    queries: Tuple[str, ...]
    sources: Tuple[SearchSource, ...]

    def hosts(self) -> Tuple[str, ...]:
        """Distinct source hostnames, order-preserved.

        Logs take these rather than full URIs: the URI set is long, and the
        queries themselves embed user-supplied book titles, which the log
        allowlist in ``common/logging.py`` exists to keep out of Sentry.
        """
        seen: list[str] = []
        for source in self.sources:
            host = source.host
            if host and host not in seen:
                seen.append(host)
        return tuple(seen)


def _get(obj: object, key: str) -> object:
    """Read ``key`` off an object or a mapping; None when absent or unset."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _clean_queries(raw: object) -> Tuple[str, ...]:
    """Non-empty query strings, deduped with order preserved."""
    if not isinstance(raw, (list, tuple)):
        return ()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        query = item.strip()
        if query and query not in out:
            out.append(query)
    return tuple(out)


def _clean_sources(raw: object) -> Tuple[SearchSource, ...]:
    """Web sources from grounding chunks, deduped by URI, order preserved.

    A chunk carrying no ``web`` block is skipped rather than treated as a
    source: ``GroundingChunk`` also covers maps and retrieved-context
    variants, and only the web one is a citable page. A web block with no
    ``uri`` is likewise skipped — a title alone cannot be linked or verified.
    """
    if not isinstance(raw, (list, tuple)):
        return ()
    out: list[SearchSource] = []
    seen: set[str] = set()
    for chunk in raw:
        web = _get(chunk, "web")
        if web is None:
            continue
        uri = _get(web, "uri")
        if not isinstance(uri, str) or not uri.strip():
            continue
        uri = uri.strip()
        if uri in seen:
            continue
        seen.add(uri)
        title = _get(web, "title")
        out.append(
            SearchSource(
                title=title.strip() if isinstance(title, str) else "",
                uri=uri,
            )
        )
    return tuple(out)


def extract_search_metadata(response: object) -> Optional[SearchMetadata]:
    """Pull search receipts off a model response.

    Returns None when the response carries no usable grounding — no metadata
    block, or one with neither queries nor web sources. That None is the
    "this response did not search" signal; callers log it rather than
    treating it as a routine empty value.

    Never raises: this runs on every model response, and losing observability
    is strictly better than failing a user's request over a shape change.
    """
    try:
        metadata = _get(response, "grounding_metadata")
        if metadata is None:
            return None
        queries = _clean_queries(_get(metadata, "web_search_queries"))
        sources = _clean_sources(_get(metadata, "grounding_chunks"))
        if not queries and not sources:
            return None
        return SearchMetadata(queries=queries, sources=sources)
    except Exception:  # pragma: no cover - defensive, shape changes only
        return None
