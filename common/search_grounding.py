"""Search-grounding receipts from a grounded model response.

Nothing read the model's search receipts before this module existed, which left
one question unanswerable: the researcher prompts *instruct* the model to
search, but nothing verified it ever did. A response answered from memory is
otherwise indistinguishable from a searched one.

An absent receipt is therefore the load-bearing signal, not an edge case:
``extract_search_metadata`` returns None precisely when a response involved no
search, and the caller logs that.

**Gemini reports the same fact on two mutually exclusive channels**, and which
one you get depends on the request config — so reading only one makes a
searching agent look like it skipped:

* **Default** (built-in ``google_search``, no function declarations) ->
  ``grounding_metadata``: the queries run (``web_search_queries``) and the
  pages drawn on (``grounding_chunks[].web``).
* **With** ``tool_config.include_server_side_tool_invocations`` -> explicit
  ``tool_call`` / ``tool_response`` parts on the response content, and
  ``grounding_metadata`` comes back **None**.

That flag is not optional anywhere it appears: the Gemini API rejects a
built-in tool alongside any function declaration without it (400
INVALID_ARGUMENT), and ADK injects a ``set_model_response`` declaration into
every agent that carries both ``tools`` and ``output_schema``. So any agent
built that way reports exclusively on the second channel.

The tool-call channel yields queries but no sources: ``tool_response`` carries
only Google's ``search_suggestions`` chip markup, not the underlying pages, so
there is nothing citable to attribute. A queries-only ``SearchMetadata`` is
therefore a normal, fully-grounded result — not a degraded one.

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


def _clean_tool_call_queries(content: object) -> Tuple[str, ...]:
    """Search queries from server-side tool-call parts, order preserved.

    The second reporting channel (see the module docstring): when
    ``include_server_side_tool_invocations`` is on, the search surfaces as a
    ``tool_call`` part rather than in ``grounding_metadata``.

    Matched on the tool type rather than the part's position: a response may
    interleave search calls with ordinary function calls (``set_model_response``
    is itself one), and only ``GOOGLE_SEARCH*`` types are a web search. The
    check is a substring on ``str(...)`` because the value arrives as an enum
    whose exact member name is Google's to change.
    """
    parts = _get(content, "parts")
    if not isinstance(parts, (list, tuple)):
        return ()
    out: list[str] = []
    for part in parts:
        tool_call = _get(part, "tool_call")
        if tool_call is None:
            continue
        if "GOOGLE_SEARCH" not in str(_get(tool_call, "tool_type") or "").upper():
            continue
        args = _get(tool_call, "args")
        for query in _clean_queries(_get(args, "queries")):
            if query not in out:
                out.append(query)
    return tuple(out)


def extract_search_metadata(response: object) -> Optional[SearchMetadata]:
    """Pull search receipts off a model response, from either channel.

    Returns None when the response carries no usable grounding on *either*
    channel — no queries and no web sources. That None is the "this response
    did not search" signal; callers log it rather than treating it as a
    routine empty value. Checking only ``grounding_metadata`` would report
    every server-side-invocation agent as unsearched, which is the exact
    defect this function exists to detect.

    Never raises: this runs on every model response, and losing observability
    is strictly better than failing a user's request over a shape change.
    """
    try:
        metadata = _get(response, "grounding_metadata")
        # ``_get`` tolerates a None metadata block, so no early return here:
        # a response with no metadata may still carry tool-call receipts.
        queries = _clean_queries(_get(metadata, "web_search_queries"))
        sources = _clean_sources(_get(metadata, "grounding_chunks"))
        tool_queries = _clean_tool_call_queries(_get(response, "content"))
        merged = queries + tuple(q for q in tool_queries if q not in queries)
        if not merged and not sources:
            return None
        return SearchMetadata(queries=merged, sources=sources)
    except Exception:  # pragma: no cover - defensive, shape changes only
        return None
