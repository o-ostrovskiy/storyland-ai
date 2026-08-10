"""Contract tests for core.search_grounding.extract_search_metadata.

The load-bearing case is the NEGATIVE one: a None return is the signal that a
response was not search-grounded — the only way to tell a researcher that ran
google_search from one that answered from memory. So "absent metadata" and
"metadata present but empty" must both be None, and neither may be confused
with a successful extraction carrying zero results.

Fakes are SimpleNamespace rather than google.genai types on purpose: the
extractor is duck-typed so it can be exercised without a live model, and these
tests pin the exact attribute shapes it reads.
"""

from types import SimpleNamespace

from common.search_grounding import (
    SearchSource,
    extract_search_metadata,
)


def _web(uri=None, title=None):
    return SimpleNamespace(web=SimpleNamespace(uri=uri, title=title, domain=None))


def _response(queries=None, chunks=None):
    return SimpleNamespace(
        grounding_metadata=SimpleNamespace(
            web_search_queries=queries,
            grounding_chunks=chunks,
        )
    )


class TestNoGrounding:
    """Every shape that means "this response did not search"."""

    def test_missing_attribute_entirely(self):
        assert extract_search_metadata(SimpleNamespace()) is None

    def test_declared_field_set_to_none(self):
        """The pydantic trap: the field exists, the VALUE is None."""
        assert extract_search_metadata(SimpleNamespace(grounding_metadata=None)) is None

    def test_metadata_present_but_empty(self):
        assert extract_search_metadata(_response(queries=[], chunks=[])) is None

    def test_metadata_fields_all_none(self):
        assert extract_search_metadata(_response()) is None

    def test_chunks_with_no_web_block_are_not_sources(self):
        """A maps/retrieved-context chunk is not a citable web page."""
        chunks = [SimpleNamespace(web=None), SimpleNamespace(maps=object())]
        assert extract_search_metadata(_response(chunks=chunks)) is None

    def test_web_block_without_uri_is_dropped(self):
        """A title with no URI cannot be linked or verified."""
        assert extract_search_metadata(_response(chunks=[_web(title="Bath")])) is None


class TestExtraction:
    def test_queries_and_sources(self):
        result = extract_search_metadata(
            _response(
                queries=["persuasion filming locations"],
                chunks=[_web(uri="https://example.com/a", title="Bath")],
            )
        )
        assert result.queries == ("persuasion filming locations",)
        assert result.sources == (
            SearchSource(title="Bath", uri="https://example.com/a"),
        )

    def test_queries_alone_count_as_grounded(self):
        """A search that returned nothing still proves a search happened."""
        result = extract_search_metadata(_response(queries=["obscure book setting"]))
        assert result is not None
        assert result.sources == ()

    def test_queries_deduped_and_order_preserved(self):
        result = extract_search_metadata(_response(queries=["b", "a", "b", "  ", None]))
        assert result.queries == ("b", "a")

    def test_sources_deduped_by_uri(self):
        chunks = [
            _web(uri="https://example.com/a", title="First"),
            _web(uri="https://example.com/a", title="Duplicate"),
            _web(uri="https://example.com/b", title="Second"),
        ]
        result = extract_search_metadata(_response(chunks=chunks))
        assert [s.uri for s in result.sources] == [
            "https://example.com/a",
            "https://example.com/b",
        ]

    def test_missing_title_becomes_empty_string(self):
        result = extract_search_metadata(_response(chunks=[_web(uri="https://x.dev/a")]))
        assert result.sources[0].title == ""

    def test_dict_shaped_response(self):
        """Mirrors _extract_token_usage's dict tolerance."""
        result = extract_search_metadata(
            {
                "grounding_metadata": {
                    "web_search_queries": ["q"],
                    "grounding_chunks": [{"web": {"uri": "https://x.dev/a", "title": "T"}}],
                }
            }
        )
        assert result.queries == ("q",)
        assert result.sources[0].uri == "https://x.dev/a"


class TestHosts:
    def test_distinct_hosts_order_preserved(self):
        chunks = [
            _web(uri="https://b.example/1"),
            _web(uri="https://a.example/2"),
            _web(uri="https://b.example/3"),
        ]
        result = extract_search_metadata(_response(chunks=chunks))
        assert result.hosts() == ("b.example", "a.example")

    def test_unparseable_uri_yields_no_host(self):
        result = extract_search_metadata(_response(chunks=[_web(uri="not a url")]))
        assert result.hosts() == ()


class TestNeverRaises:
    def test_malformed_metadata_returns_none(self):
        """Observability must never fail a user's request."""

        class Exploding:
            @property
            def grounding_metadata(self):
                raise RuntimeError("shape changed under us")

        assert extract_search_metadata(Exploding()) is None

    def test_non_list_collections_are_ignored(self):
        response = _response(queries="not-a-list", chunks="not-a-list")
        assert extract_search_metadata(response) is None
