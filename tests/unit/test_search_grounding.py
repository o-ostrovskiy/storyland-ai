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


def _part(tool_type=None, queries=None):
    """One response part. Omitting tool_type gives a part with no tool_call."""
    if tool_type is None:
        return SimpleNamespace(function_call=SimpleNamespace(name="set_model_response"))
    return SimpleNamespace(
        tool_call=SimpleNamespace(tool_type=tool_type, args={"queries": queries})
    )


def _server_side_response(parts, metadata_queries=None):
    """A response from an agent with server-side tool invocations enabled.

    ``grounding_metadata`` is None unless explicitly given: that is the whole
    trap — the API reports the search through parts INSTEAD of metadata, so a
    detector reading only metadata sees a searching agent as unsearched.
    """
    return SimpleNamespace(
        grounding_metadata=(
            SimpleNamespace(web_search_queries=metadata_queries, grounding_chunks=None)
            if metadata_queries is not None
            else None
        ),
        content=SimpleNamespace(parts=parts),
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


class TestServerSideToolCallChannel:
    """The second reporting channel (MYS-818).

    An agent carrying both ``tools`` and ``output_schema`` needs
    ``include_server_side_tool_invocations``, and with it Gemini reports the
    search as ``tool_call`` parts while ``grounding_metadata`` comes back None.
    Reading metadata alone reports every such agent as never having searched —
    the precise defect this module exists to detect, aimed at itself.
    """

    def test_tool_call_alone_counts_as_grounded(self):
        result = extract_search_metadata(
            _server_side_response(
                [_part("ToolType.GOOGLE_SEARCH_WEB", ["piranesi real locations"])]
            )
        )
        assert result is not None
        assert result.queries == ("piranesi real locations",)

    def test_no_sources_on_this_channel_is_not_degraded(self):
        """tool_response carries chip markup, not pages — nothing to cite."""
        result = extract_search_metadata(
            _server_side_response([_part("ToolType.GOOGLE_SEARCH_WEB", ["a"])])
        )
        assert result.sources == ()
        assert result.hosts() == ()

    def test_parts_without_a_tool_call_are_skipped(self):
        """set_model_response is a function_call, not a search."""
        result = extract_search_metadata(
            _server_side_response(
                [_part(), _part("ToolType.GOOGLE_SEARCH_WEB", ["a"]), _part()]
            )
        )
        assert result.queries == ("a",)

    def test_non_search_tool_type_is_not_a_search(self):
        assert (
            extract_search_metadata(
                _server_side_response([_part("ToolType.CODE_EXECUTION", ["a"])])
            )
            is None
        )

    def test_both_channels_merge_and_dedupe(self):
        result = extract_search_metadata(
            _server_side_response(
                [_part("ToolType.GOOGLE_SEARCH_WEB", ["shared", "from_parts"])],
                metadata_queries=["from_metadata", "shared"],
            )
        )
        assert result.queries == ("from_metadata", "shared", "from_parts")

    def test_repeated_tool_calls_dedupe(self):
        result = extract_search_metadata(
            _server_side_response(
                [
                    _part("ToolType.GOOGLE_SEARCH_WEB", ["a", "b"]),
                    _part("ToolType.GOOGLE_SEARCH_WEB", ["b", "c"]),
                ]
            )
        )
        assert result.queries == ("a", "b", "c")

    def test_content_shapes_that_carry_nothing(self):
        for content in (None, SimpleNamespace(parts=None), SimpleNamespace(parts=[])):
            assert (
                extract_search_metadata(
                    SimpleNamespace(grounding_metadata=None, content=content)
                )
                is None
            )

    def test_tool_call_with_no_queries_is_not_grounded(self):
        assert (
            extract_search_metadata(
                _server_side_response([_part("ToolType.GOOGLE_SEARCH_WEB", None)])
            )
            is None
        )


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
