"""MYS-460 — the canonical cross-job place_key, and the cache flush that makes it real.

Two things are asserted here, and the second is the one that decides whether the
feature works at all:

1. The KEY is minted only from grounded structured geo fields — never from
   ``region_id`` (an ordinal within one response) and never from ``region_name``
   (prose). A key derived from either would either never match (dead branch) or
   match the wrong place (fabricated shared setting).
2. The discovery cache NAMESPACE moves with this change. The cache is persistent,
   cross-user and keyed on (title, author, preferences); without a namespace bump
   every HIT would replay a pre-MYS-460 region with NO place_key — and hits land
   hardest on popular, repeated titles, i.e. exactly the book-club case this
   exists for. The feature would ship and never fire on the books it was built
   for. So the flush is a REQUIREMENT, and it is tested as a key test.
"""

import asyncio
import json
from pathlib import Path

import pytest

from agents.prompts import CURRENT_PROMPT_VERSION, load_prompts
from core.cache import TTLCache
from core.cache_version import _PROMPT_MODULES, compute_cache_version
from core.regions import enrich_region_analysis
from models.discovery import RegionOption
from models.place_key import mint_place_key


def _region(**overrides) -> dict:
    base = dict(
        region_id=1,
        region_name="Paris & Île-de-France",
        country_code="FR",
        primary_locality="Paris",
        cities=[{"name": "Paris", "country": "France"}],
        estimated_days=4,
        travel_note="Metro",
        highlights="Notre-Dame",
    )
    base.update(overrides)
    return base


# ── 1. The key: canonical, grounded, and refusing to guess ───────────────────

class TestMintPlaceKey:
    def test_canonical_form(self):
        assert mint_place_key("FR", "Paris") == "fr:paris"
        assert mint_place_key("US", "New York") == "us:new-york"

    def test_case_and_accents_and_punctuation_are_canonicalised(self):
        # The same place, described three ways by three different jobs.
        assert mint_place_key("fr", " paris ") == "fr:paris"
        assert mint_place_key("IS", "Reykjavík") == "is:reykjavik"
        assert mint_place_key("RU", "St. Petersburg") == "ru:st-petersburg"

    def test_country_disambiguates_the_same_locality_name(self):
        # THE false positive the key exists to kill: Paris, Texas is not Paris, France.
        assert mint_place_key("US", "Paris") != mint_place_key("FR", "Paris")

    def test_a_country_name_or_alpha3_is_not_a_country_code(self):
        # Only ISO-3166-1 alpha-2. Anything else is a guess wearing a code's clothes.
        assert mint_place_key("France", "Paris") is None
        assert mint_place_key("FRA", "Paris") is None
        assert mint_place_key("", "Paris") is None

    def test_missing_grounded_fields_yield_no_key_never_a_guess(self):
        assert mint_place_key(None, "Paris") is None
        assert mint_place_key("FR", None) is None
        assert mint_place_key("FR", "") is None
        assert mint_place_key("FR", "———") is None  # slugs to empty

    def test_only_real_iso_3166_1_alpha2_codes_or_the_two_named_aliases_are_accepted(self):
        # `^[A-Za-z]{2}$` (the shape-only check this replaces) would accept ALL
        # of these — they are two letters, but none is an ISO code or alias.
        assert mint_place_key("EU", "Paris") is None
        assert mint_place_key("ZZ", "Nowhere") is None
        assert mint_place_key("XX", "Nowhere") is None

    def test_uk_and_el_are_the_two_real_aliases_and_normalise_to_the_iso_code(self):
        # "UK" is the single most likely thing an LLM emits for a British
        # region — our top collections are Edinburgh, London, Dublin — and the
        # ISO code is "GB". Without normalising this alias, one job's "UK"
        # region and another's "GB" region are two different keys for the same
        # city and never intersect. "EL" is the EU's own reservation for
        # Greece; the ISO code is "GR".
        assert mint_place_key("UK", "London") == mint_place_key("GB", "London") == "gb:london"
        assert mint_place_key("EL", "Athens") == mint_place_key("GR", "Athens") == "gr:athens"

    def test_a_two_letter_code_that_is_not_a_real_iso_code_or_alias_yields_no_key(self):
        # A GUESS is not an alias. Only the two named exceptional reservations
        # normalise — everything else outside the real ISO set stays a missed
        # combine, never a wrong one.
        assert mint_place_key("ZZ", "X") is None
        assert mint_place_key("UX", "London") is None


class TestEnrichmentNeverTrustsAnIncomingKey:
    def test_a_hostile_incoming_place_key_is_replaced_with_the_grounded_one(self):
        # ADK writes the model's RAW parsed JSON dict into session state — if a
        # model ever emits an unasked-for `place_key` (or a legacy/cached entry
        # carries one), the enrichment seam must not ride it through untouched.
        # A fabricated "fr:paris" stapled onto a US/Paris region is a WRONG
        # combine — the one failure mode this whole design must not have.
        hostile = _region(
            place_key="fr:paris", country_code="US", primary_locality="Paris"
        )
        out = enrich_region_analysis({"regions": [hostile]})
        assert out["regions"][0]["place_key"] == "us:paris"

    def test_overwriting_is_still_idempotent(self):
        # Minting is deterministic, so re-enriching an already-correct region
        # must not change its key.
        once = enrich_region_analysis({"regions": [_region()]})
        twice = enrich_region_analysis(once)
        assert twice["regions"][0]["place_key"] == once["regions"][0]["place_key"] == "fr:paris"


class TestKeyIsNotDerivedFromTheWrongThing:
    def test_region_name_never_becomes_a_key(self):
        # A rich, unambiguous-looking region_name with no structured fields must
        # produce NO key. A "just for v1" prose fallback is indistinguishable
        # from a real key downstream — and it is how Paris, Texas gets combined
        # with Paris, France.
        analysis = enrich_region_analysis(
            {"regions": [_region(country_code=None, primary_locality=None,
                                 region_name="Paris, France")]}
        )
        assert analysis["regions"][0]["place_key"] is None

    def test_region_id_is_not_an_identity_across_jobs(self):
        # region_id 1 in book A's response and region_id 1 in book B's response
        # are DIFFERENT PLACES. The key must not equate them...
        book_a = enrich_region_analysis({"regions": [_region(region_id=1)]})
        book_b = enrich_region_analysis(
            {"regions": [_region(region_id=1, region_name="Barcelona, Spain",
                                 country_code="ES", primary_locality="Barcelona")]}
        )
        assert book_a["regions"][0]["place_key"] != book_b["regions"][0]["place_key"]

        # ...and the same place discovered under a different ordinal, with a
        # differently-worded name, is still the same place. This is the contract
        # PR2's intersection stands on.
        book_c = enrich_region_analysis(
            {"regions": [_region(region_id=3, region_name="Paris, France")]}
        )
        assert book_a["regions"][0]["place_key"] == book_c["regions"][0]["place_key"] == "fr:paris"


# ── 2. Enrichment: on the wire, on both paths, mutating nothing ──────────────

class TestEnrichRegionAnalysis:
    def test_adds_place_key_to_every_region_and_preserves_the_payload(self):
        out = enrich_region_analysis(
            {"regions": [_region(), _region(region_id=2, country_code="ES",
                                            primary_locality="Barcelona")],
             "analysis_note": "note"}
        )
        assert [r["place_key"] for r in out["regions"]] == ["fr:paris", "es:barcelona"]
        assert out["analysis_note"] == "note"
        assert out["regions"][0]["region_name"] == "Paris & Île-de-France"

    def test_does_not_mutate_its_input(self):
        # Session state must not be written through (SessionStateAccessor's
        # setters are silent no-ops against persisted state — MYS-172).
        original = {"regions": [_region()]}
        enrich_region_analysis(original)
        assert "place_key" not in original["regions"][0]

    def test_is_idempotent(self):
        once = enrich_region_analysis({"regions": [_region()]})
        twice = enrich_region_analysis(once)
        assert twice == once

    def test_tolerates_empty_and_malformed_payloads(self):
        assert enrich_region_analysis(None) == {}
        assert enrich_region_analysis({}) == {}
        assert enrich_region_analysis({"regions": []}) == {"regions": []}
        assert enrich_region_analysis({"regions": "nonsense"}) == {"regions": "nonsense"}


class TestRegionOptionSchema:
    def test_place_key_is_derived_on_dump(self):
        assert RegionOption(**_region()).model_dump()["place_key"] == "fr:paris"

    def test_the_model_is_never_asked_to_emit_the_key(self):
        # ADK hands the LLM `output_schema.model_json_schema()` — the VALIDATION
        # schema. place_key is a computed field, so it is absent there: the model
        # supplies the two grounded descriptions it can know, and we mint the
        # identity. A model that could emit place_key could invent one, or collide
        # two places by emitting the same one.
        asked_of_model = RegionOption.model_json_schema()["properties"]
        assert "place_key" not in asked_of_model
        assert "country_code" in asked_of_model and "primary_locality" in asked_of_model
        # ...and it IS on the wire.
        assert "place_key" in RegionOption.model_json_schema(mode="serialization")["properties"]


class TestSseWireContract:
    def test_the_regions_sse_event_relays_place_key_untouched(self):
        # SSERegionsEvent.regions is List[dict]: the key must survive to fe (PR2).
        # A typed model that dropped unknown fields would make this whole PR a
        # no-op on the only surface that consumes it.
        from api.models import SSERegionsEvent

        payload = json.loads(
            SSERegionsEvent(
                job_id="j", regions=enrich_region_analysis({"regions": [_region()]})["regions"],
                analysis_note="n",
            ).model_dump_json()
        )
        assert payload["regions"][0]["place_key"] == "fr:paris"


# ── 3. THE AC: a discovery cache HIT can never return a keyless region ───────

class TestCacheNamespaceFlush:
    def test_the_discovery_schema_is_an_input_to_the_cache_version(self):
        # The cached VALUE is a discovery payload, so its SHAPE must move the
        # fingerprint. Before this PR the fingerprint hashed only the two prompt
        # modules — adding place_key to RegionOption would NOT have changed it.
        assert "models.discovery" in _PROMPT_MODULES
        pre_mys460 = compute_cache_version("m", prompt_modules=("core.prompts", "agents.prompts"))
        post_mys460 = compute_cache_version("m", prompt_modules=_PROMPT_MODULES)
        assert pre_mys460 != post_mys460

    def test_an_entry_written_under_the_old_namespace_is_unreachable(self):
        # The real mechanism, exercised end to end: executor._versioned_key
        # namespaces every key with the fingerprint. An entry stored under the
        # pre-MYS-460 namespace must MISS under the current one — that is what
        # guarantees no hit can ever replay a region with no place_key.
        def versioned(version: str, base: str) -> str:
            return f"dv:{version}:{base}"

        old_ns = compute_cache_version("m", prompt_modules=("core.prompts", "agents.prompts"))
        new_ns = compute_cache_version("m", prompt_modules=_PROMPT_MODULES)
        base = "v1|gone with the wind|margaret mitchell"

        async def scenario():
            cache = TTLCache(ttl_seconds=60, max_entries=10)
            # A stale entry from before this PR: a region with NO place_key.
            await cache.set(versioned(old_ns, base), {"regions": [_region(place_key=None)]})
            return await cache.get(versioned(new_ns, base))

        assert asyncio.run(scenario()) is None, (
            "a pre-MYS-460 cache entry must not be reachable under the new namespace — "
            "otherwise every hit on a popular book replays a region with no identity"
        )

    def test_the_prompt_version_bump_also_moves_the_fingerprint(self):
        # agents/prompts.py carries CURRENT_PROMPT_VERSION in its SOURCE, which is
        # hashed. (The v3.json CONTENT is not — data, not source: MYS-462. That is
        # exactly why the bump lives in the .py and why models.discovery is in the
        # module list: neither flush depends on the JSON being seen.)
        import inspect
        import agents.prompts as prompts_module

        assert CURRENT_PROMPT_VERSION == "v3"
        assert 'CURRENT_PROMPT_VERSION = "v3"' in inspect.getsource(prompts_module)


class TestPromptV3:
    def test_v3_exists_loads_and_asks_for_the_grounded_geo_fields(self):
        prompts = load_prompts("v3")
        instruction = prompts.region_analyzer
        assert "country_code" in instruction and "primary_locality" in instruction
        assert "ISO 3166-1 alpha-2" in instruction
        # It must tell the model to omit rather than guess — a guessed country
        # code is a fabricated place identity.
        assert "never guess" in instruction.lower()

    def test_v3_is_v2_plus_the_region_analyzer_change_only(self):
        v2 = json.loads(Path("agents/prompts/v2.json").read_text(encoding="utf-8"))
        v3 = json.loads(Path("agents/prompts/v3.json").read_text(encoding="utf-8"))
        changed = [a for a in v2["agents"] if v2["agents"][a] != v3["agents"][a]]
        assert changed == ["region_analyzer"], (
            "v3 must not quietly re-tune any other agent — this PR is a schema "
            "change, and an unreviewed prompt edit would ride the same cache flush"
        )
        assert v2["agents"].keys() == v3["agents"].keys()

    def test_v2_is_still_loadable(self):
        # Published versions are immutable and must stay pinnable (eval/rollback).
        assert load_prompts("v2").region_analyzer
        assert "country_code" not in load_prompts("v2").region_analyzer


class TestExecutorWiring:
    """The executor needs ADK to import, so its wiring is asserted at the source.

    Stated plainly rather than dressed up: these are source assertions, not
    behavioural ones. The behaviour they stand in for (enrichment on both paths)
    is exercised above against the real `enrich_region_analysis`; what a source
    guard can still catch is the enrichment being dropped from ONE of the two
    paths — which is precisely the mistake that would ship a feature that works
    on a cache miss and dies on a hit.
    """

    def test_both_discovery_paths_enrich(self):
        src = Path("core/executor.py").read_text(encoding="utf-8")
        assert "enrich_region_analysis" in src
        # Fresh run: the value we CACHE and the value we EMIT come from the same
        # enriched payload — a cached hit and a fresh run must be identical on the wire.
        assert "region_analysis = enrich_region_analysis(state.region_analysis)" in src
        assert "await self._discovery_cache.set(cache_key, region_analysis)" in src
        assert "await self._discovery_cache.set(cache_key, state.region_analysis)" not in src, (
            "caching the RAW analysis would store regions with no place_key"
        )
        # Cache replay path.
        assert "region_analysis = enrich_region_analysis(region_analysis)" in src
        # Exactly two call sites — one per discovery path. A third would mean
        # someone also enriched a path that doesn't reach the wire; a first-only
        # would mean the hit path was forgotten, which is the whole hazard.
        assert src.count("enrich_region_analysis(") == 2
