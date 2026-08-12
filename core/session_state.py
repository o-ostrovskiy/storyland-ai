"""
Typed session state access.

Wraps the ADK session.state dict with typed properties, eliminating
magic string key access scattered across the codebase.
"""

import json
from typing import Optional, List


class SessionStateKeys:
    """Constants for all session state keys used by agents and orchestration."""

    BOOK_TITLE = "book_title"
    AUTHOR = "author"
    BOOK_METADATA = "book_metadata"
    BOOK_CONTEXT = "book_context"
    READER_PROFILE = "reader_profile"
    CITY_DISCOVERY = "city_discovery"
    LANDMARK_DISCOVERY = "landmark_discovery"
    AUTHOR_SITES = "author_sites"
    REGION_ANALYSIS = "region_analysis"
    SELECTED_REGIONS = "selected_regions"
    FINAL_ITINERARY = "final_itinerary"
    COMPOSER_ENVELOPE = "composer_envelope"
    LAST_SUGGESTIONS = "last_suggestions"
    EXPANSIONS = "expansions"
    EXPANSION_COUNT = "expansion_count"
    EXPANSION_IN_PROGRESS = "expansion_in_progress"
    LAST_EXPANSION = "last_expansion"
    LAST_BOOK_RECOMMENDATIONS = "last_book_recommendations"
    LAST_PLACE_TO_BOOK = "last_place_to_book"
    BOOK_RECOMMENDATION_COUNT = "book_recommendation_count"
    BOOK_RECS_IN_PROGRESS = "book_recs_in_progress"
    BOOK_RECOMMENDATION_CHIP_ID = "book_recommendation_chip_id"
    BOOK_RECOMMENDATION_CHIP = "book_recommendation_chip"
    # Discovery payload keys whose researcher never called google_search on
    # this run (MYS-816). Written by discover(), read by grounding_research_text.
    UNVERIFIED_DISCOVERY = "unverified_discovery"
    USER_PREFERENCES = "user:preferences"
    USER_LOCATION = "user_location"
    JOB_FAILED = "job_failed"


class SessionStateAccessor:
    """Typed, READ-ONLY wrapper around session.state dict.

    Usage:
        state = SessionStateAccessor(session.state)
        metadata = state.book_metadata  # typed access, no magic strings

    MYS-172: this accessor is deliberately read-only. ADK's ``session.state``
    does not persist an in-place mutation (``state[key] = value``) across a
    ``get_session()`` call -- only ``session_service.append_event(session,
    Event(actions=EventActions(state_delta={...})))`` does. A setter here
    would look like a typed write and silently be a no-op against persisted
    state, which is exactly the bug this ticket fixes (two call sites in
    ``core/executor.py`` did precisely that). Write session state directly
    via ``append_event`` at the call site, next to the ``session`` object
    the write needs -- do not re-add a setter to this class.
    """

    def __init__(self, state: dict):
        self._state = state

    @property
    def book_metadata(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.BOOK_METADATA)

    @property
    def region_analysis(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.REGION_ANALYSIS)

    @property
    def regions(self) -> List[dict]:
        ra = self.region_analysis or {}
        return ra.get("regions", [])

    @property
    def analysis_note(self) -> str:
        ra = self.region_analysis or {}
        return ra.get("analysis_note", "")

    @property
    def selected_regions(self) -> List[dict]:
        return self._state.get(SessionStateKeys.SELECTED_REGIONS, [])

    @property
    def final_itinerary(self) -> Optional[object]:
        return self._state.get(SessionStateKeys.FINAL_ITINERARY)

    @property
    def user_preferences(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.USER_PREFERENCES)

    @property
    def failed(self) -> bool:
        return bool(self._state.get(SessionStateKeys.JOB_FAILED))

    @property
    def book_title(self) -> str:
        bm = self.book_metadata or {}
        return bm.get("book_title", "") or self._state.get(SessionStateKeys.BOOK_TITLE, "")

    @property
    def author(self) -> str:
        bm = self.book_metadata or {}
        return bm.get("author", "") or self._state.get(SessionStateKeys.AUTHOR, "")

    @property
    def composer_envelope(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.COMPOSER_ENVELOPE)

    @property
    def last_suggestions(self) -> List[dict]:
        return self._state.get(SessionStateKeys.LAST_SUGGESTIONS, [])

    @property
    def expansions(self) -> List[dict]:
        return self._state.get(SessionStateKeys.EXPANSIONS, [])

    @property
    def expansion_count(self) -> int:
        return int(self._state.get(SessionStateKeys.EXPANSION_COUNT, 0))

    @property
    def expansion_in_progress(self) -> bool:
        return bool(self._state.get(SessionStateKeys.EXPANSION_IN_PROGRESS, False))

    @property
    def last_expansion(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.LAST_EXPANSION)

    @property
    def book_context(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.BOOK_CONTEXT)

    @property
    def city_discovery(self) -> Optional[object]:
        return self._state.get(SessionStateKeys.CITY_DISCOVERY)

    @property
    def landmark_discovery(self) -> Optional[object]:
        return self._state.get(SessionStateKeys.LANDMARK_DISCOVERY)

    @property
    def author_sites(self) -> Optional[object]:
        return self._state.get(SessionStateKeys.AUTHOR_SITES)

    @property
    def unverified_discovery(self) -> List[str]:
        """Discovery payload keys produced without a search (MYS-816)."""
        value = self._state.get(SessionStateKeys.UNVERIFIED_DISCOVERY)
        return value if isinstance(value, list) else []

    @property
    def discovery_verification_ran(self) -> bool:
        """Did the fail-closed pass actually write a verdict for this run?

        The positive representation of "every researcher searched" (MYS-816
        r3). ``unverified_discovery`` returns ``[]`` for two opposite states:
        the pass ran and cleared every researcher, and the pass never ran at
        all (an early error, a pre-v3 cache entry, a broken observation seam).
        Both make ``all_discovery_unverified`` False, so the guard fails OPEN
        -- and until now it did so silently, with nothing anywhere able to say
        which of the two had happened.

        That is the same absence-as-evidence shape this ticket has had to
        close three times over: on the metric, on the ledger, and here on the
        state that feeds the guard. The presence of the KEY is the receipt, so
        the executor writes it unconditionally -- including the empty list --
        and the cache bundle replays it by TYPE rather than by truthiness. An
        empty list is a verdict; a missing key is silence.
        """
        return isinstance(
            self._state.get(SessionStateKeys.UNVERIFIED_DISCOVERY), list
        )

    def _discovery_payloads(self) -> List[tuple]:
        """The (key, value) discovery payloads that are actually present."""
        return [
            (key, value)
            for key, value in (
                (SessionStateKeys.BOOK_CONTEXT, self.book_context),
                (SessionStateKeys.CITY_DISCOVERY, self.city_discovery),
                (SessionStateKeys.LANDMARK_DISCOVERY, self.landmark_discovery),
                (SessionStateKeys.AUTHOR_SITES, self.author_sites),
            )
            if value
        ]

    @property
    def all_discovery_unverified(self) -> bool:
        """True when NO PRESENT discovery payload is usable as evidence.

        The difference between *no evidence* and *disqualified evidence*, and
        they need opposite handling (MYS-816 r2). ``grounding_research_text``
        returns "" for both: the local-atmosphere path has no discovery
        research at all, and a run whose researchers skipped ``google_search``
        has research that may not be used as proof. The downstream guard is
        fail-OPEN on an empty haystack -- correct for the first case, exactly
        backwards for the second, where every literal claim survives *because*
        nothing was verified. Callers pass this to
        ``downgrade_ungrounded_match_types`` so the second case fails closed.

        🔴 **The exact trigger, because every earlier description of this flag
        overstated it** (MYS-816 r3). It is NOT "the run where all four
        researchers skipped". ``_discovery_payloads`` filters on ``if value``,
        so an EMPTY payload counts as absent, and ``all()`` over a
        single-element list is True. The real rule is:

            at least one discovery payload is non-empty, and every non-empty
            one came from a researcher with no search receipt.

        A run where three researchers returned nothing and the fourth answered
        from memory therefore blanket-demotes the whole itinerary -- and that
        is CORRECT, which is why the trigger is documented rather than
        tightened. Requiring the full researcher set would restore the
        fail-open at a slightly different maximum: the haystack in that run is
        empty (the one payload present is excluded as unverified), so without
        the flag every literal claim would survive on no evidence at all --
        precisely the inversion r2 fixed. The words were wrong, not the rule.
        Pinned by ``test_a_single_unverified_payload_still_disqualifies`` and
        its converse.

        False when no discovery ran, which keeps the local-atmosphere path
        untouched -- and ALSO false when the fail-closed pass never ran, which
        is a different state entirely. This property cannot tell those two
        apart and does not try; ``discovery_verification_ran`` is the positive
        receipt that separates them, and callers that fail open on this flag
        should read it before treating False as "everything checked out".
        """
        present = self._discovery_payloads()
        if not present:
            return False
        skip = set(self.unverified_discovery)
        return all(key in skip for key, _ in present)

    @property
    def grounding_research_text(self) -> str:
        """Concatenate the grounded discovery research into one text blob.

        Joins the book-context, city, landmark, and author-site discovery
        outputs (the grounded research the composer draws from) into a single
        string so itinerary claims can be checked against what the grounding
        chain actually found. Returns "" when no discovery research is present
        (e.g. the local-atmosphere path), which callers treat as "cannot prove
        anything ungrounded" and leave labels unchanged.

        **Payloads whose researcher never searched are excluded** (MYS-816).
        Researchers skip ``google_search`` stochastically -- roughly one to two
        of the four on most runs -- and still emit places from model memory
        (an observed run produced the author site "Personal Office"). Those
        names are not evidence, so they must not appear in the haystack that
        ``downgrade_ungrounded_match_types`` treats as proof. Excluding them
        is the entire enforcement: a composer stop traceable only to an
        unsearched payload stops qualifying as grounded and is demoted to the
        weakest claim. The composer still SEES those payloads -- they are
        often correct, and dropping them would thin results -- they just can
        no longer back a literal/historical claim.
        """
        skip = set(self.unverified_discovery)
        parts: List[str] = []
        for key, value in self._discovery_payloads():
            if key in skip:
                continue
            if isinstance(value, str):
                parts.append(value)
            else:
                try:
                    parts.append(json.dumps(value, ensure_ascii=False, default=str))
                except (TypeError, ValueError):
                    parts.append(str(value))
        return "\n".join(parts)

    @property
    def last_book_recommendations(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.LAST_BOOK_RECOMMENDATIONS)

    @property
    def last_place_to_book(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.LAST_PLACE_TO_BOOK)

    @property
    def book_recommendation_count(self) -> int:
        return int(self._state.get(SessionStateKeys.BOOK_RECOMMENDATION_COUNT, 0))

    @property
    def book_recs_in_progress(self) -> bool:
        return bool(self._state.get(SessionStateKeys.BOOK_RECS_IN_PROGRESS, False))

    @property
    def book_recommendation_chip_id(self) -> Optional[str]:
        return self._state.get(SessionStateKeys.BOOK_RECOMMENDATION_CHIP_ID)

    @property
    def book_recommendation_chip(self) -> Optional[dict]:
        return self._state.get(SessionStateKeys.BOOK_RECOMMENDATION_CHIP)
