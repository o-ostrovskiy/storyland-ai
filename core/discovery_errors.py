"""Typed discovery-compose errors + a classifier for the async-TaskGroup boundary.

The discovery composition runs its parallel work under an async ``TaskGroup``
(inside the ADK runner). When a child task raises, Python wraps the failure in
an ``ExceptionGroup`` whose ``str()`` is the opaque
``"unhandled errors in a TaskGroup (1 sub-exception)"``. That raw string — and
any other internal async machinery — must never reach a client.

This module collapses any such failure into a single, client-safe
``DiscoveryComposeError`` classified as one of two kinds:

* ``transient`` — an upstream/timeout/child-crash compose failure (the
  intermittent case; an immediate retry with another title composes fine).
* ``taste_validation`` — a ``taste_context`` title genuinely failed validation
  (the deterministic soft-trap case; re-fails identically on retry). Carries the
  offending title so the caller can offer a "search without my taste" escape.

This PR *defines the typed-error contract*; the gateway (be) maps ``kind`` →
HTTP status + a machine-readable ``reason`` in the body, and fe branches on
``reason`` to pick the right recovery state. Do not over-broaden
``taste_validation`` — everything that is not a genuine taste-title validation
failure is ``transient``, so the soft-trap state is never shown spuriously.
"""

from __future__ import annotations

from typing import Iterator, Optional

try:  # Python 3.11+: ExceptionGroup / BaseExceptionGroup are builtins.
    _BASE_EXCEPTION_GROUP: type = BaseExceptionGroup  # type: ignore[name-defined]
except NameError:  # pragma: no cover - py3.10 has no group semantics
    _BASE_EXCEPTION_GROUP = ()  # type: ignore[assignment]

# Literal kinds the contract exposes. (Kept as a tuple for runtime validation;
# the gateway/fe treat these as the machine-readable ``reason`` values.)
DISCOVERY_ERROR_KINDS = ("transient", "taste_validation")

# Client-safe messages. NEVER embed the raw exception / ExceptionGroup string.
TRANSIENT_MESSAGE = (
    "We couldn't finish composing your journey this time. This is almost "
    "always temporary — please try again."
)
TASTE_VALIDATION_MESSAGE = (
    "One of your saved books is tripping up this search."
)


class TasteContextValidationError(ValueError):
    """A specific ``taste_context`` title failed validation.

    Raised (or forwarded) when a stored/imported taste title cannot be used to
    shape discovery — deterministic, so the same ``taste_context`` re-fails
    identically until the offending title is removed. Carries ``offending_title``
    so the compose boundary can surface a ``taste_validation`` error with an
    escape hatch instead of an opaque transient failure.
    """

    def __init__(
        self, offending_title: Optional[str] = None, message: str = ""
    ) -> None:
        self.offending_title = offending_title
        super().__init__(message or "taste_context title failed validation")


class DiscoveryComposeError(Exception):
    """A single, client-safe error for a failed discovery composition.

    ``kind`` is ``"transient"`` (a retry may succeed) or ``"taste_validation"``
    (a stored taste_context title is the cause; a plain retry re-fails).
    ``message`` is always a friendly, internals-free string; the real
    stacktrace is logged server-side only and never carried here.
    """

    def __init__(
        self,
        kind: str,
        message: str,
        offending_title: Optional[str] = None,
    ) -> None:
        if kind not in DISCOVERY_ERROR_KINDS:
            raise ValueError(f"unknown DiscoveryComposeError kind: {kind!r}")
        self.kind = kind
        self.message = message
        self.offending_title = offending_title
        super().__init__(message)


def _iter_leaves(exc: BaseException) -> Iterator[BaseException]:
    """Yield the leaf exceptions of a (possibly nested) ExceptionGroup.

    A non-group exception yields itself; a group is unwrapped recursively so a
    ``TaskGroup``-wrapped child failure is inspected directly.
    """
    if _BASE_EXCEPTION_GROUP and isinstance(exc, _BASE_EXCEPTION_GROUP):
        for sub in exc.exceptions:  # type: ignore[attr-defined]
            yield from _iter_leaves(sub)
    else:
        yield exc


def _find_taste_validation(
    exc: BaseException,
) -> Optional[TasteContextValidationError]:
    for leaf in _iter_leaves(exc):
        if isinstance(leaf, TasteContextValidationError):
            return leaf
    return None


def classify_discovery_failure(
    exc: BaseException,
    *,
    taste_context: Optional[dict] = None,
) -> DiscoveryComposeError:
    """Collapse any discovery-compose failure into a typed, client-safe error.

    Unwraps an async-``TaskGroup`` ``ExceptionGroup`` to its leaves and
    classifies: a ``TasteContextValidationError`` leaf → ``taste_validation``
    (with the offending title, falling back to the first taste_context title
    when the leaf did not record one); anything else → ``transient``. The
    returned error's ``message`` never contains the raw exception text.
    """
    if isinstance(exc, DiscoveryComposeError):
        return exc

    taste_err = _find_taste_validation(exc)
    if taste_err is not None:
        offending = taste_err.offending_title
        if offending is None and taste_context:
            titles = taste_context.get("titles") or []
            offending = titles[0] if titles else None
        return DiscoveryComposeError(
            kind="taste_validation",
            message=TASTE_VALIDATION_MESSAGE,
            offending_title=offending,
        )

    return DiscoveryComposeError(kind="transient", message=TRANSIENT_MESSAGE)
