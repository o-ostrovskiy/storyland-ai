"""https-only URLs for every URL field the model or a tool can emit (ai-private#78).

A URL that reaches a reader's browser (a cover ``<img src>``, a link) must not be
``javascript:``, ``data:`` or plain ``http:``. Anything that is not an ``https://``
URL becomes ``None``: the reader sees no cover, never a script or a mixed-content
request. be already upgrades Google Books' ``http://`` thumbnails to ``https://``
before it sends a book here, so a real cover is not lost.
"""
from __future__ import annotations

from typing import Optional


def https_only(value: Optional[str]) -> Optional[str]:
    """Return the URL (trimmed) if it is ``https://``, else ``None``."""
    if value is None:
        return None
    url = str(value).strip()
    return url if url.lower().startswith("https://") else None
