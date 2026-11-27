"""Where a run came from, for Langfuse (storyland-services#266).

The gateway's MCP server (storyland-services ``app/mcp``) sends
``X-Request-Source: mcp`` plus ``X-MCP-Tool`` / ``X-MCP-Client`` /
``X-MCP-Client-Version`` / ``X-MCP-Session`` on the discover and compose calls a
tool makes. The website sends none of them, so a web run has no source and its
trace is unchanged.

The values are labels, never identity: ``X-User-ID`` stays the only principal.
They are trusted only as far as every other header behind the gateway secret,
and are re-cleaned here because they become Langfuse tags (short, ASCII).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

MCP_SOURCE = "mcp"
# Session-state key the job keeps its source under (SessionStateKeys.REQUEST_SOURCE).
STATE_KEY = "request_source"

_UNSAFE = re.compile(r"[^A-Za-z0-9._@/+-]+")
_MAX_LEN = 64


def _clean(value: Any, default: str) -> str:
    if not isinstance(value, str):
        return default
    cleaned = _UNSAFE.sub("-", value.strip()).strip("-")[:_MAX_LEN]
    return cleaned or default


@dataclass(frozen=True)
class RequestSource:
    """An MCP-started run: which tool, which client, which MCP session."""

    tool: str
    client: str
    client_version: str
    session: str

    @classmethod
    def mcp(
        cls,
        tool: Optional[str],
        client: Optional[str],
        client_version: Optional[str],
        session: Optional[str],
    ) -> "RequestSource":
        return cls(
            tool=_clean(tool, "unknown"),
            client=_clean(client, "unknown"),
            client_version=_clean(client_version, "unknown"),
            session=_clean(session, "none"),
        )

    def to_state(self) -> dict[str, str]:
        """The session-state form, so compose (and any later step on the same
        job) traces with the source discover was called with."""
        return {
            "source": MCP_SOURCE,
            "tool": self.tool,
            "client": self.client,
            "client_version": self.client_version,
            "session": self.session,
        }

    @classmethod
    def from_state(cls, value: Any) -> Optional["RequestSource"]:
        """Read back what ``to_state`` wrote; anything else is no source."""
        if not isinstance(value, dict) or value.get("source") != MCP_SOURCE:
            return None
        return cls.mcp(
            value.get("tool"),
            value.get("client"),
            value.get("client_version"),
            value.get("session"),
        )

    def langfuse_tags(self) -> list[str]:
        return [MCP_SOURCE, f"tool:{self.tool}", f"client:{self.client}"]

    def langfuse_metadata(self) -> dict[str, str]:
        return {
            "source": MCP_SOURCE,
            "mcp_tool": self.tool,
            "mcp_client": self.client,
            "mcp_client_version": self.client_version,
            "mcp_session": self.session,
        }
