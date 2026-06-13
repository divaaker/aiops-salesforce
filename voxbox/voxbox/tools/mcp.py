"""
MCPToolProvider — exposes an MCP server's tools to the voice agent.

It wraps an already-connected MCP `ClientSession` (the official `mcp` Python SDK
handles the transport + OAuth). Connecting/authing is done by `connect()` or by
your own setup code, then the session is handed to this provider. Keeping the
provider session-agnostic makes it testable with a fake session (no SDK needed).

    pip install mcp
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .base import ToolSpec


def parse_oauth_callback(url: str) -> Tuple[str, Optional[str]]:
    """Extract (code, state) from an OAuth redirect URL like
    http://localhost:<port>/callback?code=...&state=... — used by the paste-the-URL
    auth flow in scripts/talk_mcp.py."""
    from urllib.parse import parse_qs, urlparse

    q = parse_qs(urlparse(url).query)
    code = (q.get("code") or [None])[0]
    state = (q.get("state") or [None])[0]
    if not code:
        raise ValueError("No ?code=... found in the callback URL.")
    return code, state


class MCPToolProvider:
    def __init__(self, session) -> None:
        if session is None:
            raise RuntimeError(
                "MCPToolProvider needs a connected MCP ClientSession. "
                "Use MCPToolProvider.connect(url) or the mcp SDK to obtain one."
            )
        self._session = session

    def list_tools(self) -> List[ToolSpec]:
        result = self._session.list_tools()
        return [
            ToolSpec(
                name=t.name,
                description=getattr(t, "description", "") or "",
                input_schema=getattr(t, "inputSchema", None) or {"type": "object", "properties": {}},
            )
            for t in result.tools
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._session.call_tool(name, arguments)
        parts = []
        for item in getattr(result, "content", None) or []:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else str(result)

    @staticmethod
    def connect(url: str):  # pragma: no cover - needs network + mcp SDK + OAuth
        """Convenience connector (laptop use). Returns a ready MCPToolProvider.
        Requires `pip install mcp`. For OAuth servers the SDK will open the
        browser auth flow; follow its prompts."""
        raise NotImplementedError(
            "Connect with the mcp SDK (streamablehttp_client + ClientSession, "
            "plus an OAuth provider for authed servers), call session.initialize(), "
            "then MCPToolProvider(session). See docs/MCP.md."
        )
