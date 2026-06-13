"""
Live voice agent wired to YOUR MCP server's tools.

Run on your LAPTOP (it reaches the tunnel directly; the Claude sandbox can't).
This connects to the MCP server with the `mcp` SDK (handling OAuth via a
paste-the-callback-URL flow that works headless), exposes its tools to the voice
agent, and runs the mic loop:

    voice -> Whisper -> LLM (calls MCP tools) -> Piper -> speaker

Setup:
    pip install mcp
    ollama pull llama3.1            # a tool-capable model
Run:
    VOX_STT=faster_whisper VOX_LLM=ollama VOX_TTS=piper \
    VOX_PIPER_VOICE=en_US-amy-medium.onnx VOX_VAD_THRESHOLD=-33 \
    MCP_URL=https://safari-managing-wireless-boot.trycloudflare.com/mcp \
    python scripts/talk_mcp.py

The `mcp` SDK's API shifts between versions; this is best-effort. If a call fails,
paste the error and we'll adjust the import/handshake.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxbox.tools.mcp import MCPToolProvider, parse_oauth_callback  # noqa: E402


# ── async MCP session running on its own event loop thread ────────────────────


class MCPRuntime:
    """Owns a background asyncio loop that keeps the MCP session open, and lets
    synchronous code (the voice agent's tool calls) drive it via run_coroutine_threadsafe."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.loop = asyncio.new_event_loop()
        self.session = None
        self._ready = threading.Event()
        self._error = None
        self._stop = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main())
        except Exception as exc:  # surface connect/auth failures to the constructor
            self._error = exc
            self._ready.set()

    async def _main(self) -> None:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        auth = await self._make_auth()
        async with streamablehttp_client(self.url, auth=auth) as transport:
            read, write = transport[0], transport[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.session = session
                self._stop = asyncio.Event()
                self._ready.set()
                await self._stop.wait()

    async def _make_auth(self):
        """Build an OAuth provider with a paste-the-URL callback. Returns None if the
        SDK auth pieces aren't available (server may not need auth / token cached)."""
        try:
            from mcp.client.auth import OAuthClientProvider, TokenStorage
            from mcp.shared.auth import OAuthClientMetadata
            from pydantic import AnyUrl
        except Exception:
            print("ℹ️  mcp OAuth classes not found; trying without explicit auth.")
            return None

        class _MemStorage(TokenStorage):
            def __init__(self):
                self._t = None
                self._c = None

            async def get_tokens(self):
                return self._t

            async def set_tokens(self, tokens):
                self._t = tokens

            async def get_client_info(self):
                return self._c

            async def set_client_info(self, info):
                self._c = info

        async def redirect_handler(auth_url: str) -> None:
            print(f"\n🔐 Authorize VoxBox in your browser:\n   {auth_url}\n")
            try:
                webbrowser.open(auth_url)
            except Exception:
                pass

        async def callback_handler():
            print("After approving, your browser redirects to a localhost URL that")
            print("fails to load — copy that FULL URL from the address bar.")
            pasted = input("Paste callback URL: ").strip()
            return parse_oauth_callback(pasted)

        metadata = OAuthClientMetadata(
            client_name="VoxBox",
            redirect_uris=[AnyUrl("http://localhost:8765/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
        return OAuthClientProvider(
            server_url=self.url,
            client_metadata=metadata,
            storage=_MemStorage(),
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )

    def call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def stop(self) -> None:
        if self._stop is not None:
            self.loop.call_soon_threadsafe(self._stop.set)


class _SyncSession:
    """Adapts the async MCP session to the sync interface MCPToolProvider expects."""

    def __init__(self, rt: MCPRuntime) -> None:
        self._rt = rt

    def list_tools(self):
        return self._rt.call(self._rt.session.list_tools())

    def call_tool(self, name, arguments):
        return self._rt.call(self._rt.session.call_tool(name, arguments))


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    url = os.environ.get("MCP_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not url:
        print("usage: MCP_URL=https://.../mcp python scripts/talk_mcp.py")
        sys.exit(2)

    try:
        import mcp  # noqa: F401
    except ImportError:
        print("❌ Needs the MCP SDK: pip install mcp")
        sys.exit(1)

    print(f"🔌 connecting to MCP server {url} ...")
    try:
        rt = MCPRuntime(url)
    except Exception as exc:
        print(f"❌ MCP connect/auth failed: {exc}")
        print("   Paste this error back and we'll adjust for your mcp SDK version.")
        sys.exit(1)

    provider = MCPToolProvider(_SyncSession(rt))
    tools = provider.list_tools()
    print(f"✅ connected. MCP tools: {[t.name for t in tools]}\n")

    # Build the voice pipeline and wrap the LLM with tool-calling.
    from voxbox.pipeline import build_pipeline
    from voxbox.runtime import config_from_env, live_env_defaults
    from voxbox.llm.ollama import OllamaToolModel
    from voxbox.tools import ToolCallingLLM
    from voxbox.engine import ConversationEngine
    from voxbox.audio.live import MicSource, QueueingSpeaker

    cfg = config_from_env(live_env_defaults())
    orch = build_pipeline(cfg)
    orch.llm = ToolCallingLLM(
        OllamaToolModel(model=os.environ.get("VOX_TOOLS_MODEL", "llama3.1"),
                        host=os.environ.get("VOX_OLLAMA_HOST", "http://localhost:11434")),
        provider,
    )

    print("🎙️  VoxBox live with MCP tools. Speak, then pause. Ctrl-C to quit.\n")
    engine = ConversationEngine(
        orch, QueueingSpeaker(),
        on_chunk=lambda c: print(f"  🔊 {c.text}"),
        on_turn=lambda d: print(f"  🗣️ heard: {d.transcript.text!r}   "
                                f"⏱ {d.metrics.total_ms:.0f}ms\n"),
        on_interrupt=lambda: print("  ✋ (interrupted)\n"),
        on_error=lambda e: print(f"  ⚠ {e}\n"),
    )
    try:
        engine.run(MicSource())
    except KeyboardInterrupt:
        print("\n👋 bye")
    finally:
        rt.stop()


if __name__ == "__main__":
    main()
