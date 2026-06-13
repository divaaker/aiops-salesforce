# Using MCP tools in the voice agent

VoxBox's LLM stage can call **tools** — including those exposed by an MCP server —
before it speaks. The flow:

```
your voice → STT → LLM decides to call a tool → ToolProvider runs it
          → result fed back to LLM → final text → TTS → speaker
```

Pieces (`voxbox/tools/`): `ToolProvider` (where tools come from), `ToolModel` (a
tool-capable LLM), and `ToolCallingLLM` (runs the loop, drops into the orchestrator
as the `llm` stage). Backends: `MockToolProvider` (built-in `add` / `get_time`) and
`MCPToolProvider` (your server). The tool-capable model is `OllamaToolModel`.

## 1. Test the tool flow now — no MCP, no auth (recommended first)
This proves the whole voice→tool→speech path with built-in mock tools. You need a
**tool-capable** Ollama model (gemma3 often isn't — use llama3.1/qwen2.5):
```bash
ollama pull llama3.1
VOX_TOOLS=mock VOX_TOOLS_MODEL=llama3.1 \
VOX_STT=faster_whisper VOX_LLM=ollama VOX_TTS=piper \
VOX_PIPER_VOICE=en_US-amy-medium.onnx VOX_VAD_THRESHOLD=-33 python scripts/talk.py
```
Then say **"what is two plus three?"** — you'll see `🔧 tools enabled`, the agent
calls `add`, and speaks "...is 5." That confirms tool-calling works end-to-end.

Headless check (no mic): `python scripts/tools_demo.py`.

## 2. Point it at YOUR MCP server
Your laptop reaches the tunnel directly (the Claude-sandbox egress limit doesn't
apply locally). You connect with the MCP SDK (it handles OAuth), then hand the
session to `MCPToolProvider`:

```bash
pip install mcp
```
```python
# connect_my_mcp.py  (sketch — fill in per the mcp SDK version you install)
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from voxbox.tools.mcp import MCPToolProvider

async def main():
    url = "https://safari-managing-wireless-boot.trycloudflare.com/mcp"
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()           # OAuth flow happens here
            provider = MCPToolProvider(session)
            print([t.name for t in provider.list_tools()])
            # wire provider into ToolCallingLLM + the orchestrator (see scripts/talk.py)
```
Notes:
- For an **OAuth** server, pass an auth provider to the client per the SDK docs; the
  first run opens your browser to authorize.
- The async MCP session must be driven from an event loop; the simplest integration
  is to run the agent loop inside the same async context, or use a small sync bridge.
  Once you have a working `provider`, set `orch.llm = ToolCallingLLM(OllamaToolModel(...),
  provider)` exactly like `VOX_TOOLS=mock` does in `scripts/talk.py`.

## Why not from a Claude Code web session?
The remote sandbox blocks outbound traffic to non-allowlisted hosts (you'd see
`403 Host not in allowlist`). Either add `*.trycloudflare.com` under **Custom**
network access (see docs/TUNING.md links) or just run on your laptop, where there's
no such restriction.

## Troubleshooting
- **Model never calls tools** → use a tool-capable model (`VOX_TOOLS_MODEL=llama3.1`/`qwen2.5`).
- **Connection refused** → Ollama not running (`brew services start ollama`).
- **403 from the MCP server** → you're inside the Claude sandbox; run locally or allowlist the host.
