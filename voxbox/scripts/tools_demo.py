"""
Demo of the voice agent's tool-calling, all mocked (no network/LLM/MCP). Shows the
LLM stage deciding to call a tool, the tool running, and the final reply — the same
path a real MCP server's tools would take.

    python scripts/tools_demo.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxbox.tools import MockToolProvider, ScriptedToolModel, ToolCall, ToolCallingLLM  # noqa: E402


def main() -> None:
    provider = MockToolProvider()
    print("Discovered tools:", [t.name for t in provider.list_tools()], "\n")

    # Scripted model: first turn asks to call add(2,3); second turn gives the answer.
    model = ScriptedToolModel([
        ("", [ToolCall("add", {"a": 2, "b": 3})]),
        ("Two plus three is 5.", []),
    ])
    llm = ToolCallingLLM(model, provider)

    user = "what is two plus three?"
    print(f"🗣️  user : {user}")
    reply = llm.respond(user, [])
    print(f"🔧 called: {[c.name + str(c.arguments) for c in llm.last_tool_calls]}")
    print(f"🤖 agent : {reply.text}")


if __name__ == "__main__":
    main()
