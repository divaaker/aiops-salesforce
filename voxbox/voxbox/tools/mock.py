"""Deterministic tool provider + a scripted tool model, for dev/tests/demo (no
network, no MCP server, no real LLM)."""
from __future__ import annotations

from typing import List, Tuple

from .base import ToolCall, ToolSpec


class MockToolProvider:
    def list_tools(self) -> List[ToolSpec]:
        return [
            ToolSpec(
                "add", "Add two integers a and b",
                {"type": "object",
                 "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                 "required": ["a", "b"]},
            ),
            ToolSpec("get_time", "Return the current time (ISO-8601)"),
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        if name == "add":
            return str(int(arguments["a"]) + int(arguments["b"]))
        if name == "get_time":
            return "2026-06-13T20:00:00Z"
        raise KeyError(f"unknown tool: {name}")


class ScriptedToolModel:
    """A ToolModel that returns pre-scripted (text, tool_calls) steps in order —
    lets tests/demo drive the tool loop deterministically."""

    def __init__(self, steps: List[Tuple[str, List[ToolCall]]]) -> None:
        self._steps = list(steps)
        self._i = 0
        self.seen_tool_names: List[str] = []

    def chat(self, messages, tools):
        for t in tools:
            self.seen_tool_names.append(t.name)
        if self._i < len(self._steps):
            step = self._steps[self._i]
            self._i += 1
            return step
        return ("", [])
