"""
voxbox.tools.base
=================
Tool-calling for the voice agent. The LLM stage can now call external tools (e.g.
those exposed by an MCP server) before producing the spoken reply.

Pieces:
- ToolSpec / ToolCall: the data.
- ToolProvider: where tools come from (an MCP server, or a mock).
- ToolModel: an LLM that can decide to call tools (returns text + tool calls).
- ToolCallingLLM: implements the normal LLM `respond` contract by running the
  tool loop (model -> call tools -> feed results -> ... -> final text), so it drops
  straight into the orchestrator as the `llm` stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, Tuple, runtime_checkable

from ..contracts import LLMResponse


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)


@runtime_checkable
class ToolProvider(Protocol):
    def list_tools(self) -> List[ToolSpec]: ...
    def call_tool(self, name: str, arguments: dict) -> str: ...


@runtime_checkable
class ToolModel(Protocol):
    """An LLM that can request tool calls. Returns (assistant_text, tool_calls)."""

    def chat(self, messages: List[dict], tools: List[ToolSpec]) -> Tuple[str, List[ToolCall]]: ...


class ToolCallingLLM:
    """Wraps a ToolModel + ToolProvider and exposes the plain `respond` LLM
    contract, running the agentic tool loop in between."""

    def __init__(self, model: ToolModel, provider: ToolProvider, max_steps: int = 4) -> None:
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.last_tool_calls: List[ToolCall] = []

    def respond(self, text: str, history: List[dict]) -> LLMResponse:
        messages = list(history) + [{"role": "user", "content": text}]
        tools = self.provider.list_tools()
        self.last_tool_calls = []

        for _ in range(self.max_steps):
            content, calls = self.model.chat(messages, tools)
            if not calls:
                return LLMResponse(content)
            messages.append({"role": "assistant", "content": content or ""})
            for c in calls:
                self.last_tool_calls.append(c)
                try:
                    result = self.provider.call_tool(c.name, c.arguments)
                except Exception as exc:  # surface tool errors to the model, don't crash
                    result = f"ERROR: {exc}"
                messages.append({"role": "tool", "name": c.name, "content": str(result)})

        # Out of steps — ask once more for a final answer.
        content, _ = self.model.chat(messages, tools)
        return LLMResponse(content)
