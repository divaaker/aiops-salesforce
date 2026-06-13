"""
RuleBasedLLM — deterministic, offline conversational responder for dev & tests.

It is intentionally small but behaves like a real LLM stage: it reads the user
text plus prior `history` and returns a reply. Deterministic output is what lets
the integration tests assert exact conversation transcripts.
"""
from __future__ import annotations

from typing import Iterator, List

from ..contracts import LLMResponse


class RuleBasedLLM:
    def __init__(self, name: str = "VoxBox") -> None:
        self.name = name

    def respond(self, text: str, history: List[dict]) -> LLMResponse:
        t = text.strip().lower()
        if not t:
            return LLMResponse("I didn't catch that — could you say it again?")
        words = {w.strip(".,!?;:") for w in t.split()}
        if words & {"hello", "hi", "hey"}:
            return LLMResponse(f"Hello! I'm {self.name}, running fully on your machine.")
        if "your name" in t or "who are you" in t:
            return LLMResponse(f"I'm {self.name}, a private local voice agent.")
        if "time" in t:
            return LLMResponse("I don't have a clock wired up, but it's always local time here.")
        if t.endswith("?"):
            return LLMResponse("That's a good question. Locally, I'd reason about it step by step.")
        turn = sum(1 for m in history if m.get("role") == "user") + 1
        return LLMResponse(f"You said: '{text.strip()}'. (turn {turn})")

    def respond_stream(self, text: str, history: List[dict]) -> Iterator[str]:
        """Yield the reply word-by-word to emulate token streaming (deterministic)."""
        full = self.respond(text, history).text
        words = full.split(" ")
        for i, w in enumerate(words):
            yield w if i == len(words) - 1 else w + " "
