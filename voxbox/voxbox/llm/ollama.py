"""
Real LLM adapter backed by Ollama serving Gemma (the model shown in the diagram).

Talks to a local Ollama daemon over HTTP — still 100% on-device. Uses stdlib
urllib so it has no hard dependency; only a running daemon is required.

Local quickstart:
    1. Install Ollama (https://ollama.com), then:  ollama pull gemma3
    2. The daemon listens on http://localhost:11434 by default.
    3. Config(llm="ollama", options={"llm": {"model": "gemma3"}})
"""
from __future__ import annotations

import json
from typing import List
from urllib import error, request

from ..contracts import LLMResponse


class OllamaError(RuntimeError):
    """Raised when the local Ollama daemon is unreachable or returns an error."""


class OllamaLLM:
    def __init__(
        self,
        model: str = "gemma3",
        host: str = "http://localhost:11434",
        system: str = "You are a concise, helpful local voice assistant.",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.system = system
        self.timeout = timeout

    def is_available(self) -> bool:
        """True if the daemon answers. Use this for a friendly preflight check."""
        try:
            with request.urlopen(f"{self.host}/api/tags", timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def respond(self, text: str, history: List[dict]) -> LLMResponse:
        messages = [{"role": "system", "content": self.system}]
        messages.extend(history)
        messages.append({"role": "user", "content": text})
        payload = json.dumps(
            {"model": self.model, "messages": messages, "stream": False}
        ).encode()
        req = request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except error.URLError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"Try `ollama serve` and `ollama pull {self.model}`. ({exc})"
            ) from exc
        try:
            return LLMResponse(text=data["message"]["content"].strip())
        except (KeyError, TypeError) as exc:
            raise OllamaError(f"Unexpected Ollama response: {data!r}") from exc
