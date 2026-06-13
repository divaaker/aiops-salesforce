"""
Real LLM adapter backed by Ollama serving Gemma (the model shown in the diagram).

Talks to a local Ollama daemon over HTTP — still 100% on-device. Uses stdlib
urllib so it has no hard dependency; the daemon must be reachable.
"""
from __future__ import annotations

import json
from typing import List
from urllib import request

from ..contracts import LLMResponse


class OllamaLLM:
    def __init__(self, model: str = "gemma3", host: str = "http://localhost:11434",
                 system: str = "You are a concise, helpful local voice assistant.") -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.system = system

    def respond(self, text: str, history: List[dict]) -> LLMResponse:  # pragma: no cover - needs daemon
        messages = [{"role": "system", "content": self.system}]
        messages.extend(history)
        messages.append({"role": "user", "content": text})
        payload = json.dumps(
            {"model": self.model, "messages": messages, "stream": False}
        ).encode()
        req = request.Request(
            f"{self.host}/api/chat", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return LLMResponse(text=data["message"]["content"].strip())
