"""
voxbox.config
=============
Declarative selection of which backend implements each stage. This is the seam
that makes the architecture's "swap mock <-> real model" promise a config change.

Example:
    Config(vad="energy", stt="mock", llm="mock", tts="mock")          # dev / CI
    Config(vad="silero", stt="faster_whisper", llm="ollama", tts="piper")  # GPU box
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    vad: str = "energy"
    stt: str = "mock"
    llm: str = "mock"
    tts: str = "mock"
    budget_ms: float = 1200.0
    options: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        known = {f for f in cls().__dict__}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)
