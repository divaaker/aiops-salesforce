"""
voxbox.pipeline
===============
Factory that turns a Config into a wired Orchestrator. Real (GPU) backends are
imported lazily inside each branch so that the default mock pipeline — and the
test suite — never need torch/whisper/etc. installed.
"""
from __future__ import annotations

from typing import Any, Dict

from .config import Config
from .orchestrator import Orchestrator


def _opts(cfg: Config, stage: str) -> Dict[str, Any]:
    return dict(cfg.options.get(stage, {}))


def _build_vad(cfg: Config):
    if cfg.vad == "energy":
        from .vad import EnergyVAD
        return EnergyVAD(**_opts(cfg, "vad"))
    if cfg.vad == "silero":
        from .vad.silero import SileroVAD
        return SileroVAD(**_opts(cfg, "vad"))
    raise ValueError(f"Unknown vad backend: {cfg.vad!r}")


def _build_stt(cfg: Config):
    if cfg.stt == "mock":
        from .stt import MockSTT
        return MockSTT(**_opts(cfg, "stt"))
    if cfg.stt == "faster_whisper":
        from .stt.faster_whisper import FasterWhisperSTT
        return FasterWhisperSTT(**_opts(cfg, "stt"))
    raise ValueError(f"Unknown stt backend: {cfg.stt!r}")


def _build_llm(cfg: Config):
    if cfg.llm == "mock":
        from .llm import RuleBasedLLM
        return RuleBasedLLM(**_opts(cfg, "llm"))
    if cfg.llm == "ollama":
        from .llm.ollama import OllamaLLM
        return OllamaLLM(**_opts(cfg, "llm"))
    raise ValueError(f"Unknown llm backend: {cfg.llm!r}")


def _build_tts(cfg: Config):
    if cfg.tts == "mock":
        from .tts import MockTTS
        return MockTTS(**_opts(cfg, "tts"))
    if cfg.tts == "piper":
        from .tts.piper import PiperTTS
        return PiperTTS(**_opts(cfg, "tts"))
    raise ValueError(f"Unknown tts backend: {cfg.tts!r}")


def build_pipeline(cfg: Config | None = None) -> Orchestrator:
    cfg = cfg or Config()
    return Orchestrator(
        vad=_build_vad(cfg),
        stt=_build_stt(cfg),
        llm=_build_llm(cfg),
        tts=_build_tts(cfg),
        budget_ms=cfg.budget_ms,
    )
