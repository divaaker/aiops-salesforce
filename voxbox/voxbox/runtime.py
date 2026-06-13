"""
voxbox.runtime
==============
Glue for running the agent live: build a Config from environment variables (so the
same scripts work all-mock or all-real), and drive the capture→orchestrate→play
loop. The loop is a plain function taking any iterable source and any object with
`.play(AudioReply)`, which is what makes it testable without hardware.
"""
from __future__ import annotations

import os
from typing import Callable, Iterable, Optional

from .config import Config
from .contracts import AudioChunk
from .orchestrator import Orchestrator, TurnResult


def config_from_env(env: Optional[dict] = None) -> Config:
    e = env if env is not None else os.environ
    options: dict = {}

    stt = e.get("VOX_STT", "mock")
    if stt == "faster_whisper":
        o = {"device": e.get("VOX_WHISPER_DEVICE", "auto"),
             "compute_type": e.get("VOX_WHISPER_COMPUTE", "default")}
        if e.get("VOX_WHISPER_MODEL"):
            o["model_size"] = e["VOX_WHISPER_MODEL"]
        options["stt"] = o

    llm = e.get("VOX_LLM", "mock")
    if llm == "ollama":
        o = {}
        if e.get("VOX_OLLAMA_MODEL"):
            o["model"] = e["VOX_OLLAMA_MODEL"]
        if e.get("VOX_OLLAMA_HOST"):
            o["host"] = e["VOX_OLLAMA_HOST"]
        options["llm"] = o

    tts = e.get("VOX_TTS", "mock")
    if tts == "piper":
        o = {}
        if e.get("VOX_PIPER_VOICE"):
            o["voice_path"] = e["VOX_PIPER_VOICE"]
        options["tts"] = o

    return Config(
        vad=e.get("VOX_VAD", "energy"),
        stt=stt,
        llm=llm,
        tts=tts,
        budget_ms=float(e.get("VOX_BUDGET_MS", "1200")),
        options=options,
    )


def run_local(
    orch: Orchestrator,
    source: Iterable[AudioChunk],
    speaker,
    on_turn: Optional[Callable[[TurnResult], None]] = None,
    max_turns: Optional[int] = None,
) -> int:
    """Drive mic→orchestrator→speaker until the source ends (or max_turns).
    Returns the number of completed turns. `speaker` only needs `.play(reply)`."""
    turns = 0
    for chunk in source:
        result = orch.feed(chunk)
        if result is None:
            continue
        if on_turn is not None:
            on_turn(result)
        speaker.play(result.reply)
        turns += 1
        if max_turns is not None and turns >= max_turns:
            break
    return turns
