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
from .streaming import StreamChunk, StreamDone
from .vad.base import rms_dbfs


def config_from_env(env: Optional[dict] = None) -> Config:
    e = env if env is not None else os.environ
    options: dict = {}

    vad = e.get("VOX_VAD", "energy")
    if vad == "energy":
        o = {}
        if e.get("VOX_VAD_THRESHOLD"):
            o["threshold_dbfs"] = float(e["VOX_VAD_THRESHOLD"])
        if e.get("VOX_VAD_MIN_SPEECH_MS"):
            o["min_speech_ms"] = float(e["VOX_VAD_MIN_SPEECH_MS"])
        if e.get("VOX_VAD_HANGOVER_MS"):
            o["hangover_ms"] = float(e["VOX_VAD_HANGOVER_MS"])
        if o:
            options["vad"] = o

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
        vad=vad,
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


class BargeInController:
    """Watches mic chunks and interrupts an interruptible speaker the moment the
    user starts talking over the agent.

    It only acts while the speaker is active (`speaker.is_active()`), and requires
    `onset_frames` consecutive loud chunks to fire — short blips/echo won't trip it.
    Assumes the mic doesn't pick up the agent's own audio (headphones or echo
    cancellation); real AEC is out of scope (see ROADMAP).
    """

    def __init__(self, speaker, threshold_dbfs: float = -40.0, onset_frames: int = 3) -> None:
        self.speaker = speaker
        self.threshold_dbfs = threshold_dbfs
        self.onset_frames = onset_frames
        self._loud = 0

    def on_chunk(self, chunk: AudioChunk) -> bool:
        """Returns True if this chunk triggered an interruption."""
        if not self.speaker.is_active():
            self._loud = 0
            return False
        if rms_dbfs(chunk.pcm) >= self.threshold_dbfs:
            self._loud += 1
        else:
            self._loud = 0
        if self._loud >= self.onset_frames:
            self.speaker.stop()
            self._loud = 0
            return True
        return False


def run_conversation(
    orch: Orchestrator,
    source: Iterable[AudioChunk],
    speaker,
    barge_in: bool = True,
    threshold_dbfs: float = -40.0,
    onset_frames: int = 3,
    on_turn: Optional[Callable[[TurnResult], None]] = None,
    on_interrupt: Optional[Callable[[], None]] = None,
    max_turns: Optional[int] = None,
) -> int:
    """Like run_local, but with barge-in: while the agent is speaking, the user can
    talk over it to cut it off. Needs an interruptible speaker (play/stop/is_active)."""
    controller = BargeInController(speaker, threshold_dbfs, onset_frames) if barge_in else None
    turns = 0
    for chunk in source:
        if controller is not None and controller.on_chunk(chunk) and on_interrupt is not None:
            on_interrupt()
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


def run_conversation_streaming(
    orch: Orchestrator,
    source: Iterable[AudioChunk],
    speaker,
    barge_in: bool = True,
    threshold_dbfs: float = -40.0,
    onset_frames: int = 3,
    on_chunk: Optional[Callable[[StreamChunk], None]] = None,
    on_turn: Optional[Callable[[StreamDone], None]] = None,
    on_interrupt: Optional[Callable[[], None]] = None,
    max_turns: Optional[int] = None,
) -> int:
    """Live loop with streaming replies: each completed utterance is streamed
    sentence-by-sentence and enqueued on a QueueingSpeaker, so the agent starts
    talking sooner. Barge-in still works: the user's next utterance flushes any
    queued/playing sentences before the new turn is handled.

    Note: this single-threaded loop checks barge-in between turns; interrupting
    *during* token generation needs the generation moved off-thread (see ROADMAP).
    """
    controller = BargeInController(speaker, threshold_dbfs, onset_frames) if barge_in else None
    turns = 0
    for chunk in source:
        if controller is not None and controller.on_chunk(chunk) and on_interrupt is not None:
            on_interrupt()
        segment = orch.detect(chunk)
        if segment is None:
            continue
        for ev in orch.stream_segment(segment):
            if isinstance(ev, StreamChunk):
                speaker.play(ev.reply)
                if on_chunk is not None:
                    on_chunk(ev)
            elif on_turn is not None:
                on_turn(ev)
        turns += 1
        if max_turns is not None and turns >= max_turns:
            break
    return turns
