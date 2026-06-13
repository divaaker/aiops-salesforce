"""
voxbox.engine
=============
ConversationEngine — the real-time driver that runs reply generation on a worker
thread so the mic loop never blocks. This is what enables *mid-response* barge-in:
when the user talks over the agent we both cancel further generation AND cut the
audio that's currently playing.

Design for testability: the per-chunk decision logic (`process_chunk`) is pure and
synchronous, and the generation worker is created via an injectable
`handle_factory`. Tests drive the logic with a fake handle; the real thread-based
handle is exercised by a separate smoke test that simply runs to completion.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .contracts import AudioChunk
from .orchestrator import Orchestrator
from .streaming import StreamChunk, StreamDone
from .vad.base import rms_dbfs


class OnsetDetector:
    """Fires after `onset_frames` consecutive loud chunks. While the agent is busy
    (speaking/generating) the threshold is raised by `playback_margin_dbfs` so the
    agent's own (quieter) audio doesn't self-trigger — a lightweight echo mitigation
    until real AEC lands (see ROADMAP)."""

    def __init__(self, threshold_dbfs: float = -40.0, onset_frames: int = 3,
                 playback_margin_dbfs: float = 6.0) -> None:
        self.threshold_dbfs = threshold_dbfs
        self.onset_frames = onset_frames
        self.playback_margin_dbfs = playback_margin_dbfs
        self._loud = 0

    def reset(self) -> None:
        self._loud = 0

    def onset(self, chunk: AudioChunk, busy: bool) -> bool:
        thr = self.threshold_dbfs + (self.playback_margin_dbfs if busy else 0.0)
        if rms_dbfs(chunk.pcm) >= thr:
            self._loud += 1
        else:
            self._loud = 0
        if self._loud >= self.onset_frames:
            self._loud = 0
            return True
        return False


@dataclass
class _ThreadHandle:
    thread: threading.Thread
    cancel_event: threading.Event

    def cancel(self) -> None:
        self.cancel_event.set()

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        self.thread.join(timeout)


class ConversationEngine:
    def __init__(
        self,
        orch: Orchestrator,
        speaker,
        barge_in: bool = True,
        threshold_dbfs: float = -40.0,
        onset_frames: int = 3,
        playback_margin_dbfs: float = 6.0,
        on_chunk: Optional[Callable[[StreamChunk], None]] = None,
        on_turn: Optional[Callable[[StreamDone], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        handle_factory: Optional[Callable] = None,
    ) -> None:
        self.orch = orch
        self.speaker = speaker
        self.barge_in = barge_in
        self.on_chunk = on_chunk
        self.on_turn = on_turn
        self.on_interrupt = on_interrupt
        self.on_error = on_error
        self._detector = OnsetDetector(threshold_dbfs, onset_frames, playback_margin_dbfs)
        self._handle_factory = handle_factory or self._spawn_generation
        self._gen = None

    def _busy(self) -> bool:
        speaking = bool(getattr(self.speaker, "is_active", lambda: False)())
        generating = self._gen is not None and self._gen.is_alive()
        return speaking or generating

    def _cancel_current(self) -> None:
        if self._gen is not None:
            self._gen.cancel()
        self.speaker.stop()

    def process_chunk(self, chunk: AudioChunk) -> bool:
        """Handle one mic chunk. Returns True if a new turn was started."""
        busy = self._busy()
        if self.barge_in and busy:
            if self._detector.onset(chunk, busy=True):
                self._cancel_current()
                if self.on_interrupt is not None:
                    self.on_interrupt()
        else:
            self._detector.reset()

        segment = self.orch.detect(chunk)
        if segment is None:
            return False

        # A fresh utterance supersedes anything still generating.
        if self._gen is not None and self._gen.is_alive():
            self._cancel_current()
        self._gen = self._handle_factory(segment)
        return True

    def run(self, source: Iterable[AudioChunk], max_turns: Optional[int] = None) -> int:
        turns = 0
        for chunk in source:
            if self.process_chunk(chunk):
                turns += 1
                if max_turns is not None and turns >= max_turns:
                    break
        if self._gen is not None:
            self._gen.join(timeout=5.0)
        return turns

    def _spawn_generation(self, segment):
        cancel = threading.Event()

        def _work():
            try:
                for ev in self.orch.stream_segment(segment):
                    if cancel.is_set():
                        break
                    if isinstance(ev, StreamChunk):
                        self.speaker.play(ev.reply)
                        if self.on_chunk is not None:
                            self.on_chunk(ev)
                    elif self.on_turn is not None:
                        self.on_turn(ev)
            except Exception as exc:  # a backend (LLM/STT/TTS) failed mid-turn
                if self.on_error is not None:
                    self.on_error(exc)
                else:
                    raise

        t = threading.Thread(target=_work, daemon=True)
        t.start()
        return _ThreadHandle(t, cancel)
