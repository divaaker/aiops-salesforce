"""
EnergyVAD — a dependency-free voice activity detector.

It is "mock" only in that it uses simple RMS energy instead of the Silero neural
net, but the *segmentation behaviour* (speech bounded by trailing silence, with a
minimum-speech guard) is real and is exactly what the orchestrator relies on. This
lets the full turn loop run and be tested without a GPU.
"""
from __future__ import annotations

from typing import Optional

from ..contracts import AudioChunk, SpeechSegment
from .base import rms_dbfs


class EnergyVAD:
    def __init__(
        self,
        threshold_dbfs: float = -40.0,
        min_speech_ms: float = 100.0,
        hangover_ms: float = 300.0,
    ) -> None:
        self.threshold_dbfs = threshold_dbfs
        self.min_speech_ms = min_speech_ms
        self.hangover_ms = hangover_ms
        self.reset()

    def reset(self) -> None:
        self._buf = bytearray()
        self._sample_rate = 16000
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._in_speech = False
        self._t_start = 0.0
        self._label: Optional[str] = None

    def process(self, chunk: AudioChunk) -> Optional[SpeechSegment]:
        is_speech = rms_dbfs(chunk.pcm) >= self.threshold_dbfs
        self._sample_rate = chunk.sample_rate

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._t_start = chunk.ts
                self._buf = bytearray()
                self._speech_ms = 0.0
                self._silence_ms = 0.0
            self._buf.extend(chunk.pcm)
            self._speech_ms += chunk.duration_ms
            self._silence_ms = 0.0
            if chunk.label is not None:
                self._label = chunk.label
            return None

        # silence
        if self._in_speech:
            self._buf.extend(chunk.pcm)  # keep the trailing silence for natural endings
            self._silence_ms += chunk.duration_ms
            if self._silence_ms >= self.hangover_ms:
                return self._emit(chunk.ts)
        return None

    def _emit(self, t_end: float) -> Optional[SpeechSegment]:
        seg = None
        if self._speech_ms >= self.min_speech_ms:
            seg = SpeechSegment(
                pcm=bytes(self._buf),
                sample_rate=self._sample_rate,
                t_start=self._t_start,
                t_end=t_end,
                label=self._label,
            )
        # reset utterance state regardless (too-short blips are discarded)
        self._buf = bytearray()
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._in_speech = False
        self._label = None
        return seg
