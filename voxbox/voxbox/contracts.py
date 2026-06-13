"""
voxbox.contracts
================
The single source of truth for the data that flows through the pipeline and the
interfaces (Protocols) every backend must satisfy.

Why Protocols?  The architecture (see docs/ARCHITECTURE.md) requires that every
stage — VAD, STT, LLM, TTS — be swappable: a mock for tests/dev, a real model on
the GPU box. Coding against these Protocols (not concrete classes) is what makes
that swap a one-line config change.

All audio is 16-bit little-endian, mono, signed PCM. Sample rate travels with the
data so a stage never has to guess.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable

# ── Data carried between stages ──────────────────────────────────────────────


@dataclass
class AudioChunk:
    """A small slice of captured microphone audio (typically 10–30 ms)."""

    pcm: bytes
    sample_rate: int = 16000
    ts: float = field(default_factory=time.monotonic)
    # TEST-ONLY: mock audio sources stamp the ground-truth transcript here so the
    # whole pipeline can be exercised deterministically without real audio/ML.
    label: Optional[str] = None

    @property
    def num_samples(self) -> int:
        return len(self.pcm) // 2

    @property
    def duration_ms(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        return 1000.0 * self.num_samples / self.sample_rate


@dataclass
class SpeechSegment:
    """A complete utterance assembled by the VAD (speech bounded by silence)."""

    pcm: bytes
    sample_rate: int = 16000
    t_start: float = 0.0
    t_end: float = 0.0
    label: Optional[str] = None  # TEST-ONLY, see AudioChunk.label

    @property
    def num_samples(self) -> int:
        return len(self.pcm) // 2

    @property
    def duration_ms(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        return 1000.0 * self.num_samples / self.sample_rate


@dataclass
class Transcript:
    text: str
    confidence: float = 1.0
    lang: str = "en"


@dataclass
class LLMResponse:
    text: str


@dataclass
class AudioReply:
    """Synthesized speech ready for playback."""

    pcm: bytes
    sample_rate: int = 24000

    @property
    def num_samples(self) -> int:
        return len(self.pcm) // 2

    @property
    def duration_ms(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        return 1000.0 * self.num_samples / self.sample_rate


# ── Stage interfaces ─────────────────────────────────────────────────────────


@runtime_checkable
class VAD(Protocol):
    """Voice Activity Detection / turn segmentation.

    Fed one chunk at a time; returns a SpeechSegment the moment it decides an
    utterance has ended, otherwise None.
    """

    def process(self, chunk: AudioChunk) -> Optional[SpeechSegment]: ...

    def reset(self) -> None: ...


@runtime_checkable
class STT(Protocol):
    """Speech-to-text."""

    def transcribe(self, segment: SpeechSegment) -> Transcript: ...


@runtime_checkable
class LLM(Protocol):
    """Conversational language model."""

    def respond(self, text: str, history: List[dict]) -> LLMResponse: ...


@runtime_checkable
class TTS(Protocol):
    """Text-to-speech."""

    def synthesize(self, text: str) -> AudioReply: ...
