"""
MockTTS — deterministic text-to-speech for dev & tests.

Generates silent 16-bit PCM whose length is a deterministic function of the word
count, so tests can assert that synthesis happened and produced plausibly-sized
audio without shipping a vocoder.
"""
from __future__ import annotations

from ..contracts import AudioReply


class MockTTS:
    def __init__(self, sample_rate: int = 24000, seconds_per_word: float = 0.3,
                 min_seconds: float = 0.3) -> None:
        self.sample_rate = sample_rate
        self.seconds_per_word = seconds_per_word
        self.min_seconds = min_seconds

    def synthesize(self, text: str) -> AudioReply:
        words = max(1, len(text.split()))
        seconds = max(self.min_seconds, words * self.seconds_per_word)
        num_samples = int(self.sample_rate * seconds)
        pcm = b"\x00\x00" * num_samples  # silence, 16-bit
        return AudioReply(pcm=pcm, sample_rate=self.sample_rate)
