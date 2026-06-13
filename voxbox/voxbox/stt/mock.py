"""
MockSTT — deterministic speech-to-text for dev & tests.

If the segment carries a ground-truth `label` (stamped by a mock audio source) it
returns that text verbatim, which makes the whole pipeline assertable end-to-end.
Otherwise it falls back to a description of the utterance length.
"""
from __future__ import annotations

from ..contracts import SpeechSegment, Transcript


class MockSTT:
    def __init__(self, fixed_confidence: float = 0.99) -> None:
        self.fixed_confidence = fixed_confidence

    def transcribe(self, segment: SpeechSegment) -> Transcript:
        if segment.label is not None:
            return Transcript(text=segment.label, confidence=self.fixed_confidence)
        return Transcript(
            text=f"<utterance {segment.duration_ms:.0f}ms>",
            confidence=self.fixed_confidence,
        )
