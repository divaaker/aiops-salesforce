"""
Real VAD adapter backed by Silero (the model shown in the diagram).

Runs on the GPU box. Heavy deps (torch, silero-vad) are imported lazily so that
importing voxbox never requires them. Install with:  pip install voxbox[silero]
"""
from __future__ import annotations

from typing import Optional

from ..contracts import AudioChunk, SpeechSegment


class SileroVAD:
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: float = 100.0,
        hangover_ms: float = 300.0,
        sample_rate: int = 16000,
    ) -> None:
        try:
            import torch  # noqa: F401
            from silero_vad import load_silero_vad  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised only on GPU box
            raise RuntimeError(
                "SileroVAD requires torch + silero-vad. Install voxbox[silero]."
            ) from exc
        self._model = load_silero_vad()
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.hangover_ms = hangover_ms
        self.sample_rate = sample_rate
        self.reset()

    def reset(self) -> None:  # pragma: no cover - GPU only
        self._buf = bytearray()
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._in_speech = False
        self._t_start = 0.0

    def process(self, chunk: AudioChunk) -> Optional[SpeechSegment]:  # pragma: no cover
        raise NotImplementedError(
            "Wire chunk PCM -> torch tensor -> self._model(prob); reuse the "
            "hangover/min-speech segmentation from EnergyVAD. Runs on the GPU node."
        )
