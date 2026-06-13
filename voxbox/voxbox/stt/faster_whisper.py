"""
Real STT adapter backed by faster-whisper (the model shown in the diagram).

Runs on the GPU box. Install with:  pip install voxbox[whisper]
"""
from __future__ import annotations

from ..contracts import SpeechSegment, Transcript


class FasterWhisperSTT:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda",
                 compute_type: str = "float16") -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover - GPU only
            raise RuntimeError(
                "FasterWhisperSTT requires faster-whisper. Install voxbox[whisper]."
            ) from exc
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, segment: SpeechSegment) -> Transcript:  # pragma: no cover
        raise NotImplementedError(
            "Convert segment.pcm (int16) -> float32 numpy at 16kHz, call "
            "self._model.transcribe(audio); join segment texts into Transcript."
        )
