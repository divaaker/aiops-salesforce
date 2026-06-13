"""
Real STT adapter backed by faster-whisper (the model shown in the diagram).

Works on CPU too (great for local testing) — pick a small model and int8:
    Config(stt="faster_whisper",
           options={"stt": {"model_size": "base", "device": "cpu",
                            "compute_type": "int8"}})
Install with:  pip install "voxbox[whisper]"   (or: pip install faster-whisper)

The adapter feeds an in-memory WAV to the model, so it needs no numpy of its own
and faster-whisper handles any resampling to 16 kHz.
"""
from __future__ import annotations

import io
import math
import wave
from typing import List, Optional

from ..contracts import SpeechSegment, Transcript


class FasterWhisperSTT:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        language: Optional[str] = None,
        beam_size: int = 5,
        model=None,  # inject a preloaded/fake model (used by tests)
    ) -> None:
        self.language = language
        self.beam_size = beam_size
        if model is not None:
            self._model = model
            return
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                'FasterWhisperSTT requires faster-whisper. Install voxbox[whisper].'
            ) from exc
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def _to_wav_buffer(self, segment: SpeechSegment) -> io.BytesIO:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(segment.sample_rate or 16000)
            wf.writeframes(segment.pcm)
        buf.seek(0)
        return buf

    def transcribe(self, segment: SpeechSegment) -> Transcript:
        buf = self._to_wav_buffer(segment)
        segments, info = self._model.transcribe(
            buf, beam_size=self.beam_size, language=self.language
        )
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()

        # Rough confidence from avg_logprob if the backend provides it.
        logprobs: List[float] = [
            s.avg_logprob for s in segs if getattr(s, "avg_logprob", None) is not None
        ]
        confidence = math.exp(sum(logprobs) / len(logprobs)) if logprobs else 1.0
        confidence = max(0.0, min(1.0, confidence))

        lang = getattr(info, "language", None) or self.language or "en"
        return Transcript(text=text, confidence=confidence, lang=lang)
