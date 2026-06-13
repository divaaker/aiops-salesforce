"""
Real VAD adapter backed by Silero (the model shown in the diagram).

Silero is a small neural voice-activity model that's far more robust than energy
thresholding — it tells speech from noise, so you don't have to calibrate a dBFS
threshold per microphone. It runs fine on CPU.

It operates on fixed 512-sample windows at 16 kHz (~32 ms). This adapter buffers
the incoming chunk stream into those windows, asks the model for a speech
probability per window, and applies the same hangover / min-speech segmentation as
the energy VAD. torch + silero-vad are imported lazily; install with
`pip install "voxbox[silero]"`.

Testability: pass a `prob_fn(samples)->float` to drive segmentation deterministically
without torch (used by the tests); the default uses the real model.
"""
from __future__ import annotations

import array
from typing import Callable, Optional

from ..contracts import AudioChunk, SpeechSegment


class SileroVAD:
    WINDOW = 512  # samples per inference at 16 kHz

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: float = 200.0,
        hangover_ms: float = 400.0,
        sample_rate: int = 16000,
        prob_fn: Optional[Callable] = None,
        model=None,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.hangover_ms = hangover_ms
        self.sample_rate = sample_rate
        self._model = None
        self._torch = None

        if prob_fn is not None:
            self._prob = prob_fn
        elif model is not None:
            self._model = model
            self._prob = self._silero_prob
        else:
            try:
                import torch
                from silero_vad import load_silero_vad  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    'SileroVAD requires torch + silero-vad. Install voxbox[silero].'
                ) from exc
            self._torch = torch
            self._model = load_silero_vad()
            self._prob = self._silero_prob
        self.reset()

    def reset(self) -> None:
        self._buf = bytearray()
        self._utt = bytearray()
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._in_speech = False
        self._t_start = 0.0
        self._last_ts = 0.0
        self._label: Optional[str] = None
        self._pending_label: Optional[str] = None
        if self._model is not None and hasattr(self._model, "reset_states"):
            try:  # pragma: no cover - real model only
                self._model.reset_states()
            except Exception:
                pass

    def _silero_prob(self, samples) -> float:  # pragma: no cover - real model only
        t = self._torch.tensor([s / 32768.0 for s in samples], dtype=self._torch.float32)
        with self._torch.no_grad():
            return float(self._model(t, self.sample_rate).item())

    @property
    def _window_ms(self) -> float:
        return 1000.0 * self.WINDOW / self.sample_rate

    def process(self, chunk: AudioChunk) -> Optional[SpeechSegment]:
        self._last_ts = chunk.ts
        if chunk.label is not None:
            self._pending_label = chunk.label
        self._buf.extend(chunk.pcm)

        emitted: Optional[SpeechSegment] = None
        win_bytes = self.WINDOW * 2
        while len(self._buf) >= win_bytes:
            window = bytes(self._buf[:win_bytes])
            del self._buf[:win_bytes]
            seg = self._process_window(window)
            if seg is not None:
                emitted = seg
        return emitted

    def _process_window(self, window: bytes) -> Optional[SpeechSegment]:
        samples = array.array("h")
        samples.frombytes(window)
        is_speech = self._prob(samples) >= self.threshold

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._utt = bytearray()
                self._speech_ms = 0.0
                self._silence_ms = 0.0
                self._t_start = self._last_ts
                self._label = self._pending_label
            self._utt.extend(window)
            self._speech_ms += self._window_ms
            self._silence_ms = 0.0
            return None

        if self._in_speech:
            self._utt.extend(window)
            self._silence_ms += self._window_ms
            if self._silence_ms >= self.hangover_ms:
                return self._emit()
        return None

    def _emit(self) -> Optional[SpeechSegment]:
        seg = None
        if self._speech_ms >= self.min_speech_ms:
            seg = SpeechSegment(
                pcm=bytes(self._utt),
                sample_rate=self.sample_rate,
                t_start=self._t_start,
                t_end=self._last_ts,
                label=self._label,
            )
        self._in_speech = False
        self._utt = bytearray()
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._label = None
        return seg
