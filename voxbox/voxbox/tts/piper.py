"""
Real TTS adapter backed by Piper (a fast, fully-local neural TTS).

Runs on the GPU/edge box. Install with:  pip install voxbox[piper]
Swap in Kokoro/XTTS here behind the same `synthesize` contract if preferred.
"""
from __future__ import annotations

from ..contracts import AudioReply


class PiperTTS:
    def __init__(self, voice_path: str, sample_rate: int = 22050) -> None:
        try:
            from piper import PiperVoice  # type: ignore
        except Exception as exc:  # pragma: no cover - edge box only
            raise RuntimeError("PiperTTS requires piper-tts. Install voxbox[piper].") from exc
        self._voice = PiperVoice.load(voice_path)
        self.sample_rate = sample_rate

    def synthesize(self, text: str) -> AudioReply:  # pragma: no cover - edge box only
        raise NotImplementedError(
            "Call self._voice.synthesize(text) -> int16 PCM bytes; wrap in AudioReply."
        )
