"""
Real TTS adapter backed by Piper — a fast, fully-local neural TTS that runs well
on CPU. (Swap in Kokoro/XTTS behind this same `synthesize` contract if preferred.)

Install:  pip install piper-tts
Download a voice (model .onnx + .onnx.json), e.g. from
https://huggingface.co/rhasspy/piper-voices , then:
    Config(tts="piper", options={"tts": {"voice_path": "en_US-amy-medium.onnx"}})

Piper's Python API has shifted across versions, so this adapter supports the three
common shapes: synthesize_stream_raw (raw int16), the chunk-iterator synthesize,
and the WAV-writing synthesize(text, wave_write).
"""
from __future__ import annotations

import io
import wave

from ..contracts import AudioReply


class PiperTTS:
    def __init__(self, voice_path: str = "", sample_rate: int | None = None, voice=None) -> None:
        self._sample_rate_override = sample_rate
        if voice is not None:
            self._voice = voice
            return
        if not voice_path:
            raise RuntimeError("PiperTTS needs a voice_path (path to a Piper .onnx voice).")
        try:
            from piper import PiperVoice  # type: ignore
        except Exception as exc:
            raise RuntimeError("PiperTTS requires piper-tts. Install voxbox[piper].") from exc
        self._voice = PiperVoice.load(voice_path)

    @property
    def sample_rate(self) -> int:
        if self._sample_rate_override:
            return self._sample_rate_override
        cfg = getattr(self._voice, "config", None)
        return getattr(cfg, "sample_rate", None) or 22050

    def synthesize(self, text: str) -> AudioReply:
        return AudioReply(pcm=self._raw_pcm(text), sample_rate=self.sample_rate)

    def _raw_pcm(self, text: str) -> bytes:
        v = self._voice

        # 1) Preferred: raw int16 byte stream.
        if hasattr(v, "synthesize_stream_raw"):
            return b"".join(v.synthesize_stream_raw(text))

        # 2) WAV-writing API: synthesize(text, wave_write). Detected via TypeError
        #    if the installed Piper actually uses the chunk-iterator API instead.
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                v.synthesize(text, wf)
            buf.seek(0)
            with wave.open(buf, "rb") as r:
                pcm = r.readframes(r.getnframes())
            if pcm:
                return pcm
        except TypeError:
            pass

        # 3) Chunk-iterator API: synthesize(text) -> objects exposing int16 bytes.
        out = bytearray()
        for ch in v.synthesize(text):
            if isinstance(ch, (bytes, bytearray)):
                out.extend(ch)
            else:
                b = getattr(ch, "audio_int16_bytes", None)
                if b:
                    out.extend(b)
        return bytes(out)
