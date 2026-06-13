"""Tests for the faster-whisper adapter WITHOUT installing faster-whisper: we
inject a fake model. Verifies WAV-buffer construction, text joining, confidence
from avg_logprob, language passthrough, and that it slots into the real loop.
A live smoke test on an actual WAV lives in scripts/transcribe_wav.py."""
import math
import wave
from dataclasses import dataclass
from typing import Optional

from voxbox.contracts import SpeechSegment
from voxbox.stt.faster_whisper import FasterWhisperSTT
from voxbox.orchestrator import Orchestrator
from voxbox.vad import EnergyVAD
from voxbox.llm import RuleBasedLLM
from voxbox.tts import MockTTS


@dataclass
class _Seg:
    text: str
    avg_logprob: Optional[float] = None


class _Info:
    def __init__(self, language="en"):
        self.language = language


class _FakeModel:
    """Records the audio it was given and returns scripted segments."""

    def __init__(self, segs, language="en"):
        self._segs = segs
        self._language = language
        self.last_audio_frames = None

    def transcribe(self, audio, beam_size=5, language=None):
        # `audio` is a BytesIO WAV — prove it's a real, readable WAV.
        with wave.open(audio, "rb") as wf:
            self.last_audio_frames = wf.getnframes()
            self.last_sr = wf.getframerate()
            self.last_channels = wf.getnchannels()
        return iter(self._segs), _Info(self._language)


def _segment(ms=500, sr=16000, pcm_byte=b"\x10\x00"):
    return SpeechSegment(pcm=pcm_byte * int(sr * ms / 1000.0), sample_rate=sr)


def test_joins_segment_texts_and_passes_valid_wav():
    fake = _FakeModel([_Seg("hello "), _Seg(" world")])
    stt = FasterWhisperSTT(model=fake)
    out = stt.transcribe(_segment(ms=500, sr=16000))
    assert out.text == "hello world"
    # the model received a proper mono 16k WAV of the right length
    assert fake.last_channels == 1
    assert fake.last_sr == 16000
    assert fake.last_audio_frames == 8000  # 500ms @16k


def test_confidence_from_avg_logprob():
    fake = _FakeModel([_Seg("hi", avg_logprob=math.log(0.8))])
    out = FasterWhisperSTT(model=fake).transcribe(_segment())
    assert abs(out.confidence - 0.8) < 1e-6


def test_confidence_defaults_to_one_without_logprob():
    out = FasterWhisperSTT(model=_FakeModel([_Seg("hi")])).transcribe(_segment())
    assert out.confidence == 1.0


def test_language_passthrough():
    out = FasterWhisperSTT(model=_FakeModel([_Seg("hola")], language="es")).transcribe(_segment())
    assert out.lang == "es"


def test_empty_result_is_safe():
    out = FasterWhisperSTT(model=_FakeModel([])).transcribe(_segment())
    assert out.text == ""
    assert out.confidence == 1.0


def test_missing_library_raises_friendly(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "faster_whisper":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import pytest

    with pytest.raises(RuntimeError) as ei:
        FasterWhisperSTT()  # no injected model -> tries the real import
    assert "faster-whisper" in str(ei.value)


def test_whisper_runs_in_the_real_turn_loop():
    fake = _FakeModel([_Seg("what time is it")])
    orch = Orchestrator(
        vad=EnergyVAD(),
        stt=FasterWhisperSTT(model=fake),
        llm=RuleBasedLLM(),
        tts=MockTTS(),
        budget_ms=5000.0,
    )
    turn = orch.handle_segment(_segment())
    assert turn.transcript.text == "what time is it"
    assert turn.response_text  # LLM responded
    assert turn.reply.num_samples > 0
