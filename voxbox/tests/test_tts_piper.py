"""Tests for the Piper adapter WITHOUT installing piper-tts: we inject fake voices
covering all three API shapes Piper has shipped (raw stream, WAV-writer, chunk
iterator). A live smoke test on a real voice lives in scripts/speak.py."""
import wave

import pytest

from voxbox.contracts import SpeechSegment
from voxbox.tts.piper import PiperTTS
from voxbox.orchestrator import Orchestrator
from voxbox.vad import EnergyVAD
from voxbox.stt import MockSTT
from voxbox.llm import RuleBasedLLM


class _Cfg:
    def __init__(self, sr):
        self.sample_rate = sr


class _StreamVoice:
    """API #1: synthesize_stream_raw(text) -> iterable of raw int16 bytes."""

    config = _Cfg(22050)

    def synthesize_stream_raw(self, text):
        yield b"\x01\x00" * 100
        yield b"\x02\x00" * 100


class _WavVoice:
    """API #2: synthesize(text, wave_write) writes a WAV."""

    config = _Cfg(16000)

    def synthesize(self, text, wf):
        wf.writeframes(b"\x03\x00" * 240)


class _Chunk:
    def __init__(self, b):
        self.audio_int16_bytes = b


class _ChunkVoice:
    """API #3: synthesize(text) -> iterable of objects with int16 bytes.
    Note the single-arg signature makes the WAV-writer probe raise TypeError."""

    config = _Cfg(24000)

    def synthesize(self, text):
        return iter([_Chunk(b"\x04\x00" * 50), _Chunk(b"\x05\x00" * 50)])


def test_stream_raw_api():
    tts = PiperTTS(voice=_StreamVoice())
    reply = tts.synthesize("hi")
    assert reply.sample_rate == 22050
    assert len(reply.pcm) == 400  # (100+100) samples * 2 bytes


def test_wav_writer_api():
    tts = PiperTTS(voice=_WavVoice())
    reply = tts.synthesize("hi")
    assert reply.sample_rate == 16000
    assert len(reply.pcm) == 480  # 240 samples * 2 bytes


def test_chunk_iterator_api():
    tts = PiperTTS(voice=_ChunkVoice())
    reply = tts.synthesize("hi")
    assert reply.sample_rate == 24000
    assert len(reply.pcm) == 200  # (50+50) samples * 2 bytes


def test_sample_rate_override_wins():
    assert PiperTTS(voice=_StreamVoice(), sample_rate=48000).sample_rate == 48000


def test_missing_voice_path_raises():
    with pytest.raises(RuntimeError) as ei:
        PiperTTS()  # no voice, no path
    assert "voice_path" in str(ei.value)


def test_missing_library_raises_friendly(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "piper":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError) as ei:
        PiperTTS(voice_path="some.onnx")
    assert "piper-tts" in str(ei.value)


def test_piper_runs_in_the_real_turn_loop():
    orch = Orchestrator(
        vad=EnergyVAD(),
        stt=MockSTT(),
        llm=RuleBasedLLM(),
        tts=PiperTTS(voice=_StreamVoice()),
        budget_ms=5000.0,
    )
    turn = orch.handle_segment(
        SpeechSegment(pcm=b"\x00\x00" * 1600, sample_rate=16000, label="hello")
    )
    assert turn.reply.sample_rate == 22050
    assert turn.reply.num_samples > 0


def test_reply_is_writable_wav(tmp_path):
    from voxbox.audio import write_wav

    reply = PiperTTS(voice=_StreamVoice()).synthesize("hello world")
    out = tmp_path / "spoken.wav"
    write_wav(str(out), reply)
    with wave.open(str(out), "rb") as wf:
        assert wf.getframerate() == 22050
        assert wf.getnchannels() == 1
        assert wf.getnframes() == 200
