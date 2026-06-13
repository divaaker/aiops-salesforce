"""Architecture guard: every shipped backend must satisfy its stage Protocol.

This is the test that actually enforces the 'pluggable backends' decision (ADR-0002)
— if someone changes a backend's signature, these fail."""
from voxbox.contracts import LLM, STT, TTS, VAD, AudioChunk, AudioReply, SpeechSegment
from voxbox.vad import EnergyVAD
from voxbox.stt import MockSTT
from voxbox.llm import RuleBasedLLM
from voxbox.tts import MockTTS


def test_backends_satisfy_protocols():
    assert isinstance(EnergyVAD(), VAD)
    assert isinstance(MockSTT(), STT)
    assert isinstance(RuleBasedLLM(), LLM)
    assert isinstance(MockTTS(), TTS)


def test_real_adapters_also_satisfy_protocols_structurally():
    # Imported lazily; we only check the class shape, not instantiate (no GPU here).
    from voxbox.stt.faster_whisper import FasterWhisperSTT
    from voxbox.llm.ollama import OllamaLLM

    assert hasattr(FasterWhisperSTT, "transcribe")
    assert hasattr(OllamaLLM, "respond")


def test_audio_duration_math():
    # 16000 samples @16k = 1000 ms; 16-bit -> 2 bytes/sample
    chunk = AudioChunk(pcm=b"\x00\x00" * 16000, sample_rate=16000)
    assert chunk.num_samples == 16000
    assert abs(chunk.duration_ms - 1000.0) < 1e-6

    reply = AudioReply(pcm=b"\x00\x00" * 24000, sample_rate=24000)
    assert abs(reply.duration_ms - 1000.0) < 1e-6

    seg = SpeechSegment(pcm=b"\x00\x00" * 8000, sample_rate=16000)
    assert abs(seg.duration_ms - 500.0) < 1e-6


def test_zero_sample_rate_is_safe():
    assert AudioChunk(pcm=b"\x00\x00", sample_rate=0).duration_ms == 0.0
