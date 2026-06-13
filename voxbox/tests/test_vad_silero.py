"""Tests for SileroVAD's windowing + segmentation, driven by an injected
probability function so torch/silero-vad don't need to be installed."""
import pytest

from voxbox.contracts import AudioChunk, SpeechSegment, VAD
from voxbox.vad.silero import SileroVAD
from tests.helpers import loud, quiet, utterance_chunks, SR


def energy_prob(samples):
    """Stand-in for the neural model: 'speech' when the window is loud."""
    peak = max((abs(s) for s in samples), default=0)
    return 1.0 if peak > 1000 else 0.0


def test_is_a_vad():
    assert isinstance(SileroVAD(prob_fn=energy_prob), VAD)


def test_emits_one_segment_for_speech_then_silence():
    vad = SileroVAD(threshold=0.5, min_speech_ms=100, hangover_ms=200, prob_fn=energy_prob)
    emitted = [vad.process(c) for c in utterance_chunks("hi there", start_ts=0.0)]
    segs = [s for s in emitted if s is not None]
    assert len(segs) == 1
    assert isinstance(segs[0], SpeechSegment)
    assert segs[0].label == "hi there"
    assert segs[0].duration_ms > 100


def test_silence_only_yields_nothing():
    vad = SileroVAD(prob_fn=energy_prob, hangover_ms=200)
    out = [vad.process(AudioChunk(pcm=quiet(20), sample_rate=SR, ts=i * 0.02)) for i in range(60)]
    assert all(s is None for s in out)


def test_too_short_blip_discarded():
    vad = SileroVAD(prob_fn=energy_prob, min_speech_ms=400, hangover_ms=200)
    from voxbox.audio import chunks_from_pcm
    pcm = loud(80) + quiet(400)
    chunks = chunks_from_pcm(pcm, sample_rate=SR, chunk_ms=20.0, label="x")
    assert [s for s in (vad.process(c) for c in chunks) if s is not None] == []


def test_reset_clears_state():
    vad = SileroVAD(prob_fn=energy_prob)
    for c in utterance_chunks("a"):
        vad.process(c)
    vad.reset()
    assert vad._in_speech is False
    assert len(vad._buf) == 0


def test_pipeline_builds_silero_via_injected_prob():
    from voxbox.config import Config
    from voxbox.pipeline import build_pipeline

    cfg = Config(vad="silero", options={"vad": {
        "prob_fn": energy_prob, "min_speech_ms": 100, "hangover_ms": 200}})
    orch = build_pipeline(cfg)
    results = orch.run(utterance_chunks("hello", start_ts=0.0))
    assert len(results) == 1
    assert results[0].transcript.text == "hello"


def test_missing_torch_raises_friendly(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("torch", "silero_vad"):
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError) as ei:
        SileroVAD()  # no prob_fn/model -> tries the real import
    assert "silero" in str(ei.value).lower()
