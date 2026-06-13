from voxbox.contracts import AudioChunk, SpeechSegment
from voxbox.vad import EnergyVAD
from voxbox.vad.base import rms_dbfs
from tests.helpers import loud, quiet, utterance_chunks, SR


def test_rms_dbfs_loud_vs_quiet():
    assert rms_dbfs(quiet(100)) <= -100.0
    assert rms_dbfs(loud(100)) > -40.0


def test_silence_only_yields_no_segment():
    vad = EnergyVAD()
    seg = None
    for c in [AudioChunk(pcm=quiet(20), sample_rate=SR, ts=i * 0.02) for i in range(50)]:
        seg = vad.process(c) or seg
    assert seg is None


def test_speech_then_silence_emits_one_segment():
    vad = EnergyVAD(hangover_ms=300.0, min_speech_ms=100.0)
    segments = [vad.process(c) for c in utterance_chunks("hello", start_ts=0.0)]
    emitted = [s for s in segments if s is not None]
    assert len(emitted) == 1
    seg = emitted[0]
    assert isinstance(seg, SpeechSegment)
    assert seg.label == "hello"
    assert seg.duration_ms > 100.0


def test_too_short_blip_is_discarded():
    vad = EnergyVAD(hangover_ms=200.0, min_speech_ms=300.0)
    # only 60ms of speech -> below the 300ms minimum
    from voxbox.audio import chunks_from_pcm

    pcm = loud(60) + quiet(300)
    chunks = chunks_from_pcm(pcm, sample_rate=SR, chunk_ms=20.0, label="x")
    emitted = [s for s in (vad.process(c) for c in chunks) if s is not None]
    assert emitted == []


def test_reset_clears_state():
    vad = EnergyVAD()
    for c in utterance_chunks("a"):
        vad.process(c)
    vad.reset()
    assert vad._in_speech is False
    assert len(vad._buf) == 0
