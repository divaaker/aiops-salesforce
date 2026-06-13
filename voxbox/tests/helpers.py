"""Shared fixtures for building fake audio that drives the real VAD/orchestrator."""
from voxbox.audio import chunks_from_pcm

SR = 16000
_LOUD_SAMPLE = b"\x00\x40"   # ~16k amplitude -> well above -40 dBFS
_QUIET_SAMPLE = b"\x00\x00"  # silence


def loud(ms: float) -> bytes:
    return _LOUD_SAMPLE * int(SR * ms / 1000.0)


def quiet(ms: float) -> bytes:
    return _QUIET_SAMPLE * int(SR * ms / 1000.0)


def utterance_chunks(text, speech_ms=400.0, trailing_silence_ms=400.0, start_ts=0.0):
    """One full spoken turn: loud speech then enough silence to close the VAD."""
    pcm = loud(speech_ms) + quiet(trailing_silence_ms)
    return chunks_from_pcm(pcm, sample_rate=SR, chunk_ms=20.0, label=text, start_ts=start_ts)
