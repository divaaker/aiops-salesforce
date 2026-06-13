"""
voxbox.audio.io
===============
Audio sources/sinks for the orchestrator. The WAV + list helpers are pure stdlib
and power the demo and tests; live mic/playback (sounddevice) is imported lazily
and only needed on the real client (the Mac in the diagram).
"""
from __future__ import annotations

import wave
from typing import Iterator, List, Optional

from ..contracts import AudioChunk, AudioReply


def chunks_from_pcm(
    pcm: bytes,
    sample_rate: int = 16000,
    chunk_ms: float = 20.0,
    label: Optional[str] = None,
    start_ts: float = 0.0,
) -> List[AudioChunk]:
    """Slice raw 16-bit mono PCM into fixed-size chunks with monotonically
    increasing timestamps (so VAD hangover math works in tests)."""
    samples_per_chunk = max(1, int(sample_rate * chunk_ms / 1000.0))
    bytes_per_chunk = samples_per_chunk * 2
    out: List[AudioChunk] = []
    ts = start_ts
    for i in range(0, len(pcm), bytes_per_chunk):
        out.append(
            AudioChunk(
                pcm=pcm[i : i + bytes_per_chunk],
                sample_rate=sample_rate,
                ts=ts,
                label=label,
            )
        )
        ts += chunk_ms / 1000.0
    return out


def wav_source(path: str, chunk_ms: float = 20.0) -> Iterator[AudioChunk]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    for chunk in chunks_from_pcm(pcm, sample_rate=sr, chunk_ms=chunk_ms):
        yield chunk


def write_wav(path: str, reply: AudioReply) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(reply.sample_rate)
        wf.writeframes(reply.pcm)
