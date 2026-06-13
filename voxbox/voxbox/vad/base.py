"""Shared helpers for VAD backends."""
from __future__ import annotations

import array
import math


def rms_dbfs(pcm: bytes) -> float:
    """Root-mean-square loudness of 16-bit mono PCM, in dBFS (0 = full scale)."""
    if not pcm:
        return -120.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return -120.0
    acc = 0.0
    for s in samples:
        acc += float(s) * float(s)
    rms = math.sqrt(acc / len(samples))
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(rms / 32768.0)
