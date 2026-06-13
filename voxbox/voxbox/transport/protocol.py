"""
Wire protocol shared by the LAN client and server.

Per completed turn the server sends two frames:
  1. a JSON text frame:  {transcript, response, sample_rate, metrics}
  2. a binary frame:     the reply PCM (int16) at `sample_rate`
The client reads the JSON to learn the sample rate, then plays the next binary
frame. Keeping this in one place means both sides can't drift, and it's unit-tested.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid import cycle at runtime
    from ..orchestrator import TurnResult


def turn_to_meta(result: "TurnResult") -> dict:
    return {
        "transcript": result.transcript.text,
        "response": result.response_text,
        "sample_rate": result.reply.sample_rate,
        "metrics": result.metrics.as_dict(),
    }
