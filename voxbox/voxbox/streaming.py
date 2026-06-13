"""
voxbox.streaming
================
Pieces for streaming responses: chunk a token stream into speakable sentences as
soon as they're complete, and the event types the streaming orchestrator yields.

Streaming TTS one sentence at a time is what lets the agent start talking after the
first sentence instead of waiting for the whole LLM reply — the biggest perceived-
latency win (see docs/ARCHITECTURE.md §6).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator

from .contracts import AudioReply, Transcript
from .metrics import TurnMetrics

_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")


def sentence_chunker(tokens: Iterable[str]) -> Iterator[str]:
    """Yield complete sentences from a stream of text fragments as they finish.
    Any trailing partial text is emitted at the end."""
    buf = ""
    for tok in tokens:
        buf += tok
        while True:
            m = _SENTENCE_END.search(buf)
            if not m:
                break
            cut = m.end()
            sentence = buf[:cut].strip()
            buf = buf[cut:]
            if sentence:
                yield sentence
    tail = buf.strip()
    if tail:
        yield tail


@dataclass
class StreamChunk:
    """One ready-to-play piece of the reply (a synthesized sentence)."""

    text: str
    reply: AudioReply


@dataclass
class StreamDone:
    """Terminal event: the full transcript, response, and timing for the turn."""

    transcript: Transcript
    response_text: str
    metrics: TurnMetrics
