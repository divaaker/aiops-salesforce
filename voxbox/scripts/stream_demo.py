"""
Demo of STREAMING responses with mock backends (no GPU/mic). Shows the agent
emitting audio sentence-by-sentence and reports first_audio_ms — the latency the
user actually feels — vs total_ms.

    python3 scripts/stream_demo.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxbox import Config, build_pipeline  # noqa: E402
from voxbox.contracts import SpeechSegment  # noqa: E402
from voxbox.llm.mock import RuleBasedLLM  # noqa: E402
from voxbox.streaming import StreamChunk, StreamDone  # noqa: E402


class _MultiSentenceLLM(RuleBasedLLM):
    """A mock that returns several sentences so streaming is visible."""

    def respond(self, text, history):
        from voxbox.contracts import LLMResponse
        return LLMResponse(
            f"Sure. You said {text.strip()!r}. Here is a longer reply. "
            f"It has several sentences. The agent speaks each as it is ready!"
        )


def main() -> None:
    orch = build_pipeline(Config())
    orch.llm = _MultiSentenceLLM()  # swap in a chatty mock

    seg = SpeechSegment(pcm=b"\x00\x00" * 1600, sample_rate=16000, label="tell me a story")
    print("\n=== Streaming reply (mock backends) ===\n")
    spoken_ms = 0.0
    for ev in orch.stream_segment(seg):
        if isinstance(ev, StreamChunk):
            spoken_ms += ev.reply.duration_ms
            print(f"  🔊 speak: {ev.text!r}  ({ev.reply.duration_ms:.0f}ms audio)")
        elif isinstance(ev, StreamDone):
            m = ev.metrics
            print(f"\n  heard  : {ev.transcript.text!r}")
            print(f"  reply  : {ev.response_text!r}")
            print(f"\n  ⏱ first_audio={m.first_audio_ms:.1f}ms  total={m.total_ms:.1f}ms")
            print(f"     (user waits ~first_audio before hearing anything; "
                  f"then {spoken_ms:.0f}ms of speech plays)\n")


if __name__ == "__main__":
    main()
