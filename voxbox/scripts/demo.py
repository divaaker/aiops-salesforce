"""
End-to-end demo of the VoxBox pipeline using all-mock backends (no GPU, no audio
hardware). It synthesizes a scripted conversation, runs it through the real turn
loop, and prints the metrics table.

    python voxbox/scripts/demo.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxbox import Config, build_pipeline  # noqa: E402
from voxbox.audio import chunks_from_pcm  # noqa: E402

# A pretend utterance: 400ms of "speech" (loud) then 400ms of silence to close the turn.
SR = 16000
LOUD = (b"\x00\x40") * int(SR * 0.4)   # ~ -6 dBFS-ish constant tone-ish energy
QUIET = (b"\x00\x00") * int(SR * 0.4)  # silence -> triggers VAD hangover


def utterance(text: str, start_ts: float):
    pcm = LOUD + QUIET
    return chunks_from_pcm(pcm, sample_rate=SR, chunk_ms=20.0, label=text, start_ts=start_ts)


def main() -> None:
    orch = build_pipeline(Config(vad="energy", stt="mock", llm="mock", tts="mock"))

    script = ["hello there", "what is your name", "tell me something about latency?"]
    chunks = []
    ts = 0.0
    for line in script:
        u = utterance(line, ts)
        chunks.extend(u)
        ts = u[-1].ts + 0.5

    results = orch.run(chunks)

    print("\n=== VoxBox local conversation (mock backends) ===\n")
    for r in results:
        print(f"  🗣️  user : {r.transcript.text}")
        print(f"  🤖 agent: {r.response_text}")
        m = r.metrics
        print(
            f"      ⏱  vad={m.vad_ms:.1f} stt={m.stt_ms:.1f} "
            f"llm={m.llm_ms:.1f} tts={m.tts_ms:.1f} total={m.total_ms:.1f}ms "
            f"reply_audio={r.reply.duration_ms:.0f}ms"
            + ("  ⚠ OVER BUDGET" if m.over_budget else "")
        )
        print()

    print("=== Metrics summary ===")
    for k, v in orch.metrics.summary().items():
        print(f"  {k}: {v}")
    print()


if __name__ == "__main__":
    main()
