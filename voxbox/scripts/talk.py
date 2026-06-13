"""
Live, co-located talking assistant: microphone -> VoxBox -> speaker, all on one
machine. No network. Backends are chosen by environment variables so the same
script runs all-mock (no models) or fully-real on a laptop.

Install audio:  pip install "voxbox[client]"

All-mock smoke test (hear placeholder audio, prove the loop):
    python3 scripts/talk.py

Fully-local real agent (needs faster-whisper + Ollama + Piper installed):
    VOX_STT=faster_whisper VOX_LLM=ollama VOX_TTS=piper \
    VOX_OLLAMA_MODEL=gemma3 VOX_PIPER_VOICE=en_US-amy-medium.onnx \
    python3 scripts/talk.py

Speak, pause, and the agent replies. Ctrl-C to stop.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxbox.pipeline import build_pipeline  # noqa: E402
from voxbox.runtime import config_from_env, run_conversation  # noqa: E402


def main() -> None:
    cfg = config_from_env()
    barge_in = os.environ.get("VOX_BARGE_IN", "1") != "0"
    streaming = os.environ.get("VOX_STREAM", "1") != "0"
    try:
        orch = build_pipeline(cfg)
        from voxbox.audio.live import MicSource, QueueingSpeaker, StreamingSpeaker
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except ImportError:
        print("❌ Live audio needs sounddevice. Install with: pip install 'voxbox[client]'")
        sys.exit(1)

    mode = "streaming" if streaming else "whole-reply"
    print(f"🎙️  VoxBox live (stt={cfg.stt} llm={cfg.llm} tts={cfg.tts}, "
          f"barge-in={'on' if barge_in else 'off'}, {mode}). "
          f"Speak, then pause. Ctrl-C to quit.\n")

    interrupt = lambda: print("  ✋ (interrupted — listening)\n")  # noqa: E731

    try:
        if streaming:
            # ConversationEngine generates off-thread -> true mid-response barge-in.
            from voxbox.engine import ConversationEngine

            engine = ConversationEngine(
                orch, QueueingSpeaker(), barge_in=barge_in,
                on_chunk=lambda c: print(f"  🔊 {c.text}"),
                on_turn=lambda d: print(f"     ⏱ first_audio={d.metrics.first_audio_ms:.0f}ms "
                                        f"total={d.metrics.total_ms:.0f}ms\n"),
                on_interrupt=interrupt,
            )
            engine.run(MicSource())
        else:
            def on_turn(turn):
                m = turn.metrics
                print(f"  🗣️  {turn.transcript.text!r}\n  🤖 {turn.response_text}")
                print(f"     ⏱ total={m.total_ms:.0f}ms"
                      + ("  ⚠ OVER BUDGET" if m.over_budget else "") + "\n")

            run_conversation(orch, MicSource(), StreamingSpeaker(),
                             barge_in=barge_in, on_turn=on_turn, on_interrupt=interrupt)
    except KeyboardInterrupt:
        print("\n👋 bye —", orch.metrics.summary())


if __name__ == "__main__":
    main()
