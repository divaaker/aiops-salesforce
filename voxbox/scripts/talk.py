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
from voxbox.runtime import config_from_env, live_env_defaults, run_conversation  # noqa: E402


def main() -> None:
    cfg = config_from_env(live_env_defaults())
    barge_in = os.environ.get("VOX_BARGE_IN", "1") != "0"
    streaming = os.environ.get("VOX_STREAM", "1") != "0"
    # Barge-in sensitivity (tune to your mic; see scripts/calibrate.py)
    barge_threshold = float(os.environ.get("VOX_BARGE_THRESHOLD", "-30"))
    onset_frames = int(os.environ.get("VOX_ONSET_FRAMES", "6"))
    playback_margin = float(os.environ.get("VOX_PLAYBACK_MARGIN", "8"))
    try:
        orch = build_pipeline(cfg)
        from voxbox.audio.live import MicSource, QueueingSpeaker, StreamingSpeaker
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except ImportError:
        print("❌ Live audio needs sounddevice. Install with: pip install 'voxbox[client]'")
        sys.exit(1)

    # Optional tool-calling: wrap the LLM so it can call tools (mock or MCP).
    tools_mode = os.environ.get("VOX_TOOLS", "none")
    if tools_mode != "none":
        if cfg.llm != "ollama":
            print("❌ VOX_TOOLS needs VOX_LLM=ollama (a tool-capable model).")
            sys.exit(1)
        from voxbox.llm.ollama import OllamaToolModel
        from voxbox.tools import MockToolProvider, ToolCallingLLM

        model = OllamaToolModel(
            model=os.environ.get("VOX_TOOLS_MODEL", "llama3.1"),
            host=os.environ.get("VOX_OLLAMA_HOST", "http://localhost:11434"),
        )
        if tools_mode == "mock":
            provider = MockToolProvider()
        elif tools_mode == "mcp":
            print("❌ VOX_TOOLS=mcp needs a connected MCP session (OAuth). "
                  "See docs/MCP.md; use VOX_TOOLS=mock to test the flow now.")
            sys.exit(1)
        else:
            print(f"❌ unknown VOX_TOOLS={tools_mode!r} (use 'mock' or 'mcp').")
            sys.exit(1)
        orch.llm = ToolCallingLLM(model, provider)
        print(f"🔧 tools enabled ({tools_mode}): {[t.name for t in provider.list_tools()]}")

    vad_thr = cfg.options.get("vad", {}).get("threshold_dbfs", -40.0)
    hangover = cfg.options.get("vad", {}).get("hangover_ms", 300.0)
    whisper = cfg.options.get("stt", {}).get("model_size", "base")
    mode = "streaming" if streaming else "whole-reply"
    stt_label = f"{cfg.stt}:{whisper}" if cfg.stt == "faster_whisper" else cfg.stt
    print(f"🎙️  VoxBox live (stt={stt_label} llm={cfg.llm} tts={cfg.tts}, "
          f"barge-in={'on' if barge_in else 'off'}, {mode}).\n"
          f"    vad_threshold={vad_thr}dBFS  endpoint_pause={hangover}ms. "
          f"Speak, then pause. Ctrl-C to quit.\n")

    interrupt = lambda: print("  ✋ (interrupted — listening)\n")  # noqa: E731

    try:
        if streaming:
            # ConversationEngine generates off-thread -> true mid-response barge-in.
            from voxbox.engine import ConversationEngine

            engine = ConversationEngine(
                orch, QueueingSpeaker(), barge_in=barge_in,
                threshold_dbfs=barge_threshold, onset_frames=onset_frames,
                playback_margin_dbfs=playback_margin,
                on_chunk=lambda c: print(f"  🔊 {c.text}"),
                on_turn=lambda d: print(f"  🗣️ heard: {d.transcript.text!r}    "
                                        f"⏱ first_audio={d.metrics.first_audio_ms:.0f}ms "
                                        f"total={d.metrics.total_ms:.0f}ms\n"),
                on_interrupt=interrupt,
                on_error=lambda e: print(f"  ⚠ backend error: {e}\n"),
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
