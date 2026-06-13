"""
Local text-chat against a REAL local LLM (Ollama/Gemma), driven through the actual
VoxBox turn loop. STT and TTS are mocked, so you need no microphone or GPU — just
a running Ollama daemon. This is the on-architecture way to test a genuine LLM turn.

Setup:
    1. Install Ollama from https://ollama.com
    2. ollama pull gemma3          # or any model you have
    3. python3 scripts/chat_ollama.py            # uses gemma3 @ localhost:11434
       python3 scripts/chat_ollama.py llama3.2   # pick a different model

Type your message and press Enter. Ctrl-C / "quit" to exit.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxbox import Config, build_pipeline  # noqa: E402
from voxbox.contracts import SpeechSegment  # noqa: E402
from voxbox.llm.ollama import OllamaLLM  # noqa: E402


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma3"
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    # Preflight: fail friendly if the daemon isn't up.
    if not OllamaLLM(model=model, host=host).is_available():
        print(f"❌ Ollama not reachable at {host}.")
        print(f"   Start it with `ollama serve` and `ollama pull {model}`.")
        sys.exit(1)

    orch = build_pipeline(
        Config(
            vad="energy",
            stt="mock",          # echoes the text we feed in as the transcript
            llm="ollama",        # the REAL local model
            tts="mock",          # produces placeholder audio (length ~ words)
            budget_ms=3000.0,
            options={"llm": {"model": model, "host": host}},
        )
    )

    print(f"💬 VoxBox local chat — model={model} @ {host}")
    print("   (STT/TTS mocked; LLM is real & fully local). Type 'quit' to exit.\n")

    while True:
        try:
            text = input("you  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 bye")
            break
        if not text or text.lower() in {"quit", "exit"}:
            print("👋 bye")
            break

        # Feed the typed text into the real turn loop via a labeled segment.
        seg = SpeechSegment(pcm=b"\x00\x00" * 1600, sample_rate=16000, label=text)
        turn = orch.handle_segment(seg)
        m = turn.metrics
        print(f"voox > {turn.response_text}")
        print(
            f"        ⏱ llm={m.llm_ms:.0f}ms total={m.total_ms:.0f}ms "
            f"reply_audio={turn.reply.duration_ms:.0f}ms"
            + ("  ⚠ OVER BUDGET" if m.over_budget else "")
            + "\n"
        )


if __name__ == "__main__":
    main()
