"""
Synthesize speech locally with Piper TTS and save to a WAV you can play. CPU-ok,
no GPU. This is the real TTS stage of the VoxBox pipeline.

Setup:
    pip install piper-tts
    # download a voice (.onnx + .onnx.json), e.g.:
    #   https://huggingface.co/rhasspy/piper-voices  -> en_US-amy-medium.onnx
Usage:
    python3 scripts/speak.py en_US-amy-medium.onnx "Hello from a fully local agent."
    python3 scripts/speak.py en_US-amy-medium.onnx "Hi" out.wav

Combine with the other stages for a full local chain:
    text  -> speak.py                              (LLM-less TTS)
    WAV   -> transcribe_wav.py (STT+LLM) -> feed the reply text into speak.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxbox.audio import write_wav  # noqa: E402
from voxbox.tts.piper import PiperTTS  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        print('usage: python3 scripts/speak.py <voice.onnx> "<text>" [out.wav]')
        sys.exit(2)
    voice_path, text = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "out.wav"

    if not os.path.exists(voice_path):
        print(f"❌ voice not found: {voice_path}")
        print("   Download one from https://huggingface.co/rhasspy/piper-voices")
        sys.exit(1)

    try:
        tts = PiperTTS(voice_path=voice_path)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    reply = tts.synthesize(text)
    write_wav(out, reply)
    print(f"🔊 wrote {out}  ({reply.duration_ms:.0f}ms @ {reply.sample_rate}Hz)")
    print(f"   play it:  ffplay -autoexit {out}    # or: afplay {out} (macOS)")


if __name__ == "__main__":
    main()
