"""
Calibrate VoxBox's energy VAD to YOUR microphone. The default thresholds assume a
quiet mic; laptop mics with auto-gain are often louder, which makes the agent think
you're always talking (constant barge-in) or never talking. This measures your
mic's quiet vs speaking levels and prints the env vars to use.

    python scripts/calibrate.py

Needs: pip install sounddevice
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxbox.audio.live import MicSource  # noqa: E402
from voxbox.vad.base import rms_dbfs  # noqa: E402


def _measure(seconds: float) -> list:
    vals = []
    needed = int(seconds / 0.02)  # 20ms chunks
    for chunk in MicSource(chunk_ms=20.0):
        vals.append(rms_dbfs(chunk.pcm))
        if len(vals) >= needed:
            break
    return vals


def _pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


def main() -> None:
    print("🔧 VoxBox mic calibration\n")
    input("1) Stay SILENT for 3 seconds. Press Enter, then be quiet...")
    quiet = _measure(3.0)
    floor = _pct(quiet, 0.9)
    print(f"   noise floor ≈ {floor:.1f} dBFS\n")

    input("2) Now TALK normally for 3 seconds. Press Enter, then speak...")
    speech = _measure(3.0)
    level = _pct(speech, 0.5)
    print(f"   speaking level ≈ {level:.1f} dBFS\n")

    if level <= floor + 3:
        print("⚠️  Could not separate speech from noise. Try a quieter room or "
              "an external mic/headset, then re-run.")
        return

    vad_thr = round((floor + level) / 2, 1)        # midpoint
    barge_thr = round(level - 3, 1)                # need clear speech to interrupt
    print("✅ Suggested settings — run the agent with:\n")
    print(f"   VOX_VAD_THRESHOLD={vad_thr} \\")
    print(f"   VOX_BARGE_THRESHOLD={barge_thr} \\")
    print("   VOX_STT=faster_whisper VOX_LLM=ollama VOX_TTS=piper \\")
    print("   VOX_OLLAMA_MODEL=gemma3 VOX_PIPER_VOICE=en_US-amy-medium.onnx \\")
    print("   python scripts/talk.py\n")
    print("Tip: if it still self-interrupts, raise VOX_VAD_THRESHOLD a few dB "
          "(toward 0) or use headphones.")


if __name__ == "__main__":
    main()
