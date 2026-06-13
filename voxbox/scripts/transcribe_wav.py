"""
Transcribe a WAV file end-to-end through the REAL VoxBox loop:
real STT (faster-whisper, CPU-ok) -> LLM -> TTS. No GPU or mic required.

Setup:
    pip install faster-whisper          # first run downloads the model
Usage:
    python3 scripts/transcribe_wav.py path/to/speech.wav
    python3 scripts/transcribe_wav.py path/to/speech.wav small   # bigger model
    OLLAMA=1 python3 scripts/transcribe_wav.py speech.wav        # use real LLM too

Input must be mono, 16-bit PCM WAV. Convert anything with:
    ffmpeg -i in.mp3 -ac 1 -ar 16000 -sample_fmt s16 out.wav
"""
from __future__ import annotations

import os
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxbox import Config, build_pipeline  # noqa: E402
from voxbox.audio import wav_source  # noqa: E402
from voxbox.contracts import SpeechSegment  # noqa: E402
from voxbox.llm.ollama import OllamaLLM  # noqa: E402


def _check_wav(path: str) -> None:
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            print("❌ WAV must be mono, 16-bit. Convert it with:")
            print(f"   ffmpeg -i in.* -ac 1 -ar 16000 -sample_fmt s16 {path}")
            sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3 scripts/transcribe_wav.py <file.wav> [model_size]")
        sys.exit(2)
    path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
    if not os.path.exists(path):
        print(f"❌ no such file: {path}")
        sys.exit(1)
    _check_wav(path)

    use_ollama = os.environ.get("OLLAMA") == "1" and OllamaLLM().is_available()
    cfg = Config(
        vad="energy",
        stt="faster_whisper",
        llm="ollama" if use_ollama else "mock",
        tts="mock",
        budget_ms=5000.0,
        options={"stt": {"model_size": model_size, "device": "auto",
                         "compute_type": "default"}},
    )

    try:
        orch = build_pipeline(cfg)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    print(f"🎧 transcribing {path} with faster-whisper '{model_size}' "
          f"(LLM={'ollama' if use_ollama else 'mock'})\n")

    # Drive the full loop (VAD segments the file into turns)...
    results = orch.run(wav_source(path, chunk_ms=20.0))

    # ...but if the clip has no trailing silence, VAD may not close the turn.
    if not results:
        print("(no VAD turn detected — transcribing the whole file as one segment)\n")
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())
        results = [orch.handle_segment(SpeechSegment(pcm=pcm, sample_rate=sr))]

    for r in results:
        print(f"  🗣️  heard : {r.transcript.text!r} "
              f"(conf={r.transcript.confidence:.2f}, lang={r.transcript.lang})")
        print(f"  🤖 agent : {r.response_text}")
        m = r.metrics
        print(f"      ⏱ stt={m.stt_ms:.0f}ms llm={m.llm_ms:.0f}ms "
              f"total={m.total_ms:.0f}ms\n")


if __name__ == "__main__":
    main()
