# VoxBox — a fully-local, private voice agent

> Inspired by the "I built a voice agent and open-sourced it — nothing goes out
> of your computer" architecture. VoxBox is a real-time voice assistant whose
> entire pipeline (VAD → STT → LLM → TTS) runs on hardware you own.

```
   ┌──────────────── MAC (orchestrator/client) ─────────────────┐
   │  mic capture → [chunks] ───────────────┐     ┌── playback   │
   │                                        │     │  ▲           │
   │           timing & metrics  ◀──────────┼─────┼──┘           │
   └────────────────────────────────────────┼─────┼─────────────┘
                                             ▼     │ AudioReply
   ┌──────────── GPU BOX (e.g. 3x Blackwell) ┼─────┼─────────────┐
   │   VAD (Silero) → STT (faster-whisper) → LLM (Gemma) → TTS   │
   └─────────────────────────────────────────────────────────────┘
                       100% local · private · open-source
```

## Why this design
Every stage is coded against a **Protocol** (`voxbox/contracts.py`), so each can be
swapped between a **mock** (zero deps, runs anywhere, used in CI) and a **real model**
(on the GPU box) with a one-line config change. That's how the whole pipeline is
testable without a GPU — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and the
[ADRs](docs/).

## Quickstart (no GPU, no audio hardware)
```bash
cd voxbox
python3 scripts/demo.py        # runs a scripted local conversation + metrics
python3 -m pytest -q           # 29 tests, all green on stdlib alone
```

## Using it in code
```python
from voxbox import Config, build_pipeline

orch = build_pipeline(Config())          # all-mock pipeline
for chunk in mic_chunks:                 # 16-bit/16k mono AudioChunks
    turn = orch.feed(chunk)
    if turn:                             # an utterance completed
        print(turn.transcript.text, "→", turn.response_text)
        play(turn.reply.pcm)             # 16-bit/24k mono
```

## Test a real LLM locally (no GPU, no mic)
Ollama runs on CPU too, so you can drive a genuine LLM turn through the real loop:
```bash
# 1. install Ollama (https://ollama.com), then:
ollama pull gemma3
# 2. chat through the actual VoxBox turn loop (STT/TTS mocked, LLM is real):
python3 scripts/chat_ollama.py            # or: python3 scripts/chat_ollama.py llama3.2
```
The adapter is unit-tested without a daemon (mocked HTTP) in `tests/test_llm_ollama.py`.

## Going real (the GPU box)
Flip the config and install extras:
```python
Config(vad="silero", stt="faster_whisper", llm="ollama", tts="piper")
```
```bash
pip install -e ".[silero,whisper,piper,server]"
```
Then serve the inference pipeline over the LAN (`voxbox/transport/server.py`) and
point the Mac client at it. The real adapters are stubbed with exact wiring notes.

## Layout
| Path | Role (diagram box) |
|------|--------------------|
| `voxbox/orchestrator.py` | the **Orchestrator** turn loop |
| `voxbox/metrics.py` | **Timing & metrics** |
| `voxbox/vad/` | **VAD** (energy mock + Silero) |
| `voxbox/stt/` | **STT** (mock + faster-whisper) |
| `voxbox/llm/` | **LLM** (rule-based mock + Ollama/Gemma) |
| `voxbox/tts/` | **TTS** (mock + Piper) |
| `voxbox/audio/` | mic/playback + WAV/list sources |
| `voxbox/transport/` | LAN WebSocket service |
| `tests/` | architecture, contract, integration & latency tests |

See [`docs/TESTING.md`](docs/TESTING.md) for the QA strategy and
[`docs/ROADMAP.md`](docs/ROADMAP.md) for what ships next.
