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

## Transcribe a WAV end-to-end (real STT, CPU-ok)
faster-whisper runs on CPU, so you can go audio → text → reply with no GPU:
```bash
pip install faster-whisper           # first run downloads the model
python3 scripts/transcribe_wav.py path/to/speech.wav          # mono 16-bit WAV
python3 scripts/transcribe_wav.py path/to/speech.wav small    # bigger model
OLLAMA=1 python3 scripts/transcribe_wav.py speech.wav         # real STT + real LLM
```
Convert any audio first: `ffmpeg -i in.mp3 -ac 1 -ar 16000 -sample_fmt s16 out.wav`.
The adapter is unit-tested without the library (injected fake model) in
`tests/test_stt_whisper.py`.

## Synthesize speech locally (real TTS, CPU-ok)
Piper is fast on CPU, so you can get real spoken audio out:
```bash
pip install piper-tts
# download a voice (.onnx + .onnx.json) from https://huggingface.co/rhasspy/piper-voices
python3 scripts/speak.py en_US-amy-medium.onnx "Hello from a fully local agent." out.wav
```
Adapter handles all of Piper's API shapes and is tested with injected fake voices
in `tests/test_tts_piper.py`. With Ollama + faster-whisper + Piper installed you can
run the **entire STT→LLM→TTS chain on a laptop, fully local**:
`Config(vad="energy", stt="faster_whisper", llm="ollama", tts="piper", options=...)`.

## Streaming responses (lowest felt latency)
The agent can start speaking after the first sentence instead of waiting for the
whole reply — token-stream the LLM, synthesize per sentence:
```bash
python3 scripts/stream_demo.py     # shows sentence-by-sentence audio + first_audio_ms
```
```python
for ev in orch.stream_segment(segment):
    if isinstance(ev, StreamChunk): speaker.play(ev.reply)   # play as it arrives
    else: print("felt latency:", ev.metrics.first_audio_ms, "ms")
```
Works with the mock LLM and the real Ollama adapter (`respond_stream`). Tested in
`tests/test_streaming.py`.

## Talk to it live (real mic + speaker)
Co-located (one machine, mic → agent → speaker):
```bash
pip install "voxbox[client]"
python3 scripts/talk.py                 # all-mock smoke test (proves the loop)
# fully-local real agent:
VOX_STT=faster_whisper VOX_LLM=ollama VOX_TTS=piper \
VOX_OLLAMA_MODEL=gemma3 VOX_PIPER_VOICE=en_US-amy-medium.onnx \
python3 scripts/talk.py
```
Split topology (thin Mac ↔ GPU box, exactly as in the diagram):
```bash
# on the GPU box:
pip install "voxbox[server]"
VOX_STT=faster_whisper VOX_LLM=ollama VOX_TTS=piper VOX_PIPER_VOICE=amy.onnx python3 scripts/serve.py
# on the Mac:
pip install "voxbox[client]"
python3 scripts/talk_remote.py ws://<gpu-ip>:8765/stream
```
Backends are selected via `VOX_*` env vars (`config_from_env`). The loop and the
LAN wire protocol are unit-tested without hardware in `tests/test_runtime.py` and
`tests/test_transport.py`.

**Barge-in** is on by default: talk over the agent and it stops to listen
(`StreamingSpeaker` + `BargeInController`, driven by `run_conversation`). Disable
with `VOX_BARGE_IN=0`. Tested in `tests/test_barge_in.py`. (Assumes headphones or
echo cancellation so the mic doesn't hear the agent — real AEC is on the roadmap.)

**Streaming live** (default, `VOX_STREAM=1`): replies stream sentence-by-sentence
through a `QueueingSpeaker` (plays clips back-to-back, flushes instantly on
barge-in), driven by `run_conversation_streaming`. Set `VOX_STREAM=0` for
whole-reply playback. Tested in `tests/test_queueing_speaker.py` and
`tests/test_streaming_loop.py`.

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
