# VoxBox architecture

## 1. Goals & constraints
- **Privacy first:** the full pipeline runs on owned hardware; no stage calls the
  public internet by default. (Ollama talks only to a local daemon.)
- **Real-time:** sub-second-ish turn latency is the primary quality metric.
- **Swappable models:** STT/LLM/TTS/VAD evolve fast — changing one is a config edit.
- **Runs everywhere for dev/CI:** the logic must be exercisable with no GPU/mic.

## 2. Physical topology (from the reference image)
```
 MAC (orchestrator/client)                 GPU BOX (3x Blackwell / Windows)
 ───────────────────────                   ────────────────────────────────
 mic capture ─┐                            VAD (Silero)
              ├─ chunks ──▶ transport ──▶  STT (faster-whisper)
 playback  ◀──┤    (LAN WebSocket)         LLM (Gemma via Ollama)
 timing/metrics                            TTS (Piper/Kokoro/XTTS)
```
The Mac is "thin" (capture/playback/orchestration). The GPU box is "fat" (models).
The transport boundary (`voxbox/transport/`) is the only place the two meet, so the
same `Orchestrator` can run **co-located** (one machine) or **split** (two machines).

## 3. Logical pipeline (data flow)
```
AudioChunk(16-bit/16k) ─▶ VAD.process ─(SpeechSegment)─▶ STT.transcribe
   ─(Transcript)─▶ LLM.respond(history) ─(LLMResponse)─▶ TTS.synthesize
   ─(AudioReply 16-bit/24k)─▶ playback
```
- **VAD** turns a stream of chunks into bounded **utterances** (speech + trailing
  silence, with a minimum-speech guard). This is what enables natural turn-taking.
- **Orchestrator** owns no models. It times each stage, enforces the **latency
  budget**, and threads **conversation history** into the LLM.
- All inter-stage types live in `contracts.py` and carry their own `sample_rate`.

## 4. Key modules
| Module | Responsibility |
|--------|----------------|
| `contracts.py` | data types + stage `Protocol`s (the architectural seam) |
| `orchestrator.py` | the turn loop, history, budget enforcement |
| `metrics.py` | `Stopwatch`, `TurnMetrics`, p50/p95 aggregation |
| `pipeline.py` | `Config → Orchestrator` factory (lazy-imports real backends) |
| `config.py` | declarative backend selection + per-stage options |
| `vad/ stt/ llm/ tts/` | one `mock` + one `real` adapter per stage |
| `audio/io.py` | WAV/list/chunk sources & sinks (mic/playback lazy) |
| `transport/server.py` | LAN WebSocket inference service skeleton |

## 5. The pluggable-backend pattern
Each stage ships two implementations behind one Protocol:
- **mock** — stdlib only, deterministic, used by demo + tests + dev.
- **real** — heavy deps imported *lazily inside the adapter*, so importing
  `voxbox` (and running the test suite) never requires torch/whisper/etc.

```python
Config(vad="energy", stt="mock",          llm="mock",   tts="mock")    # dev/CI
Config(vad="silero", stt="faster_whisper", llm="ollama", tts="piper")  # GPU box
```

## 6. Latency model
`total = vad + stt + llm + tts`. Each turn is timed; `over_budget` is set when
`total > budget_ms`. `MetricsCollector` exposes p50/p95 per stage. The budget is a
**tested** concept (`tests/test_latency_budget.py`), not just a log line.

Indicative production budget (GPU box): VAD ≪10 ms, STT 150–300 ms, LLM 200–500 ms
(first token much sooner with streaming), TTS 100–250 ms → ~0.5–1.0 s/turn.

## 7. Privacy posture
- Default config = all-mock = no network at all.
- Real config: STT/TTS/VAD are in-process; LLM uses a **local** Ollama daemon.
- The WebSocket transport is LAN-only. No telemetry. (Verified by design in tests:
  no backend imports an internet client in the default path.)

## 8. Concurrency note (production)
v0.1's loop is synchronous for testability. On the real client, run capture,
inference, and playback on separate threads/async tasks and **stream** LLM tokens
into a streaming TTS to overlap stages — that's where most latency is won
(see ROADMAP). The contracts already allow chunked/streaming variants later.
