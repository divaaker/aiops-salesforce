# VoxBox roadmap

## v0.1 — Architecture + tested core ✅ (this PR)
- Contracts, orchestrator turn loop, metrics + latency budget.
- Mock VAD/STT/LLM/TTS (deterministic) → full pipeline runs & is asserted.
- Real adapter stubs (Silero, faster-whisper, Ollama/Gemma, Piper) + wiring notes.
- Audio file/list IO, LAN WebSocket server skeleton.
- 29 tests (contract, integration, latency), demo script, docs + ADRs.

## v0.2 — Real models on the GPU box (hardware-gated)
- Implement the four real adapters end-to-end.
- A hardware-gated test tier: accuracy (WER on a small set) + real latency benchmark.
- `/metrics` dashboard endpoint with live p50/p95.
- **Owner:** Dev 1 · **QA:** WER + latency thresholds in CI-on-GPU runner.

## v0.3 — Real-time client (the Mac)
- Live mic capture + low-latency playback (sounddevice), threaded/async loop.
- WebSocket streaming client ↔ server; reconnect/backpressure handling.
- **Owner:** Dev 2 · **QA:** soak test (long sessions, no drift/leak).

## v0.4 — Conversational quality
- **Streaming everything:** partial STT, token-streamed LLM → streaming TTS to slash
  perceived latency (overlap stages).
- **Barge-in:** VAD interrupts playback when the user starts talking (designer's #1 ask).
- Endpointing tuning + configurable system prompt/persona.
- **Owner:** Dev 1+Dev 2 · **Designer:** interaction spec · **QA:** barge-in latency test.

## v0.5 — Easy deployment (image pillar)
- One-command install per node; Docker/compose for the GPU service.
- Model download/cache management; healthchecks; sample LAN setup guide.
- **Owner:** Dev 2 · **QA:** clean-machine bring-up test.

## Cross-cutting / always-on
- Privacy guarantee kept verifiable (no default-path internet calls).
- Keep the mock pipeline green so contributors without GPUs can develop.
