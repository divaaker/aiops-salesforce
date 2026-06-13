# VoxBox roadmap

## v0.1 — Architecture + tested core ✅ (this PR)
- Contracts, orchestrator turn loop, metrics + latency budget.
- Mock VAD/STT/LLM/TTS (deterministic) → full pipeline runs & is asserted.
- Real adapter stubs (Silero, faster-whisper, Ollama/Gemma, Piper) + wiring notes.
- Audio file/list IO, LAN WebSocket server skeleton.
- 29 tests (contract, integration, latency), demo script, docs + ADRs.

## Neural VAD ✅ (delivered)
- `SileroVAD` (lazy torch) with 512-sample windowing + hangover/min-speech
  segmentation; `VOX_VAD=silero` drops the need to calibrate an energy threshold.
  Segmentation tested via an injected probability fn (no torch needed in CI).

## v0.2 — Real models on the GPU box (hardware-gated)
- Implement the four real adapters end-to-end.
- A hardware-gated test tier: accuracy (WER on a small set) + real latency benchmark.
- `/metrics` dashboard endpoint with live p50/p95.
- **Owner:** Dev 1 · **QA:** WER + latency thresholds in CI-on-GPU runner.

## v0.3 — Real-time client (the Mac) ✅ (delivered)
- Live mic capture + playback via sounddevice (`voxbox/audio/live.py`, raw int16).
- Co-located runner `scripts/talk.py` + env-driven backend selection (`runtime.py`).
- WebSocket split: `scripts/serve.py` (GPU box) + `scripts/talk_remote.py` (Mac),
  shared wire protocol (`transport/protocol.py`, `transport/client.py`).
- Loop + protocol unit-tested without hardware (test_runtime, test_transport).
- ⏭️ Remaining: reconnect/backpressure handling, soak test (long sessions).

## v0.4 — Conversational quality
- ✅ **Barge-in:** `BargeInController` + `StreamingSpeaker` cut off playback when the
  user talks over the agent; driven by `run_conversation` (designer's #1 ask).
  Tested in `tests/test_barge_in.py` (onset detection + end-to-end interrupt).
- ✅ **Streaming responses:** token-streamed LLM → per-sentence TTS via
  `orchestrator.stream_segment` + `sentence_chunker`; reports `first_audio_ms`
  (felt latency). Streaming Ollama adapter + mock. Tested in `tests/test_streaming.py`.
- ✅ Streaming wired into the live loop: `QueueingSpeaker` plays sentences
  back-to-back and flushes on barge-in; `run_conversation_streaming` drives it
  (talk.py, `VOX_STREAM=1`). Tested in `test_queueing_speaker.py` + `test_streaming_loop.py`.
- ✅ Off-thread generation: `ConversationEngine` runs reply generation on a worker
  thread, so barge-in cuts in *mid-response* (cancels generation + flushes audio).
  Echo-margin onset detector raises the threshold while speaking (lightweight AEC
  mitigation). Tested in `tests/test_engine.py`.
- ⏭️ Partial/streaming STT (needs real Whisper streaming) and full acoustic echo
  cancellation remain — both are genuinely hardware/DSP-bound, not stubbed.
- ⏭️ Acoustic echo cancellation (so the mic ignores the agent's own audio without
  headphones), endpointing tuning, configurable system prompt/persona.
- **Owner:** Dev 1+Dev 2 · **Designer:** interaction spec · **QA:** barge-in latency test.

## v0.5 — Easy deployment (image pillar) ✅ (delivered)
- `Dockerfile` (EXTRAS build-arg), `docker-compose.yml` (server + local Ollama,
  healthcheck on /healthz), `Makefile` task runner.
- ⏭️ Remaining: model download/cache automation, multi-client soak test.
- **Owner:** Dev 2.

## Cross-cutting / always-on
- Privacy guarantee kept verifiable (no default-path internet calls).
- Keep the mock pipeline green so contributors without GPUs can develop.
