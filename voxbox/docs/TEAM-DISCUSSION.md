# Team discussion & decisions — VoxBox kickoff

**Attendees:** Product Owner (facilitator), Dev 1 (backend/ML), Dev 2 (audio/transport),
QA, Designer. **Goal:** turn the reference image (a fully-local voice agent split
across a Mac orchestrator + a GPU box) into a buildable, testable plan.

## 1. What the image tells us (shared reading)
- A **real-time voice assistant**; the headline promise is **privacy**: nothing
  leaves the local network.
- Two physical roles: **Mac** = mic capture, playback, orchestration, timing/metrics;
  **GPU box ("3 Blackwells")** = the heavy models.
- Pipeline stages, in order: **VAD (Silero) → STT (faster-whisper) → LLM (Gemma) → TTS**.
- Cross-cutting pillars: 100% local, private, open-source, easy deployment.

## 2. Positions
- **Dev 1:** The four models will change often (new Whisper, new Gemma, swap TTS).
  Lock each stage behind one interface or we rewrite the loop every time.
- **Dev 2:** Real-time audio + a second machine is the hard part. But we should be
  able to develop the *logic* with no mic and no GPU, then attach hardware later.
- **QA:** "Test everything according to architecture" means: (a) prove each backend
  honors its contract, (b) prove the end-to-end loop, (c) **measure latency** — for a
  voice agent, latency *is* correctness. We need deterministic backends to assert on.
- **Designer:** The felt experience = responsiveness + barge-in (interrupting the
  agent). Make latency observable (p50/p95 per stage) and make VAD turn-taking a
  first-class, tunable thing.
- **PO:** We can't ship a GPU in CI. So the *first* deliverable is the architecture
  + a fully-mocked, fully-tested core, with real adapters stubbed and documented.

## 3. Decisions (what we agreed)
| # | Decision | Owner | Rationale |
|---|----------|-------|-----------|
| D1 | Every stage is a `Protocol`; backends are pluggable by config | Dev 1 | future-proof model swaps → [ADR-0002](ADR-0002-pluggable-backends.md) |
| D2 | Mac↔GPU split kept as an explicit transport boundary (WebSocket) | Dev 2 | mirrors the diagram → [ADR-0001](ADR-0001-distributed-mac-gpu-split.md) |
| D3 | Ship mock backends (zero deps) so the whole loop runs/tests w/o GPU | Dev 1+QA | CI + fast local dev |
| D4 | Latency budget is a tested, enforced concept | QA | latency = UX |
| D5 | Core stdlib-only; heavy deps lazy-imported as extras | Dev 1 | install only what a node needs |
| D6 | Deterministic mocks (labels/rules) so E2E asserts exact transcripts | QA | reliable tests |
| D7 | Privacy is a non-functional requirement we keep verifiable | All | no stage may call the public internet by default |

## 4. Scope for v0.1 (this PR)
- ✅ Contracts, orchestrator turn loop, metrics + budget.
- ✅ Mock VAD/STT/LLM/TTS that make the pipeline real and assertable.
- ✅ Real adapter stubs (Silero, faster-whisper, Ollama/Gemma, Piper) with wiring notes.
- ✅ Audio file/list IO + LAN WebSocket server skeleton.
- ✅ Full test suite (contract, integration, latency) + demo.
- ⏭️ Out of scope now: live mic streaming, barge-in, GPU benchmarking → [ROADMAP](ROADMAP.md).

## 5. Definition of done (v0.1)
`pytest` green on stdlib alone; `scripts/demo.py` prints a 3-turn conversation with
per-stage metrics; docs explain how to flip to real models on the GPU box.
