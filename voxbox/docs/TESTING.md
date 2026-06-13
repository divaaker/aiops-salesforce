# QA strategy — testing against the architecture

The brief was "test everything according to architecture." We map each
architectural claim to a test that would fail if the claim broke.

| Architectural claim | Test(s) | What it guards |
|---------------------|---------|----------------|
| Stages are swappable behind Protocols (D1/ADR-0002) | `test_contracts.py::test_backends_satisfy_protocols` | every backend conforms to its Protocol |
| Real adapters keep the same shape | `test_contracts.py::test_real_adapters_also_satisfy_protocols_structurally` | GPU adapters won't break the loop |
| Audio types carry correct timing | `test_contracts.py::test_audio_duration_math`, `test_zero_sample_rate_is_safe` | sample-rate math, no div-by-zero |
| VAD segments speech bounded by silence | `test_vad.py` (emit / no-emit / too-short / reset) | turn-taking correctness |
| Each stage behaves deterministically | `test_stages.py` | reliable E2E assertions |
| Metrics & percentiles are correct | `test_metrics.py` | latency reporting is trustworthy |
| Full loop VAD→STT→LLM→TTS works | `test_pipeline_integration.py` | the product actually runs |
| Conversation history threads through | `test_pipeline_integration.py::test_history_accumulates_across_turns` | multi-turn context |
| Output audio is real/playable | `test_pipeline_integration.py::test_reply_is_writable_wav` | TTS → WAV roundtrip |
| Bad config fails loudly | `test_pipeline_integration.py::test_unknown_backend_raises` | safe configuration |
| Latency budget is enforced | `test_latency_budget.py` | slow turns are flagged; total = Σ stages |
| Audio IO is lossless & timed | `test_audio_io.py` | chunking/WAV roundtrip |

## Test design principles
- **No GPU, no mic, no network** in CI — mocks are stdlib-only and deterministic.
- **Drive the real loop, not shims:** integration tests push raw PCM chunks through
  the actual `EnergyVAD` + `Orchestrator`, not a faked pipeline.
- **Ground-truth via labels:** mock audio sources stamp the intended transcript on
  chunks; `MockSTT` echoes it, so we can assert exact conversation text end-to-end.
- **Latency is asserted, not hoped:** a deliberately slow stub LLM proves the budget
  flag and the `total = vad+stt+llm+tts` invariant.

## Running
```bash
cd voxbox
python3 -m pytest -q          # full suite (currently 29 tests)
python3 -m pytest -q tests/test_latency_budget.py   # one area
python3 scripts/demo.py       # human-readable E2E + metrics
```

## What's deliberately NOT unit-tested here
Real model adapters (Silero/whisper/Ollama/Piper) require GPU/daemons and are
verified on the target node, not in CI. They're covered structurally (shape) and
guarded by the Protocol conformance test. GPU latency/quality benchmarking is a
ROADMAP item (a separate, hardware-gated test tier).
