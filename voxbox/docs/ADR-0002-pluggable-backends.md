# ADR-0002: Pluggable backends behind Protocols, with lazy-imported real models

- **Status:** Accepted
- **Date:** 2026-06-13
- **Deciders:** PO, Dev 1, QA

## Context
The four stages (VAD, STT, LLM, TTS) use models that change frequently and have
heavy, platform-specific dependencies (torch/CUDA, faster-whisper, Ollama, Piper).
We must (a) allow swapping models without touching the turn loop, and (b) keep the
package importable and fully testable on machines without those deps or a GPU.

## Decision
1. Define one `Protocol` per stage in `contracts.py`. The `Orchestrator` depends only
   on these Protocols.
2. Ship two implementations per stage: a **mock** (stdlib-only, deterministic) and a
   **real** adapter. Selection is by name in `Config`; `pipeline.py` is the factory.
3. **Lazy-import** heavy deps *inside* each real adapter's `__init__`, never at module
   top level. Real backends are declared as optional `extras` in `pyproject.toml`.

## Consequences
- ➕ `import voxbox` and `pytest` need nothing but the stdlib.
- ➕ Swapping a model is a one-line `Config` change.
- ➕ Each node installs only the extras it needs (`[whisper]`, `[silero]`, …).
- ➕ Protocol conformance is enforced by a test, so a bad adapter fails CI early.
- ➖ Misconfiguration surfaces at build time (mitigated: `build_pipeline` raises a clear
  `ValueError` on unknown backends — tested).
- ➖ Slightly more indirection than calling models directly; judged worth it.

## Alternatives considered
- **Direct model calls in the orchestrator:** simplest, but couples the loop to specific
  libraries and makes GPU-free testing impossible. Rejected.
- **Plugin entry-points / dynamic discovery:** more flexible, premature for v0.1. The
  registry in `pipeline.py` is enough and can grow into entry-points later.
