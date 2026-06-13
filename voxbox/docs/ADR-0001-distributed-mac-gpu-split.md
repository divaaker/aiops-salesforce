# ADR-0001: Keep the Mac-orchestrator / GPU-box split as an explicit transport boundary

- **Status:** Accepted
- **Date:** 2026-06-13
- **Deciders:** PO, Dev 1, Dev 2

## Context
The reference architecture runs capture/playback/orchestration on a Mac and the
heavy models on a separate GPU machine ("3 Blackwells"), communicating over the
local network. We need this split to be real (it's the privacy + performance story)
without forcing every developer to own two machines.

## Decision
Model the split as a single, explicit **transport boundary**. The `Orchestrator` is
transport-agnostic and operates on in-memory contracts; a thin transport layer
(`voxbox/transport/`, a LAN WebSocket: PCM in → JSON meta + PCM out) carries data
between the Mac client and the GPU service. Because the boundary is the only coupling
point, the exact same `Orchestrator` can run **co-located** (one box) or **split**.

## Consequences
- ➕ Dev/CI run co-located with mocks; production runs split with real models — no
  code fork.
- ➕ Privacy is enforceable at one chokepoint (LAN-only, no telemetry).
- ➕ The GPU box can serve multiple clients later.
- ➖ Split mode adds network latency/serialization; mitigated by sending raw PCM and
  (future) streaming partial STT/LLM/TTS.
- ➖ Two deployment targets to document (covered in README + ROADMAP).

## Alternatives considered
- **Single fat machine only:** simpler, but loses the Mac-as-thin-client UX and the
  multi-GPU scaling shown in the image.
- **gRPC instead of WebSocket:** great typing, but WebSocket is trivial for bidirectional
  binary audio streaming from browsers/native clients and easier to debug. Revisit if
  we need strict schemas.
