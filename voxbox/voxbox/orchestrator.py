"""
voxbox.orchestrator
===================
The turn loop — the "ORCHESTRATOR" box from the diagram. It owns no models; it
wires the four stages together, times each one, maintains conversation history,
and enforces the latency budget.

    audio chunks ─▶ VAD ─(segment)─▶ STT ─▶ LLM ─▶ TTS ─▶ AudioReply (playback)

Everything is synchronous and side-effect-free here so it is trivial to test; the
real-time transport (mic/socket/playback) lives outside, in audio/ and transport/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .contracts import AudioChunk, AudioReply, SpeechSegment, Transcript
from .contracts import LLM, STT, TTS, VAD
from .metrics import MetricsCollector, Stopwatch, TurnMetrics


@dataclass
class TurnResult:
    transcript: Transcript
    response_text: str
    reply: AudioReply
    metrics: TurnMetrics
    segment: SpeechSegment


@dataclass
class Orchestrator:
    vad: VAD
    stt: STT
    llm: LLM
    tts: TTS
    budget_ms: float = 1200.0
    metrics: MetricsCollector = field(default_factory=MetricsCollector)
    history: List[dict] = field(default_factory=list)

    def feed(self, chunk: AudioChunk) -> Optional[TurnResult]:
        """Push one audio chunk. Returns a TurnResult only when an utterance
        completed and was answered, else None."""
        with Stopwatch() as t_vad:
            segment = self.vad.process(chunk)
        if segment is None:
            return None
        return self._handle_segment(segment, vad_ms=t_vad.ms)

    def handle_segment(self, segment: SpeechSegment) -> TurnResult:
        """Process an already-segmented utterance (used by tests and by transports
        that do VAD upstream)."""
        return self._handle_segment(segment, vad_ms=0.0)

    def _handle_segment(self, segment: SpeechSegment, vad_ms: float) -> TurnResult:
        with Stopwatch() as t_stt:
            transcript = self.stt.transcribe(segment)
        with Stopwatch() as t_llm:
            response = self.llm.respond(transcript.text, self.history)
        with Stopwatch() as t_tts:
            reply = self.tts.synthesize(response.text)

        total = vad_ms + t_stt.ms + t_llm.ms + t_tts.ms
        turn = TurnMetrics(
            vad_ms=round(vad_ms, 3),
            stt_ms=round(t_stt.ms, 3),
            llm_ms=round(t_llm.ms, 3),
            tts_ms=round(t_tts.ms, 3),
            total_ms=round(total, 3),
            budget_ms=self.budget_ms,
            over_budget=total > self.budget_ms,
            transcript=transcript.text,
            response=response.text,
        )
        self.metrics.record(turn)

        # keep conversational context for the LLM
        self.history.append({"role": "user", "content": transcript.text})
        self.history.append({"role": "assistant", "content": response.text})

        return TurnResult(
            transcript=transcript,
            response_text=response.text,
            reply=reply,
            metrics=turn,
            segment=segment,
        )

    def run(self, chunks: Iterable[AudioChunk]) -> List[TurnResult]:
        """Drive the loop over a finite stream of chunks (file source / tests)."""
        results: List[TurnResult] = []
        for chunk in chunks:
            res = self.feed(chunk)
            if res is not None:
                results.append(res)
        return results
