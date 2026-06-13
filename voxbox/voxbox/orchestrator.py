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

import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Union

from .contracts import AudioChunk, AudioReply, SpeechSegment, Transcript
from .contracts import LLM, STT, TTS, VAD
from .metrics import MetricsCollector, Stopwatch, TurnMetrics
from .streaming import StreamChunk, StreamDone, sentence_chunker


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

    # ── Streaming path ────────────────────────────────────────────────────────

    def detect(self, chunk: AudioChunk) -> Optional[SpeechSegment]:
        """Run only VAD on a chunk (used by streaming loops that then call
        stream_segment)."""
        return self.vad.process(chunk)

    def _token_stream(self, text: str) -> Iterator[str]:
        """Use the LLM's streaming API if present, else fall back to one chunk."""
        if hasattr(self.llm, "respond_stream"):
            return self.llm.respond_stream(text, self.history)
        return iter([self.llm.respond(text, self.history).text])

    def stream_segment(
        self, segment: SpeechSegment
    ) -> Iterator[Union[StreamChunk, StreamDone]]:
        """Stream a turn: STT, then token-stream the LLM, synthesize each sentence
        as it completes, and yield it immediately. Ends with a StreamDone carrying
        timing — including first_audio_ms, the latency the user actually feels."""
        turn_start = time.perf_counter()
        with Stopwatch() as t_stt:
            transcript = self.stt.transcribe(segment)

        first_audio_ms: Optional[float] = None
        tts_ms_total = 0.0
        parts: List[str] = []

        for sentence in sentence_chunker(self._token_stream(transcript.text)):
            with Stopwatch() as t_tts:
                reply = self.tts.synthesize(sentence)
            tts_ms_total += t_tts.ms
            if first_audio_ms is None:
                first_audio_ms = (time.perf_counter() - turn_start) * 1000.0
            parts.append(sentence)
            yield StreamChunk(text=sentence, reply=reply)

        total = (time.perf_counter() - turn_start) * 1000.0
        response_text = " ".join(parts)
        if first_audio_ms is None:  # empty reply
            first_audio_ms = total
        turn = TurnMetrics(
            vad_ms=0.0,
            stt_ms=round(t_stt.ms, 3),
            tts_ms=round(tts_ms_total, 3),
            llm_ms=round(max(0.0, total - t_stt.ms - tts_ms_total), 3),
            total_ms=round(total, 3),
            first_audio_ms=round(first_audio_ms, 3),
            budget_ms=self.budget_ms,
            over_budget=total > self.budget_ms,
            transcript=transcript.text,
            response=response_text,
        )
        self.metrics.record(turn)
        self.history.append({"role": "user", "content": transcript.text})
        self.history.append({"role": "assistant", "content": response_text})
        yield StreamDone(transcript=transcript, response_text=response_text, metrics=turn)
