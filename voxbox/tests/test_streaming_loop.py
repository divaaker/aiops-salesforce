"""End-to-end test of the streaming live loop with barge-in, using a deterministic
fake speaker (no hardware). Verifies sentences are enqueued per turn and that the
user's next utterance flushes the queue before the new turn streams."""
from voxbox import Config, build_pipeline
from voxbox.contracts import LLMResponse
from voxbox.runtime import run_conversation_streaming
from tests.helpers import utterance_chunks


class _MultiSentenceLLM:
    def respond_stream(self, text, history):
        for tok in ["Part one. ", "Part two."]:
            yield tok

    def respond(self, text, history):
        return LLMResponse("Part one. Part two.")


class _FakeStreamSpeaker:
    def __init__(self):
        self.played = []
        self.stops = 0
        self._active = False

    def play(self, reply):
        self.played.append(reply)
        self._active = True

    def stop(self):
        self.stops += 1
        self._active = False

    def is_active(self):
        return self._active


def _two_utterances():
    chunks = []
    ts = 0.0
    for line in ["hello", "hey stop talking"]:
        u = utterance_chunks(line, start_ts=ts)
        chunks.extend(u)
        ts = u[-1].ts + 0.3
    return chunks


def test_streaming_loop_enqueues_sentences_per_turn():
    orch = build_pipeline(Config())
    orch.llm = _MultiSentenceLLM()
    sp = _FakeStreamSpeaker()
    chunks_seen, turns_done = [], []

    turns = run_conversation_streaming(
        orch, _two_utterances(), sp, barge_in=False,
        on_chunk=lambda c: chunks_seen.append(c.text),
        on_turn=lambda d: turns_done.append(d.response_text),
    )

    assert turns == 2
    # 2 sentences per turn * 2 turns
    assert chunks_seen == ["Part one.", "Part two.", "Part one.", "Part two."]
    assert len(sp.played) == 4
    assert turns_done == ["Part one. Part two.", "Part one. Part two."]


def test_streaming_loop_barge_in_flushes_before_next_turn():
    orch = build_pipeline(Config())
    orch.llm = _MultiSentenceLLM()
    sp = _FakeStreamSpeaker()
    interrupts = []

    turns = run_conversation_streaming(
        orch, _two_utterances(), sp, barge_in=True, onset_frames=3,
        on_interrupt=lambda: interrupts.append(1),
    )

    assert turns == 2
    assert len(interrupts) >= 1   # 2nd utterance talked over the 1st reply
    assert sp.stops >= 1          # queue/playback was flushed
