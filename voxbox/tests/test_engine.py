"""Tests for ConversationEngine: the per-chunk decision logic (deterministic, via a
fake generation handle) plus a real-thread smoke test. Also covers the echo-margin
onset detector."""
from voxbox import Config, build_pipeline
from voxbox.contracts import LLMResponse
from voxbox.engine import ConversationEngine, OnsetDetector
from voxbox.contracts import AudioChunk
from tests.helpers import loud, quiet, utterance_chunks, SR


# ── OnsetDetector / echo margin ───────────────────────────────────────────────


def _chunk(pcm):
    return AudioChunk(pcm=pcm, sample_rate=SR)


def test_onset_fires_after_consecutive_loud():
    d = OnsetDetector(threshold_dbfs=-40, onset_frames=3, playback_margin_dbfs=0)
    res = [d.onset(_chunk(loud(20)), busy=False) for _ in range(3)]
    assert res == [False, False, True]


def test_playback_margin_suppresses_self_echo():
    # A medium-loud chunk: above the base threshold, below threshold+margin.
    d = OnsetDetector(threshold_dbfs=-50, onset_frames=1, playback_margin_dbfs=40)
    medium = (b"\x00\x08") * int(SR * 0.02)  # ~ -24 dBFS-ish
    # not busy -> base threshold -> fires
    assert d.onset(_chunk(medium), busy=False) is True
    d.reset()
    # busy -> threshold raised by 40 dB -> same audio no longer trips it
    assert d.onset(_chunk(medium), busy=True) is False


# ── Engine decision logic with a fake handle (deterministic) ──────────────────


class _FakeHandle:
    def __init__(self):
        self.cancelled = False
        self._alive = True

    def cancel(self):
        self.cancelled = True
        self._alive = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self._alive = False


class _FakeSpeaker:
    def __init__(self):
        self.stops = 0
        self._active = False  # fake handle path doesn't drive playback

    def stop(self):
        self.stops += 1

    def is_active(self):
        return self._active


def _two_utterances():
    chunks = []
    ts = 0.0
    for line in ["hello", "hey stop now"]:
        u = utterance_chunks(line, start_ts=ts)
        chunks.extend(u)
        ts = u[-1].ts + 0.3
    return chunks


def _engine_with_fake(barge_in):
    orch = build_pipeline(Config())
    handles = []
    interrupts = []
    sp = _FakeSpeaker()

    def factory(seg):
        h = _FakeHandle()
        handles.append(h)
        return h

    eng = ConversationEngine(
        orch, sp, barge_in=barge_in, onset_frames=3, playback_margin_dbfs=0,
        on_interrupt=lambda: interrupts.append(1), handle_factory=factory,
    )
    return eng, sp, handles, interrupts


def test_mid_response_barge_in_cancels_active_generation():
    eng, sp, handles, interrupts = _engine_with_fake(barge_in=True)
    turns = eng.run(_two_utterances())

    assert turns == 2
    assert len(handles) == 2              # two turns started
    assert handles[0].cancelled is True   # first reply was cut off mid-response
    assert len(interrupts) >= 1
    assert sp.stops >= 1                   # audio was stopped too


def test_no_barge_in_still_supersedes_but_doesnt_signal_interrupt():
    eng, sp, handles, interrupts = _engine_with_fake(barge_in=False)
    turns = eng.run(_two_utterances())
    assert turns == 2
    assert len(handles) == 2
    assert handles[0].cancelled is True   # 2nd utterance supersedes the 1st gen
    assert interrupts == []               # but no barge-in signal


# ── Real-thread smoke test (runs to completion; no timing asserts) ────────────


class _MultiSentenceLLM:
    def respond_stream(self, text, history):
        for tok in ["Alpha. ", "Beta."]:
            yield tok

    def respond(self, text, history):
        return LLMResponse("Alpha. Beta.")


class _RecordingSpeaker:
    def __init__(self):
        self.played = []
        self._active = False

    def play(self, reply):
        self.played.append(reply)

    def stop(self):
        self._active = False

    def is_active(self):
        return self._active


def test_real_thread_generation_runs_to_completion():
    orch = build_pipeline(Config())
    orch.llm = _MultiSentenceLLM()
    sp = _RecordingSpeaker()
    dones = []
    eng = ConversationEngine(orch, sp, barge_in=True,
                             on_turn=lambda d: dones.append(d.response_text))
    eng.run(utterance_chunks("hello"))  # run() joins the worker before returning
    assert [r.num_samples > 0 for r in sp.played] == [True, True]
    assert dones == ["Alpha. Beta."]


def test_backend_error_is_reported_not_crashed():
    class _BrokenLLM:
        def respond_stream(self, text, history):
            raise RuntimeError("ollama down")
            yield  # make it a generator

        def respond(self, text, history):
            raise RuntimeError("ollama down")

    orch = build_pipeline(Config())
    orch.llm = _BrokenLLM()
    errors = []
    eng = ConversationEngine(orch, _RecordingSpeaker(), barge_in=True,
                             on_error=lambda e: errors.append(str(e)))
    eng.run(utterance_chunks("hello"))  # must not raise
    assert any("ollama down" in e for e in errors)
