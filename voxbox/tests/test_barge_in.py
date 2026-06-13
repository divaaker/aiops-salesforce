"""Barge-in: the agent must stop talking when the user speaks over it. Tested with
a fake interruptible speaker — no audio hardware."""
from voxbox.contracts import AudioChunk
from voxbox.pipeline import build_pipeline
from voxbox.config import Config
from voxbox.runtime import BargeInController, run_conversation
from tests.helpers import loud, quiet, utterance_chunks, SR


class _FakeInterruptibleSpeaker:
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


def _chunk(pcm, ts=0.0):
    return AudioChunk(pcm=pcm, sample_rate=SR, ts=ts)


def test_no_interrupt_when_speaker_idle():
    sp = _FakeInterruptibleSpeaker()  # not active
    c = BargeInController(sp, onset_frames=2)
    fired = [c.on_chunk(_chunk(loud(20))) for _ in range(5)]
    assert not any(fired)
    assert sp.stops == 0


def test_interrupt_after_consecutive_loud_chunks():
    sp = _FakeInterruptibleSpeaker()
    sp.play(object())  # agent is now speaking
    c = BargeInController(sp, onset_frames=3)
    results = [c.on_chunk(_chunk(loud(20))) for _ in range(3)]
    assert results == [False, False, True]  # fires on the 3rd loud frame
    assert sp.stops == 1
    assert sp.is_active() is False


def test_silence_resets_onset_counter():
    sp = _FakeInterruptibleSpeaker()
    sp.play(object())
    c = BargeInController(sp, onset_frames=3)
    c.on_chunk(_chunk(loud(20)))
    c.on_chunk(_chunk(loud(20)))
    c.on_chunk(_chunk(quiet(20)))   # reset
    c.on_chunk(_chunk(loud(20)))
    assert sp.stops == 0            # never reached 3 in a row


def test_run_conversation_barge_in_end_to_end():
    """Turn 1 starts playing; the user's next utterance interrupts it, and still
    gets answered as turn 2."""
    orch = build_pipeline(Config())
    sp = _FakeInterruptibleSpeaker()
    interrupts = []

    chunks = []
    ts = 0.0
    for line in ["hello", "actually stop"]:
        u = utterance_chunks(line, start_ts=ts)
        chunks.extend(u)
        ts = u[-1].ts + 0.3

    turns = run_conversation(
        orch, chunks, sp, barge_in=True, onset_frames=3,
        on_interrupt=lambda: interrupts.append(1),
    )

    assert turns == 2                 # both utterances answered
    assert len(sp.played) == 2
    assert len(interrupts) >= 1       # the 2nd utterance cut off the 1st reply
    assert sp.stops >= 1


def test_barge_in_can_be_disabled():
    orch = build_pipeline(Config())
    sp = _FakeInterruptibleSpeaker()
    interrupts = []
    chunks = []
    ts = 0.0
    for line in ["hello", "more talking"]:
        u = utterance_chunks(line, start_ts=ts)
        chunks.extend(u)
        ts = u[-1].ts + 0.3
    run_conversation(orch, chunks, sp, barge_in=False,
                     on_interrupt=lambda: interrupts.append(1))
    assert interrupts == []
    assert sp.stops == 0
