"""Tests for QueueingSpeaker's queue/interrupt logic via an injected writer — no
audio hardware. Covers back-to-back playback (for streaming) and flush-on-stop
(for barge-in)."""
import threading
import time

from voxbox.audio.live import QueueingSpeaker
from voxbox.contracts import AudioReply


def _reply(tag: int) -> AudioReply:
    return AudioReply(pcm=b"\x00\x00" * 10, sample_rate=tag)  # use sample_rate as a tag


def test_plays_clips_back_to_back_in_order():
    played = []
    sp = QueueingSpeaker(writer=lambda reply, should_stop: played.append(reply.sample_rate))
    for t in (1, 2, 3):
        sp.play(_reply(t))
    sp.wait()
    assert played == [1, 2, 3]
    assert sp.is_active() is False
    sp.close()


def test_is_active_true_while_queued():
    gate = threading.Event()

    def writer(reply, should_stop):
        gate.wait(1.0)  # hold the first clip until released

    sp = QueueingSpeaker(writer=writer)
    sp.play(_reply(1))
    # give the worker a moment to pick it up
    for _ in range(100):
        if sp.is_active():
            break
        time.sleep(0.005)
    assert sp.is_active() is True
    gate.set()
    sp.wait()
    assert sp.is_active() is False
    sp.close()


def test_stop_flushes_queue_and_cuts_current_clip():
    started = threading.Event()
    starts = []

    def writer(reply, should_stop):
        starts.append(reply.sample_rate)
        started.set()
        while not should_stop():  # block until barge-in
            time.sleep(0.002)

    sp = QueueingSpeaker(writer=writer)
    sp.play(_reply(1))  # will start and block
    sp.play(_reply(2))  # should be flushed, never started
    assert started.wait(1.0)
    sp.stop()
    assert starts == [1]
    assert sp.is_active() is False
    sp.close()


def test_can_play_again_after_stop():
    played = []
    sp = QueueingSpeaker(writer=lambda reply, should_stop: played.append(reply.sample_rate))
    sp.play(_reply(1))
    sp.wait()
    sp.stop()  # interrupt while idle — must not wedge the worker
    sp.play(_reply(2))
    sp.wait()
    assert played == [1, 2]
    sp.close()
