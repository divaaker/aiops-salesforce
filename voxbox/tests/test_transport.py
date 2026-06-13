"""Tests for the LAN wire protocol: turn_to_meta on the server side and the
client's frame dispatch/playback — all in-memory, no network or audio."""
import json

from voxbox.pipeline import build_pipeline
from voxbox.config import Config
from voxbox.transport.protocol import turn_to_meta
from voxbox.transport.client import ReplyHandler, dispatch_message
from tests.helpers import utterance_chunks


class _FakeSpeaker:
    def __init__(self):
        self.played = []

    def play(self, reply):
        self.played.append(reply)


def _one_turn():
    orch = build_pipeline(Config())
    return orch.run(utterance_chunks("hello"))[0]


def test_turn_to_meta_shape():
    meta = turn_to_meta(_one_turn())
    assert set(meta) == {"transcript", "response", "sample_rate", "metrics"}
    assert meta["transcript"] == "hello"
    assert meta["sample_rate"] == 24000  # MockTTS default
    assert "total_ms" in meta["metrics"]


def test_handler_learns_sample_rate_then_plays():
    sp = _FakeSpeaker()
    h = ReplyHandler(sp)
    h.on_text({"sample_rate": 22050, "transcript": "hi", "response": "yo"})
    h.on_bytes(b"\x01\x00" * 100)
    assert len(sp.played) == 1
    assert sp.played[0].sample_rate == 22050
    assert sp.played[0].num_samples == 100
    assert h.last_meta["response"] == "yo"


def test_dispatch_handles_str_json_and_bytes():
    sp = _FakeSpeaker()
    h = ReplyHandler(sp, default_sample_rate=24000)
    dispatch_message(json.dumps({"sample_rate": 16000}), h)  # str frame
    dispatch_message(b"\x02\x00" * 50, h)                    # binary frame
    assert sp.played[0].sample_rate == 16000


def test_full_protocol_roundtrip_in_memory():
    """Simulate server->client for a real turn without sockets."""
    turn = _one_turn()
    # server side: emit meta json then pcm bytes
    frames = [json.dumps(turn_to_meta(turn)), turn.reply.pcm]
    # client side: dispatch them
    sp = _FakeSpeaker()
    h = ReplyHandler(sp)
    for f in frames:
        dispatch_message(f, h)
    assert h.last_meta["transcript"] == "hello"
    assert len(sp.played) == 1
    assert sp.played[0].sample_rate == turn.reply.sample_rate
    assert sp.played[0].pcm == turn.reply.pcm
