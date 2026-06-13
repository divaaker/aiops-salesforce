"""End-to-end architecture test: drive raw audio chunks through the *whole* turn
loop (VAD -> STT -> LLM -> TTS) and assert the conversation + audio output."""
from voxbox import Config, build_pipeline
from voxbox.audio import write_wav
from tests.helpers import utterance_chunks


def _conversation_chunks():
    chunks = []
    ts = 0.0
    for line in ["hello", "what is your name", "anything else?"]:
        u = utterance_chunks(line, start_ts=ts)
        chunks.extend(u)
        ts = u[-1].ts + 0.5
    return chunks


def test_full_pipeline_three_turns():
    orch = build_pipeline(Config())  # all mock
    results = orch.run(_conversation_chunks())

    assert len(results) == 3
    assert [r.transcript.text for r in results] == ["hello", "what is your name", "anything else?"]
    assert "VoxBox" in results[0].response_text         # greeting
    assert "VoxBox" in results[1].response_text         # name question
    # every turn produced playable audio and recorded metrics
    for r in results:
        assert r.reply.num_samples > 0
        assert r.metrics.total_ms >= 0.0
    assert orch.metrics.count == 3


def test_history_accumulates_across_turns():
    orch = build_pipeline(Config())
    orch.run(_conversation_chunks())
    # 3 turns -> 3 user + 3 assistant messages
    roles = [m["role"] for m in orch.history]
    assert roles == ["user", "assistant"] * 3


def test_reply_is_writable_wav(tmp_path):
    orch = build_pipeline(Config())
    result = orch.run(utterance_chunks("hello"))[0]
    out = tmp_path / "reply.wav"
    write_wav(str(out), result.reply)
    assert out.exists() and out.stat().st_size > 44  # header + data


def test_unknown_backend_raises():
    import pytest

    with pytest.raises(ValueError):
        build_pipeline(Config(stt="does-not-exist"))
