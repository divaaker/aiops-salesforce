from voxbox.audio import chunks_from_pcm, wav_source, write_wav
from voxbox.contracts import AudioReply


def test_chunking_preserves_total_bytes_and_increments_ts():
    pcm = b"\x01\x02" * 16000  # 1s @16k
    chunks = chunks_from_pcm(pcm, sample_rate=16000, chunk_ms=20.0, start_ts=0.0)
    assert sum(len(c.pcm) for c in chunks) == len(pcm)
    assert chunks[0].ts == 0.0
    assert abs((chunks[1].ts - chunks[0].ts) - 0.02) < 1e-9
    assert all(c.sample_rate == 16000 for c in chunks)


def test_label_propagates_to_chunks():
    chunks = chunks_from_pcm(b"\x00\x00" * 320, sample_rate=16000, label="hi")
    assert all(c.label == "hi" for c in chunks)


def test_wav_roundtrip(tmp_path):
    reply = AudioReply(pcm=b"\x11\x22" * 2400, sample_rate=24000)
    p = tmp_path / "a.wav"
    write_wav(str(p), reply)
    read_back = list(wav_source(str(p), chunk_ms=20.0))
    assert sum(len(c.pcm) for c in read_back) == len(reply.pcm)
    assert read_back[0].sample_rate == 24000
