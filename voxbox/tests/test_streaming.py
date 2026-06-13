"""Tests for streaming responses: sentence chunking, the mock streaming LLM, the
Ollama streaming adapter (mocked HTTP), and the orchestrator's stream_segment."""
import json

from voxbox import Config, build_pipeline
from voxbox.contracts import LLMResponse, SpeechSegment, StreamingLLM
from voxbox.llm import RuleBasedLLM
from voxbox.llm.ollama import OllamaLLM
from voxbox.streaming import StreamChunk, StreamDone, sentence_chunker


# ── sentence_chunker ──────────────────────────────────────────────────────────


def test_chunker_splits_on_sentence_ends_as_tokens_arrive():
    tokens = ["Hello", " world.", " How ", "are ", "you?", " Bye"]
    assert list(sentence_chunker(tokens)) == ["Hello world.", "How are you?", "Bye"]


def test_chunker_handles_no_terminal_punctuation():
    assert list(sentence_chunker(["just ", "a ", "fragment"])) == ["just a fragment"]


def test_chunker_empty():
    assert list(sentence_chunker([])) == []


def test_chunker_multiple_terminators():
    assert list(sentence_chunker(["Wow!!! ", "Really?"])) == ["Wow!!!", "Really?"]


# ── mock streaming LLM ────────────────────────────────────────────────────────


def test_rulebased_is_a_streaming_llm():
    assert isinstance(RuleBasedLLM(), StreamingLLM)


def test_respond_stream_reconstructs_respond():
    llm = RuleBasedLLM()
    streamed = "".join(llm.respond_stream("hello", []))
    assert streamed == llm.respond("hello", []).text


# ── orchestrator streaming ────────────────────────────────────────────────────


class _ThreeSentenceLLM:
    def respond_stream(self, text, history):
        for tok in ["First. ", "Second. ", "Third."]:
            yield tok

    def respond(self, text, history):
        return LLMResponse("First. Second. Third.")


def test_stream_segment_yields_chunks_then_done():
    orch = build_pipeline(Config())
    orch.llm = _ThreeSentenceLLM()
    events = list(orch.stream_segment(
        SpeechSegment(pcm=b"\x00\x00" * 1600, sample_rate=16000, label="go")
    ))
    chunks = [e for e in events if isinstance(e, StreamChunk)]
    dones = [e for e in events if isinstance(e, StreamDone)]

    assert [c.text for c in chunks] == ["First.", "Second.", "Third."]
    assert all(c.reply.num_samples > 0 for c in chunks)
    assert len(dones) == 1
    done = dones[0]
    assert done.transcript.text == "go"
    assert done.response_text == "First. Second. Third."
    # the StreamDone is the LAST event
    assert isinstance(events[-1], StreamDone)


def test_streaming_records_metrics_and_history():
    orch = build_pipeline(Config())
    orch.llm = _ThreeSentenceLLM()
    list(orch.stream_segment(SpeechSegment(pcm=b"\x00\x00" * 1600, sample_rate=16000, label="hi")))
    assert orch.metrics.count == 1
    m = orch.metrics.history[0]
    # first audio must arrive no later than the whole turn
    assert 0.0 <= m.first_audio_ms <= m.total_ms + 1.0
    assert [x["role"] for x in orch.history] == ["user", "assistant"]


def test_stream_segment_falls_back_for_nonstreaming_llm():
    class _PlainLLM:
        def respond(self, text, history):
            return LLMResponse("One sentence only.")

    orch = build_pipeline(Config())
    orch.llm = _PlainLLM()  # no respond_stream
    chunks = [e for e in orch.stream_segment(
        SpeechSegment(pcm=b"\x00\x00" * 1600, sample_rate=16000, label="x")
    ) if isinstance(e, StreamChunk)]
    assert [c.text for c in chunks] == ["One sentence only."]


# ── Ollama streaming adapter (mocked HTTP) ────────────────────────────────────


class _FakeStreamResp:
    def __init__(self, lines):
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ollama_respond_stream_parses_line_json(monkeypatch):
    lines = [
        json.dumps({"message": {"content": "Hello"}}).encode(),
        json.dumps({"message": {"content": " world."}}).encode(),
        json.dumps({"message": {"content": ""}, "done": True}).encode(),
    ]
    monkeypatch.setattr(
        "voxbox.llm.ollama.request.urlopen",
        lambda req, timeout=None: _FakeStreamResp(lines),
    )
    out = list(OllamaLLM().respond_stream("hi", []))
    assert out == ["Hello", " world."]
