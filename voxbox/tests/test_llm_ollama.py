"""Tests for the real Ollama adapter that run WITHOUT a daemon by stubbing the
HTTP call. Verifies payload construction, response parsing, history threading,
and friendly errors. (A live smoke test against a real daemon lives in
scripts/chat_ollama.py.)"""
import json
from urllib import error

import pytest

from voxbox.contracts import SpeechSegment
from voxbox.llm.ollama import OllamaError, OllamaLLM
from voxbox.orchestrator import Orchestrator
from voxbox.vad import EnergyVAD
from voxbox.stt import MockSTT
from voxbox.tts import MockTTS


class _FakeResp:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, capture, body):
    def fake_urlopen(req, timeout=None):
        capture["url"] = req.full_url
        capture["payload"] = json.loads(req.data.decode()) if req.data else None
        return _FakeResp(json.dumps(body).encode())

    monkeypatch.setattr("voxbox.llm.ollama.request.urlopen", fake_urlopen)


def test_respond_builds_payload_and_parses(monkeypatch):
    cap = {}
    _patch_urlopen(monkeypatch, cap, {"message": {"content": "  hi there  "}})
    llm = OllamaLLM(model="gemma3")

    out = llm.respond("hello", history=[])

    assert out.text == "hi there"  # stripped
    assert cap["url"].endswith("/api/chat")
    assert cap["payload"]["model"] == "gemma3"
    assert cap["payload"]["stream"] is False
    roles = [m["role"] for m in cap["payload"]["messages"]]
    assert roles == ["system", "user"]  # system prompt + the user turn


def test_respond_threads_history(monkeypatch):
    cap = {}
    _patch_urlopen(monkeypatch, cap, {"message": {"content": "ok"}})
    hist = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]

    OllamaLLM().respond("c", history=hist)

    roles = [m["role"] for m in cap["payload"]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_connection_failure_raises_friendly(monkeypatch):
    def boom(req, timeout=None):
        raise error.URLError("connection refused")

    monkeypatch.setattr("voxbox.llm.ollama.request.urlopen", boom)
    with pytest.raises(OllamaError) as ei:
        OllamaLLM().respond("hi", [])
    assert "Is it running" in str(ei.value)


def test_malformed_response_raises(monkeypatch):
    _patch_urlopen(monkeypatch, {}, {"unexpected": "shape"})
    with pytest.raises(OllamaError):
        OllamaLLM().respond("hi", [])


def test_is_available_true_and_false(monkeypatch):
    monkeypatch.setattr(
        "voxbox.llm.ollama.request.urlopen", lambda u, timeout=None: _FakeResp(b"{}")
    )
    assert OllamaLLM().is_available() is True

    def boom(u, timeout=None):
        raise error.URLError("down")

    monkeypatch.setattr("voxbox.llm.ollama.request.urlopen", boom)
    assert OllamaLLM().is_available() is False


def test_ollama_runs_in_the_real_turn_loop(monkeypatch):
    """Same path scripts/chat_ollama.py uses: real LLM stage, mock STT/TTS."""
    _patch_urlopen(monkeypatch, {}, {"message": {"content": "local reply"}})
    orch = Orchestrator(
        vad=EnergyVAD(), stt=MockSTT(), llm=OllamaLLM(), tts=MockTTS(), budget_ms=5000.0
    )
    turn = orch.handle_segment(
        SpeechSegment(pcm=b"\x00\x00" * 1600, sample_rate=16000, label="hi")
    )
    assert turn.transcript.text == "hi"
    assert turn.response_text == "local reply"
    assert turn.reply.num_samples > 0
    assert orch.metrics.count == 1
