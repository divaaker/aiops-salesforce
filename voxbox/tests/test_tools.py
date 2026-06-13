"""Tests for the tool-calling layer: mock provider, the tool loop, the Ollama tool
model (mocked HTTP), and the MCP provider (fake session). No network/SDK needed."""
import json

import pytest

from voxbox.contracts import LLMResponse
from voxbox.tools import (
    MockToolProvider, ScriptedToolModel, ToolCall, ToolCallingLLM, ToolProvider, ToolSpec,
)
from voxbox.tools.base import ToolModel
from voxbox.tools.mcp import MCPToolProvider
from voxbox.llm.ollama import OllamaToolModel


# ── mock provider ─────────────────────────────────────────────────────────────


def test_mock_provider_is_a_provider_and_runs_tools():
    p = MockToolProvider()
    assert isinstance(p, ToolProvider)
    names = [t.name for t in p.list_tools()]
    assert "add" in names and "get_time" in names
    assert p.call_tool("add", {"a": 2, "b": 3}) == "5"
    with pytest.raises(KeyError):
        p.call_tool("nope", {})


# ── tool loop ─────────────────────────────────────────────────────────────────


def test_scripted_model_is_a_tool_model():
    assert isinstance(ScriptedToolModel([]), ToolModel)


def test_tool_loop_calls_tool_then_returns_final_text():
    model = ScriptedToolModel([
        ("", [ToolCall("add", {"a": 2, "b": 3})]),
        ("The sum is 5.", []),
    ])
    llm = ToolCallingLLM(model, MockToolProvider())
    out = llm.respond("add 2 and 3", [])
    assert isinstance(out, LLMResponse)
    assert out.text == "The sum is 5."
    assert [c.name for c in llm.last_tool_calls] == ["add"]


def test_tool_loop_no_call_returns_immediately():
    model = ScriptedToolModel([("just chatting", [])])
    out = ToolCallingLLM(model, MockToolProvider()).respond("hi", [])
    assert out.text == "just chatting"


def test_tool_loop_handles_tool_error_without_crashing():
    model = ScriptedToolModel([
        ("", [ToolCall("nope", {})]),     # unknown tool -> provider raises
        ("Sorry, that tool failed.", []),
    ])
    out = ToolCallingLLM(model, MockToolProvider()).respond("do it", [])
    assert out.text == "Sorry, that tool failed."


def test_tool_loop_respects_max_steps():
    # model always wants to call a tool -> loop must terminate via max_steps
    class _Loopy:
        def chat(self, messages, tools):
            return ("", [ToolCall("get_time", {})])

    out = ToolCallingLLM(_Loopy(), MockToolProvider(), max_steps=2).respond("time?", [])
    assert isinstance(out, LLMResponse)  # returned, didn't hang


# ── Ollama tool model (mocked HTTP) ───────────────────────────────────────────


class _FakeResp:
    def __init__(self, body):
        self._b = json.dumps(body).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ollama_tool_model_parses_tool_calls(monkeypatch):
    cap = {}

    def fake_urlopen(req, timeout=None):
        cap["payload"] = json.loads(req.data.decode())
        return _FakeResp({"message": {
            "content": "",
            "tool_calls": [{"function": {"name": "add", "arguments": {"a": 2, "b": 3}}}],
        }})

    monkeypatch.setattr("voxbox.llm.ollama.request.urlopen", fake_urlopen)
    text, calls = OllamaToolModel(model="llama3.1").chat(
        [{"role": "user", "content": "add"}], MockToolProvider().list_tools())
    assert text == ""
    assert calls[0].name == "add" and calls[0].arguments == {"a": 2, "b": 3}
    # tools were sent in the OpenAI-style function format
    assert cap["payload"]["tools"][0]["function"]["name"] == "add"


def test_ollama_tool_model_parses_string_arguments(monkeypatch):
    monkeypatch.setattr(
        "voxbox.llm.ollama.request.urlopen",
        lambda req, timeout=None: _FakeResp({"message": {
            "content": "", "tool_calls": [
                {"function": {"name": "add", "arguments": "{\"a\": 1, \"b\": 4}"}}]}}),
    )
    _, calls = OllamaToolModel().chat([], [])
    assert calls[0].arguments == {"a": 1, "b": 4}


def test_ollama_tool_model_in_full_loop(monkeypatch):
    """OllamaToolModel + MockToolProvider through ToolCallingLLM."""
    seq = [
        {"message": {"content": "", "tool_calls": [
            {"function": {"name": "add", "arguments": {"a": 10, "b": 5}}}]}},
        {"message": {"content": "The answer is 15.", "tool_calls": []}},
    ]
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        r = _FakeResp(seq[calls["n"]])
        calls["n"] += 1
        return r

    monkeypatch.setattr("voxbox.llm.ollama.request.urlopen", fake_urlopen)
    llm = ToolCallingLLM(OllamaToolModel(), MockToolProvider())
    out = llm.respond("what is 10 + 5?", [])
    assert out.text == "The answer is 15."
    assert llm.last_tool_calls[0].name == "add"


# ── MCP provider (fake session) ───────────────────────────────────────────────


class _FakeTool:
    def __init__(self, name, desc, schema):
        self.name = name
        self.description = desc
        self.inputSchema = schema


class _FakeToolsResult:
    def __init__(self, tools):
        self.tools = tools


class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeCallResult:
    def __init__(self, texts):
        self.content = [_FakeContent(t) for t in texts]


class _FakeSession:
    def list_tools(self):
        return _FakeToolsResult([_FakeTool("weather", "Get weather", {"type": "object"})])

    def call_tool(self, name, arguments):
        return _FakeCallResult([f"{name}({arguments}) -> sunny"])


def test_mcp_provider_maps_tools_and_results():
    p = MCPToolProvider(session=_FakeSession())
    assert isinstance(p, ToolProvider)
    specs = p.list_tools()
    assert specs[0].name == "weather" and specs[0].description == "Get weather"
    assert "sunny" in p.call_tool("weather", {"city": "NYC"})


def test_mcp_provider_requires_session():
    with pytest.raises(RuntimeError):
        MCPToolProvider(session=None)


def test_parse_oauth_callback():
    from voxbox.tools.mcp import parse_oauth_callback

    code, state = parse_oauth_callback(
        "http://localhost:8765/callback?code=abc123&state=xyz")
    assert code == "abc123"
    assert state == "xyz"

    code, state = parse_oauth_callback("http://localhost:1/callback?code=only")
    assert code == "only" and state is None

    with pytest.raises(ValueError):
        parse_oauth_callback("http://localhost:1/callback?state=nocode")
