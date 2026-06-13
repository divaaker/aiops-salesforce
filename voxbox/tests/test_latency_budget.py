"""Latency is the product (see docs/TESTING.md). These tests prove the budget
machinery actually flags slow turns — using a deliberately slow stub LLM."""
import time

from voxbox.contracts import LLMResponse, SpeechSegment
from voxbox.orchestrator import Orchestrator
from voxbox.vad import EnergyVAD
from voxbox.stt import MockSTT
from voxbox.tts import MockTTS


class _SlowLLM:
    def __init__(self, delay_ms):
        self.delay_ms = delay_ms

    def respond(self, text, history):
        time.sleep(self.delay_ms / 1000.0)
        return LLMResponse("slow reply")


def _orch(llm, budget_ms):
    return Orchestrator(vad=EnergyVAD(), stt=MockSTT(), llm=llm, tts=MockTTS(), budget_ms=budget_ms)


def test_fast_turn_is_within_budget():
    orch = _orch(_SlowLLM(0), budget_ms=1000.0)
    r = orch.handle_segment(SpeechSegment(pcm=b"\x00\x00" * 1600, sample_rate=16000, label="hi"))
    assert r.metrics.over_budget is False
    assert r.metrics.total_ms < 1000.0


def test_slow_turn_trips_budget_flag():
    orch = _orch(_SlowLLM(120), budget_ms=50.0)
    r = orch.handle_segment(SpeechSegment(pcm=b"\x00\x00" * 1600, sample_rate=16000, label="hi"))
    assert r.metrics.llm_ms >= 100.0
    assert r.metrics.over_budget is True
    assert orch.metrics.over_budget_count() == 1


def test_total_is_sum_of_stages():
    orch = _orch(_SlowLLM(30), budget_ms=10000.0)
    r = orch.handle_segment(SpeechSegment(pcm=b"\x00\x00" * 1600, sample_rate=16000, label="hi"))
    m = r.metrics
    assert abs(m.total_ms - (m.vad_ms + m.stt_ms + m.llm_ms + m.tts_ms)) < 1.0
