"""
voxbox.metrics
==============
Latency is the product. This module is how we keep ourselves honest: every turn
is timed per-stage and end-to-end, and a configurable budget flags regressions.

`Stopwatch` is a context manager returning elapsed milliseconds; `MetricsCollector`
aggregates turns and exposes p50/p95 so QA and the designer can read the
experience as numbers (see docs/TESTING.md).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class TurnMetrics:
    vad_ms: float = 0.0
    stt_ms: float = 0.0
    llm_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0
    budget_ms: float = 0.0
    over_budget: bool = False
    transcript: str = ""
    response: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class _Elapsed:
    __slots__ = ("ms",)

    def __init__(self) -> None:
        self.ms = 0.0


@contextmanager
def Stopwatch():
    """`with Stopwatch() as t: ...` then read `t.ms` for elapsed milliseconds."""
    e = _Elapsed()
    start = time.perf_counter()
    try:
        yield e
    finally:
        e.ms = (time.perf_counter() - start) * 1000.0


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


class MetricsCollector:
    def __init__(self) -> None:
        self.history: List[TurnMetrics] = []

    def record(self, turn: TurnMetrics) -> None:
        self.history.append(turn)

    @property
    def count(self) -> int:
        return len(self.history)

    def p50(self, field_name: str = "total_ms") -> float:
        return _percentile([getattr(t, field_name) for t in self.history], 50)

    def p95(self, field_name: str = "total_ms") -> float:
        return _percentile([getattr(t, field_name) for t in self.history], 95)

    def over_budget_count(self) -> int:
        return sum(1 for t in self.history if t.over_budget)

    def summary(self) -> dict:
        return {
            "turns": self.count,
            "over_budget": self.over_budget_count(),
            "total_p50_ms": round(self.p50("total_ms"), 1),
            "total_p95_ms": round(self.p95("total_ms"), 1),
            "stt_p50_ms": round(self.p50("stt_ms"), 1),
            "llm_p50_ms": round(self.p50("llm_ms"), 1),
            "tts_p50_ms": round(self.p50("tts_ms"), 1),
        }
