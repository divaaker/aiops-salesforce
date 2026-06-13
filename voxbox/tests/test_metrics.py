import time

from voxbox.metrics import MetricsCollector, Stopwatch, TurnMetrics


def test_stopwatch_measures_elapsed():
    with Stopwatch() as t:
        time.sleep(0.02)
    assert t.ms >= 15.0  # ~20ms with slack


def test_percentiles_and_summary():
    mc = MetricsCollector()
    for v in [100, 200, 300, 400, 500]:
        mc.record(TurnMetrics(total_ms=v, stt_ms=v / 2, llm_ms=v / 4, tts_ms=v / 4))
    assert mc.count == 5
    assert mc.p50("total_ms") == 300
    assert mc.p95("total_ms") >= 400
    s = mc.summary()
    assert s["turns"] == 5
    assert "total_p95_ms" in s


def test_over_budget_counting():
    mc = MetricsCollector()
    mc.record(TurnMetrics(total_ms=100, over_budget=False))
    mc.record(TurnMetrics(total_ms=2000, over_budget=True))
    assert mc.over_budget_count() == 1


def test_empty_collector_is_safe():
    mc = MetricsCollector()
    assert mc.p50() == 0.0
    assert mc.summary()["turns"] == 0
