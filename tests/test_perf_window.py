"""Tests for PerfWindow, which gathers timings and summarises them as
n / p50 / p95 / max every STATS_INTERVAL_SEC.

Note on p50: PerfWindow takes the value at index len // 2 of the sorted list.
For an even count that is the upper of the two middle values, not their
average, so the median of [1, 2, 3, 4] is 3 here, not 2.5.
"""
import time

import pytest

import interpreter
from interpreter import PerfWindow


def test_summary_of_empty_window_is_empty():
    assert PerfWindow().summary() == {}


def test_single_sample():
    pw = PerfWindow()
    pw.add("vad_ms", 2.4)
    assert pw.summary() == {"vad_ms": {"n": 1, "p50": 2.4, "p95": 2.4, "max": 2.4}}


def test_percentiles_of_1_to_100():
    pw = PerfWindow()
    for v in range(1, 101):
        pw.add("x", v)
    s = pw.summary()["x"]
    assert s["n"] == 100
    assert s["p50"] == 51      # sorted index 50 (upper middle, see note above)
    assert s["p95"] == 96      # sorted index 95
    assert s["max"] == 100


def test_order_of_samples_does_not_matter():
    pw = PerfWindow()
    for v in [9, 1, 5, 3, 7]:
        pw.add("x", v)
    assert pw.summary()["x"] == {"n": 5, "p50": 5, "p95": 9, "max": 9}


def test_p95_ignores_a_single_outlier_in_a_large_sample():
    # 99 fast calls and one very slow one: p95 shows "normal slow",
    # max shows the worst case.
    pw = PerfWindow()
    for _ in range(99):
        pw.add("ms", 10)
    pw.add("ms", 500)
    s = pw.summary()["ms"]
    assert s["p95"] == 10
    assert s["max"] == 500


def test_values_are_rounded_to_2_decimal_places():
    pw = PerfWindow()
    pw.add("ms", 1.23456)
    s = pw.summary()["ms"]
    assert s["p50"] == 1.23
    assert s["max"] == 1.23


def test_each_metric_is_summarised_separately():
    pw = PerfWindow()
    pw.add("vad_ms", 2)
    pw.add("vad_ms", 4)
    pw.add("whisper_ms", 300)
    s = pw.summary()
    assert set(s) == {"vad_ms", "whisper_ms"}
    assert s["vad_ms"]["n"] == 2
    assert s["whisper_ms"]["n"] == 1


def test_summary_resets_the_samples():
    pw = PerfWindow()
    pw.add("x", 1)
    pw.summary()
    assert pw.summary() == {}


# ── due() ─────────────────────────────────────────────────────────────────
# Rather than sleep for 10 s, these tests move `started` back in time.

def test_new_window_is_not_due():
    assert not PerfWindow().due()


def test_window_is_due_after_the_interval():
    pw = PerfWindow()
    pw.started = time.monotonic() - interpreter.STATS_INTERVAL_SEC
    assert pw.due()


def test_window_is_not_due_just_before_the_interval():
    pw = PerfWindow()
    pw.started = time.monotonic() - (interpreter.STATS_INTERVAL_SEC - 1)
    assert not pw.due()


def test_summary_restarts_the_timer():
    pw = PerfWindow()
    pw.started = time.monotonic() - interpreter.STATS_INTERVAL_SEC
    assert pw.due()
    pw.summary()
    assert not pw.due()


def test_due_follows_the_interval_setting(monkeypatch):
    # monkeypatch changes a value for one test and puts it back afterwards.
    monkeypatch.setattr(interpreter, "STATS_INTERVAL_SEC", 0)
    assert PerfWindow().due()


@pytest.mark.parametrize("count", [1, 2, 19, 20, 21])
def test_p95_index_never_runs_past_the_end(count):
    # min(len - 1, ...) in summary() guards the index. Check small counts.
    pw = PerfWindow()
    for v in range(count):
        pw.add("x", v)
    s = pw.summary()["x"]
    assert s["p95"] <= s["max"]
    assert s["n"] == count
