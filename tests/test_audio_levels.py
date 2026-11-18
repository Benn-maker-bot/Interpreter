"""Tests for the audio level helpers rms() and levels_dbfs().

Pattern used in every test ("Arrange, Act, Assert"):
  Arrange - build the input (here, a block of 16-bit audio samples)
  Act     - call the function under test
  Assert  - check the result with `assert`
"""
import math

import numpy as np
import pytest

from interpreter import levels_dbfs, rms


def pcm16(samples):
    """Turn a list of integers into raw 16-bit audio bytes, the format the
    capture thread produces (int16, little-endian)."""
    return np.array(samples, dtype=np.int16).tobytes()


# ── rms() ─────────────────────────────────────────────────────────────────

def test_rms_of_empty_block_is_zero():
    assert rms(b"") == 0.0


def test_rms_of_silence_is_zero():
    assert rms(pcm16([0] * 512)) == 0.0


def test_rms_of_constant_signal_equals_its_level():
    # RMS of a constant value is that value (sign doesn't matter).
    assert rms(pcm16([1000, -1000] * 256)) == pytest.approx(1000.0)


def test_rms_ignores_an_odd_trailing_byte():
    # Half a sample can't be read as int16, so rms() should ignore it.
    assert rms(pcm16([1000] * 4) + b"\x01") == pytest.approx(1000.0)


def test_rms_of_a_single_byte_is_zero():
    # Less than one whole sample: nothing to measure.
    assert rms(b"\x01") == 0.0


def test_rms_handles_the_most_negative_sample_without_overflow():
    # -32768 squared doesn't fit in int16. rms() converts to float first,
    # so the answer must be 32768, not a wrapped-round nonsense value.
    assert rms(pcm16([-32768] * 8)) == pytest.approx(32768.0)


def test_rms_of_two_level_signal():
    # Mean of squares: (3² + 4²) / 2 = 12.5, so RMS = sqrt(12.5).
    assert rms(pcm16([3, 4] * 100)) == pytest.approx(math.sqrt(12.5))


# ── levels_dbfs() ─────────────────────────────────────────────────────────

def test_levels_of_empty_block_are_the_floor():
    assert levels_dbfs(b"") == (-100.0, -100.0)


def test_levels_of_silence_are_the_floor():
    assert levels_dbfs(pcm16([0] * 512)) == (-100.0, -100.0)


def test_full_scale_square_wave_is_about_0_dbfs():
    # A square wave at maximum level has RMS = peak = full scale.
    rms_db, peak_db = levels_dbfs(pcm16([32767, -32768] * 256))
    assert rms_db == pytest.approx(0.0, abs=0.01)
    assert peak_db == pytest.approx(0.0, abs=0.01)


def test_sine_wave_rms_is_3_db_below_its_peak():
    # Engineering check: for a sine wave, RMS = peak / sqrt(2), i.e. -3.01 dB.
    t = np.arange(4800)
    sine = (16384 * np.sin(2 * math.pi * 1000 * t / 48000)).astype(np.int16)
    rms_db, peak_db = levels_dbfs(sine.tobytes())
    assert peak_db - rms_db == pytest.approx(3.01, abs=0.05)
    assert peak_db == pytest.approx(-6.02, abs=0.05)   # half scale = -6 dB


def test_peak_uses_the_largest_magnitude_whichever_sign():
    # One negative spike in quiet audio sets the peak.
    _, peak_db = levels_dbfs(pcm16([100] * 100 + [-16384]))
    assert peak_db == pytest.approx(-6.02, abs=0.01)


def test_smallest_non_zero_level_is_about_minus_90_dbfs():
    # One LSB (value 1) is 20*log10(1/32768) = -90.3 dBFS: the 16-bit noise
    # floor. It is above the -100 floor, so it must not be clamped.
    rms_db, peak_db = levels_dbfs(pcm16([1] * 64))
    assert rms_db == pytest.approx(-90.31, abs=0.01)
    assert peak_db == pytest.approx(-90.31, abs=0.01)


def test_levels_dbfs_agrees_with_rms():
    # The two functions are written separately; this checks they agree.
    data = pcm16([1200, -800, 300, -50] * 64)
    rms_db, _ = levels_dbfs(data)
    assert rms_db == pytest.approx(20 * math.log10(rms(data) / 32768), abs=1e-4)
