"""Unit test filter One Euro dan pemetaan waktu."""

from __future__ import annotations

from handtrack.filters import OneEuroFilter


def test_first_sample_is_raw_and_gap_resets() -> None:
    filt = OneEuroFilter(2.0, 6.0, 1.5)
    assert filt.filter(0.2, 0.0) == 0.2
    assert filt.filter(0.8, 1.0) == 0.8  # jeda panjang: reset


def test_smooths_jitter_at_rest() -> None:
    filt = OneEuroFilter(2.0, 6.0, 1.5)
    values = [0.5 + (0.004 if i % 2 else -0.004) for i in range(90)]
    outputs = [filt.filter(value, i / 30.0) for i, value in enumerate(values)]
    warm = outputs[30:]
    assert max(warm) - min(warm) < 0.006


def test_step_response_stays_within_bounds() -> None:
    filt = OneEuroFilter(2.0, 6.0, 1.5)
    filt.filter(0.0, 0.0)
    outputs = [filt.filter(1.0, i / 30.0) for i in range(1, 30)]
    assert all(0.0 <= value <= 1.0 for value in outputs)
    assert outputs[-1] > 0.99
    assert outputs == sorted(outputs)
