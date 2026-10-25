"""Unit test pemetaan koordinat dan kursor."""

from __future__ import annotations

import pytest

from handtrack.config import (
    ActiveAreaConfig,
    ControlConfig,
    SmoothingConfig,
)
from handtrack.cursor import CursorController
from handtrack.mapping import ScreenBounds, map_to_bounds

BOUNDS = ScreenBounds(0, 0, 1920, 1080)


def make_cursor(
    reacquire_ms: int = 100,
    area: ActiveAreaConfig | None = None,
    sensitivity: float = 1.0,
) -> CursorController:
    cfg = ControlConfig(
        active_area=area or ActiveAreaConfig(0.0, 1.0, 0.0, 1.0),
        smoothing=SmoothingConfig(min_cutoff=10.0, beta=0.0, derivative_cutoff=10.0),
        reacquire_ms=reacquire_ms,
        sensitivity=sensitivity,
    )
    return CursorController(BOUNDS, cfg)


def test_map_to_bounds() -> None:
    area = (0.2, 0.8, 0.2, 0.8)
    bounds = ScreenBounds(100, 50, 1000, 500)
    assert map_to_bounds((0.2, 0.2), area, bounds) == (100.0, 50.0)
    assert map_to_bounds((0.8, 0.8), area, bounds) == (1099.0, 549.0)
    assert map_to_bounds((0.0, 1.0), area, bounds) == (100.0, 549.0)


def test_bounds_clamp() -> None:
    assert ScreenBounds(0, 0, 100, 50).clamp(-5, 999) == (0, 49)
    assert ScreenBounds(-100, -50, 100, 50).clamp(-500, 500) == (-100, -1)


def test_first_update_and_mapping() -> None:
    cursor = make_cursor()
    assert cursor.update((0.5, 0.5), 0.0) == (960, 540)
    assert cursor.update((0.5, 0.5), 0.01) is None


def test_anchor_freeze_then_resume() -> None:
    cursor = make_cursor()
    assert cursor.update((0.5, 0.5), 0.0) == (960, 540)
    cursor.freeze(0.01, 0.09)
    assert cursor.update((0.6, 0.5), 0.05) is None
    # Anchor habis tepat di frame ini: masih di posisi beku.
    assert cursor.update((0.6, 0.5), 0.11) is None
    # Glide anchor sangat singkat (<=30 ms) agar drag langsung mengikuti.
    resumed = cursor.update((0.6, 0.5), 0.16)
    assert resumed is not None
    assert resumed[1] == 540
    assert 1140 <= resumed[0] <= 1151
    assert cursor.update((0.6, 0.5), 0.35) == (1151, 540)
    assert cursor.update((0.6, 0.5), 0.36) is None


def test_hold_freeze_and_reacquire() -> None:
    cursor = make_cursor()
    assert cursor.update((0.5, 0.5), 0.0) == (960, 540)
    cursor.freeze(0.1, None)
    assert cursor.update((0.7, 0.7), 0.2) is None
    assert cursor.frozen
    cursor.unfreeze(0.3)
    assert cursor.update((0.7, 0.7), 0.31) is None
    mid = cursor.update((0.7, 0.7), 0.36)
    assert mid is not None
    assert 960 < mid[0] < 1343
    assert cursor.update((0.7, 0.7), 0.5) == (1343, 755)


def test_clamps_input_and_output() -> None:
    high = make_cursor()
    assert high.update((5.0, 5.0), 0.0) == (1919, 1079)
    low = make_cursor()
    assert low.update((-3.0, -3.0), 0.0) == (0, 0)


def test_smoothing_hotkey_range() -> None:
    cursor = make_cursor()
    cursor.set_smoothing(0.05)
    assert cursor.smoothing == pytest.approx(0.3)
    cursor.set_smoothing(99.0)
    assert cursor.smoothing == pytest.approx(8.0)
    cursor.set_smoothing(1.4)
    assert cursor.smoothing == pytest.approx(1.4)


def test_sensitivity_scales_effective_area() -> None:
    cursor = make_cursor(sensitivity=2.0)
    area = cursor.area
    assert area.x_min == pytest.approx(0.25)
    assert area.x_max == pytest.approx(0.75)
    assert area.y_min == pytest.approx(0.25)
    assert area.y_max == pytest.approx(0.75)
    assert cursor.update((0.25, 0.5), 0.0) == (0, 540)
    cursor2 = make_cursor(sensitivity=2.0)
    assert cursor2.update((0.75, 0.5), 0.0) == (1919, 540)


def test_sensitivity_clamps() -> None:
    cursor = make_cursor()
    assert cursor.set_sensitivity(0.1) == pytest.approx(0.5)
    assert cursor.set_sensitivity(9.0) == pytest.approx(3.0)
