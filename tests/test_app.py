"""Test integrasi glue app: aksi -> backend, freeze, dan kursor."""

from __future__ import annotations

from dataclasses import replace

import pytest

from handtrack.app import HandTrackApp, RunOptions
from handtrack.config import (
    ActiveAreaConfig,
    AppConfig,
    ControlConfig,
    GestureConfig,
    SmoothingConfig,
)
from handtrack.cursor import CursorController
from handtrack.gestures import Action, GestureEngine
from handtrack.input import DryRunMouse
from handtrack.mapping import ScreenBounds
from tests.helpers import make_hand

BOUNDS = ScreenBounds(0, 0, 1920, 1080)


def make_setup():
    cfg = AppConfig(
        control=ControlConfig(
            active_area=ActiveAreaConfig(0.0, 1.0, 0.0, 1.0),
            smoothing=SmoothingConfig(10.0, 0.0, 10.0),
            reacquire_ms=100,
            click_anchor_ms=90,
            sensitivity=1.0,
        ),
        gestures=replace(GestureConfig(), pinch_stable_ms=0),
    )
    app = HandTrackApp(cfg, RunOptions(dry_run=True))
    cursor = CursorController(BOUNDS, cfg.control)
    mouse = DryRunMouse(bounds=BOUNDS)
    engine = GestureEngine(cfg.gestures)
    return app, cursor, mouse, engine


def test_click_anchor_hold_and_release_glue() -> None:
    app, cursor, mouse, engine = make_setup()

    def frame(t: float, hand):
        state = engine.update([hand] if hand else [], t)
        pos = app._apply_state(state, cursor, mouse, t)
        if pos is not None:
            mouse.move(*pos)
        return state

    state = frame(0.1, make_hand("point"))
    assert state.freeze is not None
    assert mouse.position == (873, 432)

    frame(0.2, make_hand("point"))  # unlock gestur
    state = frame(0.3, make_hand("point", pinch="index", dx=100))
    assert state.actions == [Action.LEFT_DOWN]
    assert mouse.left_is_down
    # Anchor membekukan kursor walau tangan sudah bergeser 100 px.
    assert mouse.position == (873, 432)

    frame(0.38, make_hand("point", pinch="index", dx=100))
    assert mouse.position == (873, 432)
    frame(0.50, make_hand("point", pinch="index", dx=100))
    assert mouse.position == (873, 432)
    frame(0.55, make_hand("point", pinch="index", dx=100))
    assert mouse.position[0] > 873
    frame(0.70, make_hand("point", pinch="index", dx=100))
    assert mouse.position == pytest.approx((1065, 432), abs=1)

    state = frame(0.8, make_hand("point", dx=100))
    assert state.actions == [Action.LEFT_UP]
    assert not mouse.left_is_down
    assert mouse.events[0] == ("left_down", 0)
    assert mouse.events[1] == ("left_up", 0)


def test_hand_loss_releases_button_glue() -> None:
    app, cursor, mouse, engine = make_setup()

    def frame(t: float, hand):
        state = engine.update([hand] if hand else [], t)
        pos = app._apply_state(state, cursor, mouse, t)
        if pos is not None:
            mouse.move(*pos)
        return state

    frame(0.1, make_hand("point"))
    frame(0.2, make_hand("point"))
    frame(0.3, make_hand("point", pinch="index"))
    assert mouse.left_is_down
    state = frame(0.4, None)
    assert state.actions == [Action.LEFT_UP]
    assert not mouse.left_is_down
    assert mouse.events[-1] == ("left_up", 0)


def test_right_click_at_frozen_position_glue() -> None:
    app, cursor, mouse, engine = make_setup()

    def frame(t: float, hand):
        state = engine.update([hand], t)
        pos = app._apply_state(state, cursor, mouse, t)
        if pos is not None:
            mouse.move(*pos)
        return state

    frame(0.1, make_hand("point"))
    frame(0.2, make_hand("point"))
    frame(0.3, make_hand("point", dx=100))
    frame(0.4, make_hand("point", dx=100))
    before = mouse.position
    frame(0.5, make_hand("point", pinch="middle", dx=100))
    assert mouse.position == before
    state = frame(0.6, make_hand("point", dx=100))
    assert state.actions == [Action.RIGHT_CLICK]
    assert ("right_click", 0) in mouse.events
    assert mouse.position == before
