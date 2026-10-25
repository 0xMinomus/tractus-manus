"""Smoke test rendering HUD tanpa kamera."""

from __future__ import annotations

import numpy as np

from handtrack.config import ActiveAreaConfig, UIConfig
from handtrack.gestures import Action, FrameState, Freeze, Mode
from handtrack.mapping import ScreenBounds
from handtrack.overlay import Overlay
from tests.helpers import make_hand


def make_overlay() -> Overlay:
    return Overlay(
        ActiveAreaConfig(),
        UIConfig(),
        pinch_press=0.36,
        screen=ScreenBounds(0, 0, 2560, 1600),
    )


def test_draw_point_and_debug() -> None:
    overlay = make_overlay()
    hand = make_hand("point")
    frame = np.full((540, 960, 3), 40, dtype=np.uint8)
    state = FrameState(
        mode=Mode.POINT,
        has_hand=True,
        cursor=hand.norm(6),
        actions=[Action.LEFT_DOWN],
        freeze=Freeze.ANCHOR,
        active_label="Right",
        active_score=0.97,
        pose=None,
        locked=False,
    )
    before = frame.copy()
    overlay.draw(frame, [hand], state, 1.0, 59.0, (1200, 800), 1.2)
    assert np.any(frame != before)
    overlay.toggle_debug()
    overlay.draw(frame, [hand], state, 1.1, 59.0, (1200, 800), 1.2)

    ambiguous = FrameState(
        mode=Mode.POINT,
        has_hand=True,
        cursor=hand.norm(6),
        active_label="Right",
        active_score=0.9,
        ambiguous=True,
    )
    overlay.draw(frame, [hand], ambiguous, 1.2, 59.0, (1200, 800), 1.2)


def test_draw_paused_and_scroll() -> None:
    overlay = make_overlay()
    hand = make_hand("open")
    paused = FrameState(
        mode=Mode.PAUSED,
        has_hand=True,
        cursor=None,
        actions=[Action.PAUSE],
        freeze=Freeze.HOLD,
        active_label="Right",
        active_score=0.9,
        pose="fist",
        pause_progress=0.6,
    )
    frame = np.full((540, 960, 3), 40, dtype=np.uint8)
    overlay.draw(frame, [hand], paused, 2.0, 30.0, (100, 100), 1.2)

    scrolling = FrameState(
        mode=Mode.SCROLL,
        has_hand=True,
        cursor=None,
        actions=[Action.WHEEL_UP],
        freeze=Freeze.HOLD,
        active_label="Right",
        active_score=0.9,
        pose="scroll",
        scroll_progress=0.4,
    )
    overlay.draw(frame, [hand], scrolling, 3.0, 30.0, (100, 100), 1.2)


def test_no_hand() -> None:
    overlay = make_overlay()
    frame = np.full((540, 960, 3), 40, dtype=np.uint8)
    state = FrameState(mode=Mode.POINT, has_hand=False, cursor=None)
    overlay.draw(frame, [], state, 1.0, 0.0, None, 1.2)
    assert frame.shape == (540, 960, 3)
