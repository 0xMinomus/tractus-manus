"""Unit test backend input (tanpa menyentuh mouse asli)."""

from __future__ import annotations

import ctypes
import sys

import pytest

from handtrack.input import INPUT, MOUSEINPUT, DryRunMouse
from handtrack.mapping import ScreenBounds


@pytest.mark.skipif(sys.platform != "win32", reason="struktur SendInput hanya Windows")
def test_sendinput_struct_size() -> None:
    size = ctypes.sizeof(INPUT)
    if sys.maxsize > 2**32:
        assert size == 40
        assert ctypes.sizeof(MOUSEINPUT) == 32
    else:
        assert size == 28


def test_dry_run_events() -> None:
    mouse = DryRunMouse(bounds=ScreenBounds(0, 0, 800, 600))
    mouse.move(10, 20)
    assert mouse.position == (10, 20)
    mouse.left_down()
    mouse.left_down()
    assert mouse.left_is_down
    mouse.left_up()
    mouse.left_up()
    assert not mouse.left_is_down
    mouse.right_click()
    mouse.middle_click()
    mouse.wheel(120)
    mouse.wheel(-120)
    mouse.release_all()
    assert mouse.events == [
        ("left_down", 0),
        ("left_up", 0),
        ("right_click", 0),
        ("middle_click", 0),
        ("wheel_up", 120),
        ("wheel_down", 120),
    ]
