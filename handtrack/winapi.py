"""Utilitas Windows: DPI awareness, batas layar, dan jendela always-on-top."""

from __future__ import annotations

import contextlib
import ctypes
import sys

from .mapping import ScreenBounds

SM_CXSCREEN = 0
SM_CYSCREEN = 1
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040


def _require_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError("HandTrack saat ini hanya mendukung Windows.")


def enable_dpi_awareness() -> None:
    """Wajib dipanggil sebelum membuat jendela / membaca metrik layar."""
    _require_windows()
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:  # noqa: BLE001 - Windows lama
        pass
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        if shcore.SetProcessDpiAwareness(2) == 0:
            return
    except Exception:  # noqa: BLE001
        pass
    with contextlib.suppress(Exception):
        ctypes.WinDLL("user32", use_last_error=True).SetProcessDPIAware()


def screen_bounds(target: str = "primary") -> ScreenBounds:
    _require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    if target == "virtual":
        return ScreenBounds(
            int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN)),
            int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN)),
            int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)),
            int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)),
        )
    return ScreenBounds(
        0,
        0,
        int(user32.GetSystemMetrics(SM_CXSCREEN)),
        int(user32.GetSystemMetrics(SM_CYSCREEN)),
    )


def set_always_on_top(window_title: str, enabled: bool = True) -> bool:
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return False
        return bool(
            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST if enabled else HWND_NOTOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        )
    except Exception:  # noqa: BLE001
        return False


def position_window(window_title: str, x: int, y: int) -> bool:
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return False
        return bool(
            user32.SetWindowPos(
                hwnd,
                0,
                int(x),
                int(y),
                0,
                0,
                SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )
        )
    except Exception:  # noqa: BLE001
        return False
