"""Backend input mouse: SendInput Windows + mode dry-run untuk uji aman."""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Protocol

from .mapping import ScreenBounds
from .winapi import screen_bounds

log = logging.getLogger(__name__)

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
WHEEL_DELTA = 120

ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT),)


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _INPUTUNION))


class MouseBackend(Protocol):
    def move(self, x: int, y: int) -> None: ...
    def left_down(self) -> None: ...
    def left_up(self) -> None: ...
    def right_click(self) -> None: ...
    def middle_click(self) -> None: ...
    def wheel(self, delta: int) -> None: ...
    def release_all(self) -> None: ...
    def close(self) -> None: ...


class WindowsMouse:
    """Kirim event mouse lewat SendInput (tanpa dependensi eksternal)."""

    def __init__(self, target_monitor: str = "primary") -> None:
        if sys.platform != "win32":
            raise RuntimeError("WindowsMouse hanya berjalan di Windows.")
        self.bounds: ScreenBounds = screen_bounds(target_monitor)
        self._virtual = target_monitor == "virtual"
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._sendinput = self._user32.SendInput
        self._sendinput.argtypes = (wintypes.UINT, ctypes.c_void_p, ctypes.c_int)
        self._sendinput.restype = wintypes.UINT
        self._left_down = False
        self._last_move: tuple[int, int] | None = None
        self._last_error_log = 0.0

    def _send(self, dx: int, dy: int, data: int, flags: int) -> bool:
        event = INPUT(
            type=INPUT_MOUSE,
            u=_INPUTUNION(
                mi=MOUSEINPUT(
                    dx=dx,
                    dy=dy,
                    mouseData=data & 0xFFFFFFFF,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        )
        sent = self._sendinput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
        if sent == 1:
            return True
        now = time.perf_counter()
        if now - self._last_error_log > 2.0:
            self._last_error_log = now
            log.warning(
                "SendInput gagal (error %s). Jika target berupa aplikasi elevated, "
                "jalankan HandTrack sebagai administrator.",
                ctypes.get_last_error(),
            )
        return False

    def move(self, x: int, y: int) -> None:
        if self._last_move == (x, y):
            return
        b = self.bounds
        nx = int(round((x - b.left) * 65535 / max(1, b.width - 1)))
        ny = int(round((y - b.top) * 65535 / max(1, b.height - 1)))
        self._last_move = (x, y)
        flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
        if self._virtual:
            flags |= MOUSEEVENTF_VIRTUALDESK
        self._send(nx, ny, 0, flags)

    def left_down(self) -> None:
        if not self._left_down:
            self._send(0, 0, 0, MOUSEEVENTF_LEFTDOWN)
            self._left_down = True

    def left_up(self) -> None:
        if self._left_down:
            self._send(0, 0, 0, MOUSEEVENTF_LEFTUP)
            self._left_down = False

    def right_click(self) -> None:
        self._send(0, 0, 0, MOUSEEVENTF_RIGHTDOWN)
        self._send(0, 0, 0, MOUSEEVENTF_RIGHTUP)

    def middle_click(self) -> None:
        self._send(0, 0, 0, MOUSEEVENTF_MIDDLEDOWN)
        self._send(0, 0, 0, MOUSEEVENTF_MIDDLEUP)

    def wheel(self, delta: int) -> None:
        self._send(0, 0, delta, MOUSEEVENTF_WHEEL)

    def release_all(self) -> None:
        self.left_up()

    def close(self) -> None:
        self.release_all()


@dataclass
class DryRunMouse:
    """Mencatat event tanpa menyentuh kursor asli."""

    bounds: ScreenBounds = field(default_factory=lambda: ScreenBounds(0, 0, 1920, 1080))
    events: list[tuple[str, int]] = field(default_factory=list)
    position: tuple[int, int] = (0, 0)
    left_is_down: bool = False

    def _record(self, name: str, value: int = 0) -> None:
        self.events.append((name, value))
        log.info("[DRY-RUN] %s%s", name, f" {value}" if value else "")

    def move(self, x: int, y: int) -> None:
        if (x, y) != self.position:
            self.position = (x, y)
            log.debug("[DRY-RUN] move -> (%d, %d)", x, y)

    def left_down(self) -> None:
        if not self.left_is_down:
            self.left_is_down = True
            self._record("left_down")

    def left_up(self) -> None:
        if self.left_is_down:
            self.left_is_down = False
            self._record("left_up")

    def right_click(self) -> None:
        self._record("right_click")

    def middle_click(self) -> None:
        self._record("middle_click")

    def wheel(self, delta: int) -> None:
        self._record("wheel_up" if delta > 0 else "wheel_down", abs(delta))

    def release_all(self) -> None:
        self.left_up()

    def close(self) -> None:
        self.release_all()


def create_mouse(dry_run: bool, target_monitor: str) -> MouseBackend:
    if dry_run:
        return DryRunMouse(bounds=screen_bounds(target_monitor))
    return WindowsMouse(target_monitor)
