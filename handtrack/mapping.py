"""Konversi koordinat ternormalisasi ke piksel layar."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenBounds:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width - 1

    @property
    def bottom(self) -> int:
        return self.top + self.height - 1

    def clamp(self, x: float, y: float) -> tuple[int, int]:
        px = min(max(int(round(x)), self.left), self.right)
        py = min(max(int(round(y)), self.top), self.bottom)
        return px, py


def map_to_bounds(
    norm: tuple[float, float], area: tuple[float, float, float, float], bounds: ScreenBounds
) -> tuple[float, float]:
    """Petakan (x, y) kamera 0..1 ke kotak `area` lalu ke `bounds` layar."""
    x_min, x_max, y_min, y_max = area
    nx = min(max(norm[0], x_min), x_max)
    ny = min(max(norm[1], y_min), y_max)
    fx = (nx - x_min) / (x_max - x_min)
    fy = (ny - y_min) / (y_max - y_min)
    return (
        bounds.left + fx * (bounds.width - 1),
        bounds.top + fy * (bounds.height - 1),
    )
