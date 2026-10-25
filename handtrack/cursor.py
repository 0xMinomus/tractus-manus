"""Pemetaan kursor + smoothing + pembekuan posisi saat klik/scroll/jeda."""

from __future__ import annotations

from .config import ActiveAreaConfig, ControlConfig
from .filters import OneEuroFilter
from .mapping import ScreenBounds, map_to_bounds


class CursorController:
    """Mengubah titik tangan ternormalisasi menjadi posisi kursor piksel.

    Filter bekerja pada koordinat ternormalisasi agar perilakunya sama di semua
    resolusi. Saat dibekukan, kursor tidak bergerak; ketika dilepas, kursor
    meluncur halus dari posisi beku ke posisi tangan (offset decay).
    """

    def __init__(self, bounds: ScreenBounds, cfg: ControlConfig) -> None:
        self.bounds = bounds
        self._base_area = cfg.active_area
        self._sensitivity = cfg.sensitivity
        self._area_cfg = self._effective_area()
        self._area = (
            self._area_cfg.x_min,
            self._area_cfg.x_max,
            self._area_cfg.y_min,
            self._area_cfg.y_max,
        )
        sm = cfg.smoothing
        self._fx = OneEuroFilter(sm.min_cutoff, sm.beta, sm.derivative_cutoff)
        self._fy = OneEuroFilter(sm.min_cutoff, sm.beta, sm.derivative_cutoff)
        self._reacquire_s = cfg.reacquire_ms / 1000.0
        self._reacquire_active_s = self._reacquire_s
        self._pos: tuple[int, int] | None = None
        self._frozen = False
        self._freeze_until: float | None = None
        self._reacquire_pending = False
        self._offset = (0.0, 0.0)
        self._offset_at: float | None = None

    @property
    def position(self) -> tuple[int, int] | None:
        return self._pos

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def smoothing(self) -> float:
        return self._fx.min_cutoff

    @property
    def area(self) -> ActiveAreaConfig:
        """Area kamera efektif setelah sensitivitas diterapkan."""
        return self._area_cfg

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    def _effective_area(self) -> ActiveAreaConfig:
        a = self._base_area
        s = self._sensitivity
        cx = (a.x_min + a.x_max) / 2.0
        cy = (a.y_min + a.y_max) / 2.0
        half_w = (a.x_max - a.x_min) / (2.0 * s)
        half_h = (a.y_max - a.y_min) / (2.0 * s)
        return ActiveAreaConfig(
            x_min=max(0.0, cx - half_w),
            x_max=min(1.0, cx + half_w),
            y_min=max(0.0, cy - half_h),
            y_max=min(1.0, cy + half_h),
        )

    def set_sensitivity(self, value: float) -> float:
        """Perbesar = area kamera mengecil = jangkauan sudut layar lebih mudah."""
        self._sensitivity = min(max(value, 0.5), 3.0)
        self._area_cfg = self._effective_area()
        self._area = (
            self._area_cfg.x_min,
            self._area_cfg.x_max,
            self._area_cfg.y_min,
            self._area_cfg.y_max,
        )
        self._reacquire_pending = True
        self._reacquire_active_s = self._reacquire_s
        return self._sensitivity

    def freeze(self, now: float, duration_s: float | None = None) -> None:
        if self._frozen:
            if duration_s is None:
                self._freeze_until = None
            elif self._freeze_until is not None:
                self._freeze_until = max(self._freeze_until, now + duration_s)
            return
        self._frozen = True
        self._freeze_until = None if duration_s is None else now + duration_s
        self._offset = (0.0, 0.0)
        self._offset_at = None

    def unfreeze(self, now: float) -> None:
        self._frozen = False
        self._freeze_until = None
        self._reacquire_pending = True
        self._reacquire_active_s = self._reacquire_s

    def update(self, norm: tuple[float, float], now: float) -> tuple[int, int] | None:
        x_min, x_max, y_min, y_max = self._area
        cx = min(max(norm[0], x_min), x_max)
        cy = min(max(norm[1], y_min), y_max)
        fx = self._fx.filter(cx, now)
        fy = self._fy.filter(cy, now)
        mapped = map_to_bounds((fx, fy), self._area, self.bounds)

        if self._frozen:
            if self._freeze_until is not None and now >= self._freeze_until:
                self._frozen = False
                self._reacquire_pending = True
                # Anchor klik hanya butuh koreksi singkat; jangan bikin drag
                # terasa tertinggal setelah anchor habis.
                self._reacquire_active_s = min(0.03, self._reacquire_s)
            else:
                return None

        if self._pos is None:
            # Kursor pertama kali muncul: langsung ke posisi tangan.
            self._pos = self.bounds.clamp(*mapped)
            self._reacquire_pending = False
            self._offset = (0.0, 0.0)
            self._offset_at = None
            return self._pos

        if self._reacquire_pending:
            self._offset = (self._pos[0] - mapped[0], self._pos[1] - mapped[1])
            self._offset_at = now
            self._reacquire_pending = False

        ox, oy = self._offset
        if self._offset_at is not None:
            t = (
                1.0
                if self._reacquire_active_s <= 0
                else (now - self._offset_at) / self._reacquire_active_s
            )
            if t >= 1.0:
                self._offset = (0.0, 0.0)
                self._offset_at = None
                ox = oy = 0.0
            else:
                t = max(t, 0.0)
                remaining = 1.0 - (t * t * (3.0 - 2.0 * t))
                ox *= remaining
                oy *= remaining

        new_pos = self.bounds.clamp(mapped[0] + ox, mapped[1] + oy)
        if new_pos == self._pos:
            return None
        self._pos = new_pos
        return self._pos

    def set_smoothing(self, min_cutoff: float) -> None:
        cutoff = min(max(min_cutoff, 0.3), 8.0)
        self._fx.min_cutoff = cutoff
        self._fy.min_cutoff = cutoff

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self._offset = (0.0, 0.0)
        self._offset_at = None
        self._reacquire_pending = True
        self._freeze_until = None
