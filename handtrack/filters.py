"""Filter One Euro untuk gerak kursor yang halus namun responsif."""

from __future__ import annotations

import math


class _LowPass:
    def __init__(self) -> None:
        self.value: float | None = None

    def reset(self) -> None:
        self.value = None

    def filter(self, x: float, alpha: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = alpha * x + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    """Filter 1D: cutoff naik saat kecepatan naik (sedikit lag saat cepat)."""

    def __init__(self, min_cutoff: float, beta: float, derivative_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t: float | None = None
        self._last_input: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last_t = None
        self._last_input = None

    def filter(self, x: float, t: float) -> float:
        gap = self._last_t is None or (t - self._last_t) > 0.25
        if gap or self._last_input is None:
            # Sampel pertama / jeda panjang: mulai dari nilai mentah, tanpa lag.
            self._x.reset()
            self._dx.reset()
            self._x.value = x
            self._dx.value = 0.0
            self._last_t = t
            self._last_input = x
            return x

        dt = max(t - self._last_t, 1e-6)
        raw_dx = (x - self._last_input) / dt
        dx = self._dx.filter(raw_dx, self._alpha(self.derivative_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(dx)
        y = self._x.filter(x, self._alpha(cutoff, dt))
        self._last_t = t
        self._last_input = x
        return y
