"""Dependency-free, timestamp-aware One Euro reference filter."""

from __future__ import annotations

import math


def _alpha(cutoff_hz: float, dt_seconds: float) -> float:
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be finite and positive")
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be finite and positive")
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_seconds)


class LowPassFilter:
    def __init__(self) -> None:
        self.value: float | None = None

    def filter(self, value: float, alpha: float) -> float:
        if not math.isfinite(value) or not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("low-pass inputs must be finite and alpha in (0, 1]")
        self.value = value if self.value is None else alpha * value + (1 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class OneEuroFilter1D:
    def __init__(
        self, *, min_cutoff_hz: float = 1.0, beta: float = 0.05, derivative_cutoff_hz: float = 1.0
    ) -> None:
        if min_cutoff_hz <= 0 or derivative_cutoff_hz <= 0 or beta < 0:
            raise ValueError("invalid One Euro configuration")
        self.min_cutoff_hz = min_cutoff_hz
        self.beta = beta
        self.derivative_cutoff_hz = derivative_cutoff_hz
        self._signal = LowPassFilter()
        self._derivative = LowPassFilter()
        self._last_timestamp_us: int | None = None
        self._last_raw: float | None = None

    def filter(self, value: float, timestamp_us: int) -> float:
        if isinstance(timestamp_us, bool) or not isinstance(timestamp_us, int) or timestamp_us < 0:
            raise ValueError("timestamp_us must be a non-negative integer")
        if not math.isfinite(value):
            raise ValueError("One Euro value must be finite")
        if self._last_timestamp_us is None:
            self._last_timestamp_us = timestamp_us
            self._last_raw = value
            return self._signal.filter(value, 1.0)
        dt_us = timestamp_us - self._last_timestamp_us
        if dt_us <= 0:
            raise ValueError("One Euro timestamps must strictly increase")
        dt = dt_us / 1_000_000.0
        derivative = (value - self._last_raw) / dt
        filtered_derivative = self._derivative.filter(
            derivative, _alpha(self.derivative_cutoff_hz, dt)
        )
        cutoff = self.min_cutoff_hz + self.beta * abs(filtered_derivative)
        result = self._signal.filter(value, _alpha(cutoff, dt))
        self._last_timestamp_us = timestamp_us
        self._last_raw = value
        return result

    def reset(self) -> None:
        self._signal.reset()
        self._derivative.reset()
        self._last_timestamp_us = None
        self._last_raw = None
