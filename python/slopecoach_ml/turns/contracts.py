"""Provisional image-space turn contracts; not skiing direction or physical angle."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol


class TurnPhaseSign(StrEnum):
    POSITIVE_PHASE = "POSITIVE_PHASE"
    NEGATIVE_PHASE = "NEGATIVE_PHASE"


class TurnSegmentStatus(StrEnum):
    VALID = "VALID"
    PARTIAL = "PARTIAL"
    REJECTED_SHORT = "REJECTED_SHORT"
    REJECTED_LONG = "REJECTED_LONG"
    REJECTED_LOW_PROMINENCE = "REJECTED_LOW_PROMINENCE"
    REJECTED_LOW_COVERAGE = "REJECTED_LOW_COVERAGE"


@dataclass(frozen=True)
class TurnSegmentationConfig:
    minimum_signal_confidence: float = 0.30
    minimum_peak_prominence: float = 0.08
    minimum_peak_amplitude: float = 0.08
    minimum_peak_separation_us: int = 400_000
    minimum_turn_duration_us: int = 300_000
    maximum_turn_duration_us: int = 4_000_000
    minimum_valid_samples_per_turn: int = 3
    maximum_missing_ratio: float = 0.40
    zero_crossing_tolerance: float = 1e-9

    def validate(self) -> None:
        for name in (
            "minimum_signal_confidence",
            "minimum_peak_prominence",
            "minimum_peak_amplitude",
            "maximum_missing_ratio",
            "zero_crossing_tolerance",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.minimum_signal_confidence <= 1:
            raise ValueError("minimum_signal_confidence must be in [0, 1]")
        if self.minimum_peak_prominence < 0 or self.minimum_peak_amplitude < 0:
            raise ValueError("peak thresholds must be non-negative")
        if not 0 <= self.maximum_missing_ratio <= 1 or self.zero_crossing_tolerance < 0:
            raise ValueError("turn ratio/tolerance configuration is invalid")
        for name in (
            "minimum_peak_separation_us",
            "minimum_turn_duration_us",
            "maximum_turn_duration_us",
            "minimum_valid_samples_per_turn",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.maximum_turn_duration_us < self.minimum_turn_duration_us:
            raise ValueError("turn duration bounds are invalid")
        if self.minimum_valid_samples_per_turn < 1:
            raise ValueError("minimum_valid_samples_per_turn must be positive")


@dataclass(frozen=True)
class TurnSignalSample:
    timestamp_us: int
    temporal_segment_id: int | None
    value: float | None
    support_confidence: float | None
    provenance: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PeakCandidate:
    timestamp_us: int
    temporal_segment_id: int
    value: float
    prominence: float
    phase_sign: TurnPhaseSign
    sample_index: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["phase_sign"] = self.phase_sign.value
        return data


@dataclass(frozen=True)
class ZeroCrossing:
    timestamp_us: int
    temporal_segment_id: int
    direction: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TurnSegment:
    turn_id: str
    temporal_segment_id: int
    start_timestamp_us: int | None
    apex_timestamp_us: int
    end_timestamp_us: int | None
    phase_sign: TurnPhaseSign
    peak_value: float
    prominence: float
    duration_us: int | None
    valid_sample_count: int
    interpolated_sample_count: int
    missing_ratio: float | None
    confidence: float
    status: TurnSegmentStatus

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["phase_sign"] = self.phase_sign.value
        data["status"] = self.status.value
        return data


class PeakDetector(Protocol):
    def detect(
        self, samples: list[TurnSignalSample], config: TurnSegmentationConfig
    ) -> list[PeakCandidate]: ...
