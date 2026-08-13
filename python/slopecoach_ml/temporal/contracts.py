"""Provisional A4 temporal pose contracts for Python research/reference only."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import FrameGeometry, Joint, PersonPose2D


class TemporalProvenance(StrEnum):
    OBSERVED = "OBSERVED"
    INTERPOLATED = "INTERPOLATED"
    MISSING = "MISSING"


class TemporalPoseQuality(StrEnum):
    GOOD = "GOOD"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class TemporalPoseConfig:
    minimum_joint_confidence: float = 0.30
    minimum_identity_confidence: float = 0.62
    maximum_interpolation_gap_us: int = 300_000
    hard_reset_gap_us: int = 500_000
    one_euro_min_cutoff_hz: float = 1.0
    one_euro_beta: float = 0.05
    one_euro_derivative_cutoff_hz: float = 1.0

    def validate(self) -> None:
        for name in ("minimum_joint_confidence", "minimum_identity_confidence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("maximum_interpolation_gap_us", "hard_reset_gap_us"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.hard_reset_gap_us < self.maximum_interpolation_gap_us:
            raise ValueError("hard_reset_gap_us must be >= maximum_interpolation_gap_us")
        for name in (
            "one_euro_min_cutoff_hz",
            "one_euro_derivative_cutoff_hz",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.one_euro_beta, bool) or not math.isfinite(self.one_euro_beta):
            raise ValueError("one_euro_beta must be finite")
        if self.one_euro_beta < 0:
            raise ValueError("one_euro_beta must be non-negative")


@dataclass(frozen=True)
class TargetPoseSample:
    timestamp_us: int
    frame_index: int
    target_id: str
    active_track_id: int | None
    identity_state: TargetIdentityState
    identity_confidence: float
    geometry: FrameGeometry
    raw_target_pose: PersonPose2D | None
    limitations: tuple[str, ...] = ()
    explicit_discontinuity: bool = False

    def validate(self) -> None:
        if isinstance(self.timestamp_us, bool) or not isinstance(self.timestamp_us, int):
            raise TypeError("timestamp_us must be an integer")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if self.timestamp_us < 0 or self.frame_index < 0:
            raise ValueError("timestamp/frame index must be non-negative")
        if not self.target_id:
            raise ValueError("target_id must be non-empty")
        if not math.isfinite(self.identity_confidence) or not 0 <= self.identity_confidence <= 1:
            raise ValueError("identity_confidence must be finite and in [0, 1]")
        self.geometry.validate()
        if self.raw_target_pose is not None:
            self.raw_target_pose.validate(self.geometry)


@dataclass(frozen=True)
class TemporalJoint2D:
    joint: Joint
    raw_x_px: float | None
    raw_y_px: float | None
    support_x_px: float | None
    support_y_px: float | None
    stabilized_x_px: float | None
    stabilized_y_px: float | None
    support_confidence: float | None
    provenance: TemporalProvenance
    was_smoothed: bool

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["joint"] = self.joint.value
        data["provenance"] = self.provenance.value
        return data


@dataclass(frozen=True)
class StabilizedPoseSample:
    timestamp_us: int
    frame_index: int
    temporal_segment_id: int | None
    geometry: FrameGeometry
    target_id: str
    active_track_id: int | None
    identity_state: TargetIdentityState
    joints: dict[Joint, TemporalJoint2D]
    observed_joint_count: int
    interpolated_joint_count: int
    missing_joint_count: int
    temporal_quality: TemporalPoseQuality
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ("IMAGE_SPACE_2D_ONLY",)

    def joint(self, joint: Joint) -> TemporalJoint2D | None:
        return self.joints.get(joint)

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_us": self.timestamp_us,
            "frame_index": self.frame_index,
            "temporal_segment_id": self.temporal_segment_id,
            "frame_geometry": self.geometry.to_dict(),
            "target_id": self.target_id,
            "active_track_id": self.active_track_id,
            "identity_state": self.identity_state.value,
            "joints": {joint.value: point.to_dict() for joint, point in self.joints.items()},
            "observed_joint_count": self.observed_joint_count,
            "interpolated_joint_count": self.interpolated_joint_count,
            "missing_joint_count": self.missing_joint_count,
            "temporal_quality": self.temporal_quality.value,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


@dataclass
class TemporalPoseRun:
    samples: list[StabilizedPoseSample]
    temporal_segment_count: int
    filter_reset_count: int
    short_gap_interpolation_count: int
    long_gap_unfilled_count: int
    interpolation_seconds: float = 0.0
    stabilization_seconds: float = 0.0
    stability: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": [sample.to_dict() for sample in self.samples],
            "temporal_segment_count": self.temporal_segment_count,
            "filter_reset_count": self.filter_reset_count,
            "short_gap_interpolation_count": self.short_gap_interpolation_count,
            "long_gap_unfilled_count": self.long_gap_unfilled_count,
            "interpolation_seconds": self.interpolation_seconds,
            "stabilization_seconds": self.stabilization_seconds,
            "stability": self.stability,
        }
