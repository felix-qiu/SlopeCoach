"""Provisional A5 image-space biomechanics research contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from slopecoach_ml.pose import Joint


class BiomechanicsFeatureFamily(StrEnum):
    STANCE_PROXY = "STANCE_PROXY"
    BALANCE_PROXY = "BALANCE_PROXY"
    SYMMETRY_PROXY = "SYMMETRY_PROXY"
    TIMING_PROXY = "TIMING_PROXY"
    EDGE_CONTROL_PROXY = "EDGE_CONTROL_PROXY"


class BiomechanicsFactScope(StrEnum):
    FRAME = "FRAME"
    TEMPORAL_SEGMENT = "TEMPORAL_SEGMENT"
    TURN = "TURN"


class BiomechanicsFactStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    REQUIRED_JOINT_MISSING = "REQUIRED_JOINT_MISSING"
    REQUIRED_JOINT_OUT_OF_FRAME = "REQUIRED_JOINT_OUT_OF_FRAME"
    UNSUPPORTED_PIXEL_ASPECT_RATIO = "UNSUPPORTED_PIXEL_ASPECT_RATIO"
    DEGENERATE_GEOMETRY = "DEGENERATE_GEOMETRY"
    INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"
    TURN_BOUNDARY_UNAVAILABLE = "TURN_BOUNDARY_UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class BiomechanicsFeatureConfig:
    minimum_joint_support_confidence: float = 0.30
    square_pixel_tolerance: float = 1e-6
    minimum_aggregate_samples: int = 3
    minimum_derivative_dt_us: int = 1
    apex_match_tolerance_us: int = 150_000

    def validate(self) -> None:
        if (
            not math.isfinite(self.minimum_joint_support_confidence)
            or not 0 <= self.minimum_joint_support_confidence <= 1
        ):
            raise ValueError("minimum_joint_support_confidence must be in [0, 1]")
        if not math.isfinite(self.square_pixel_tolerance) or self.square_pixel_tolerance < 0:
            raise ValueError("square_pixel_tolerance must be finite and non-negative")
        for name in (
            "minimum_aggregate_samples",
            "minimum_derivative_dt_us",
            "apex_match_tolerance_us",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class BiomechanicsFact:
    feature_id: str
    family: BiomechanicsFeatureFamily
    scope: BiomechanicsFactScope
    unit: str
    value: float | int | None
    status: BiomechanicsFactStatus
    timestamp_us: int | None = None
    temporal_segment_id: int | None = None
    signal_run_id: int | None = None
    turn_id: str | None = None
    support_confidence: float | None = None
    required_joints: tuple[Joint, ...] = ()
    observed_joint_count: int = 0
    interpolated_joint_count: int = 0
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is not BiomechanicsFactStatus.AVAILABLE and self.value is not None:
            raise ValueError("unavailable biomechanics fact value must be null")
        if self.value is not None and (
            isinstance(self.value, bool) or not math.isfinite(self.value)
        ):
            raise ValueError("biomechanics fact value must be finite")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["family"] = self.family.value
        data["scope"] = self.scope.value
        data["status"] = self.status.value
        data["required_joints"] = [joint.value for joint in self.required_joints]
        data["limitations"] = list(self.limitations)
        return data


@dataclass(frozen=True)
class FeatureAggregate:
    feature_id: str
    temporal_segment_id: int
    unit: str
    sample_count: int
    available_count: int
    support_ratio: float
    median: float | None
    minimum: float | None
    maximum: float | None
    range: float | None
    median_support_confidence: float | None
    observed_only_sample_count: int
    interpolated_support_sample_count: int
    observed_only_ratio: float
    interpolated_support_ratio: float
    status: BiomechanicsFactStatus

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class TurnBiomechanicsResult:
    turn_id: str
    temporal_segment_id: int
    signal_run_id: int
    phase_sign: str
    facts: tuple[BiomechanicsFact, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "temporal_segment_id": self.temporal_segment_id,
            "signal_run_id": self.signal_run_id,
            "phase_sign": self.phase_sign,
            "facts": [fact.to_dict() for fact in self.facts],
        }


@dataclass(frozen=True)
class TemporalBiomechanicsResult:
    contract_version: str
    config: BiomechanicsFeatureConfig
    frame_facts: tuple[BiomechanicsFact, ...]
    temporal_segment_features: tuple[FeatureAggregate, ...]
    turn_features: tuple[TurnBiomechanicsResult, ...]
    feature_coverage: dict[str, dict[str, object]]
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = field(
        default=(
            "IMAGE_SPACE_2D_ONLY_NOT_PHYSICAL_3D",
            "CAMERA_VIEW_DEPENDENT",
            "NO_PHYSICAL_COM_OR_EDGE_ANGLE",
            "PYTHON_RESEARCH_REFERENCE_ONLY",
        )
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "config": asdict(self.config),
            "frame_facts": [fact.to_dict() for fact in self.frame_facts],
            "temporal_segment_features": [
                feature.to_dict() for feature in self.temporal_segment_features
            ],
            "turn_features": [feature.to_dict() for feature in self.turn_features],
            "feature_coverage": self.feature_coverage,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }
