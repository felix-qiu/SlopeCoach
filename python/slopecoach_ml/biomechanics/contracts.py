"""Provisional A5 image-space biomechanics research contracts."""

from __future__ import annotations

import json
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
    boundary_match_tolerance_us: int = 150_000

    def validate(self) -> None:
        if (
            isinstance(self.minimum_joint_support_confidence, bool)
            or not isinstance(self.minimum_joint_support_confidence, int | float)
            or not math.isfinite(self.minimum_joint_support_confidence)
            or not 0 <= self.minimum_joint_support_confidence <= 1
        ):
            raise ValueError("minimum_joint_support_confidence must be in [0, 1]")
        if (
            isinstance(self.square_pixel_tolerance, bool)
            or not isinstance(self.square_pixel_tolerance, int | float)
            or not math.isfinite(self.square_pixel_tolerance)
            or self.square_pixel_tolerance < 0
        ):
            raise ValueError("square_pixel_tolerance must be finite and non-negative")
        for name in (
            "minimum_aggregate_samples",
            "minimum_derivative_dt_us",
            "apex_match_tolerance_us",
            "boundary_match_tolerance_us",
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
        if not isinstance(self.feature_id, str) or not self.feature_id.strip():
            raise ValueError("feature_id must be a non-empty string")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("unit must be a non-empty string")
        if self.status is BiomechanicsFactStatus.AVAILABLE and self.value is None:
            raise ValueError("available biomechanics fact value must be non-null")
        if self.status is not BiomechanicsFactStatus.AVAILABLE and self.value is not None:
            raise ValueError("unavailable biomechanics fact value must be null")
        if self.value is not None and (
            isinstance(self.value, bool)
            or not isinstance(self.value, int | float)
            or not math.isfinite(self.value)
        ):
            raise ValueError("biomechanics fact value must be finite")
        _optional_integer(self.timestamp_us, "timestamp_us", minimum=0)
        _optional_integer(self.temporal_segment_id, "temporal_segment_id", minimum=1)
        _optional_integer(self.signal_run_id, "signal_run_id", minimum=1)
        if self.turn_id is not None and (
            not isinstance(self.turn_id, str) or not self.turn_id.strip()
        ):
            raise ValueError("turn_id must be null or a non-empty string")
        _optional_ratio(self.support_confidence, "support_confidence")
        _count(self.observed_joint_count, "observed_joint_count")
        _count(self.interpolated_joint_count, "interpolated_joint_count")
        if self.required_joints and (
            self.observed_joint_count + self.interpolated_joint_count > len(self.required_joints)
        ):
            raise ValueError("joint evidence counts exceed unique required joints")

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

    def __post_init__(self) -> None:
        if not isinstance(self.feature_id, str) or not self.feature_id.strip():
            raise ValueError("feature_id must be a non-empty string")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("unit must be a non-empty string")
        _required_integer(self.temporal_segment_id, "temporal_segment_id", minimum=1)
        for name in (
            "sample_count",
            "available_count",
            "observed_only_sample_count",
            "interpolated_support_sample_count",
        ):
            _count(getattr(self, name), name)
        if self.available_count > self.sample_count:
            raise ValueError("available_count must not exceed sample_count")
        if (
            self.observed_only_sample_count + self.interpolated_support_sample_count
            > self.available_count
        ):
            raise ValueError("aggregate evidence counts must not exceed available_count")
        for name in ("support_ratio", "observed_only_ratio", "interpolated_support_ratio"):
            _ratio(getattr(self, name), name)
        denominator = self.sample_count
        expected_ratios = {
            "support_ratio": self.available_count / denominator if denominator else 0.0,
            "observed_only_ratio": (
                self.observed_only_sample_count / denominator if denominator else 0.0
            ),
            "interpolated_support_ratio": (
                self.interpolated_support_sample_count / denominator if denominator else 0.0
            ),
        }
        for name, expected in expected_ratios.items():
            if not math.isclose(getattr(self, name), expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{name} is inconsistent with aggregate counts")
        _optional_ratio(self.median_support_confidence, "median_support_confidence")
        statistics = (self.median, self.minimum, self.maximum, self.range)
        if self.status is BiomechanicsFactStatus.AVAILABLE:
            if any(value is None for value in statistics):
                raise ValueError("available aggregate statistics must be non-null")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                for value in statistics
            ):
                raise ValueError("available aggregate statistics must be finite")
            if not self.minimum <= self.median <= self.maximum:
                raise ValueError("aggregate statistics must satisfy minimum <= median <= maximum")
            if self.range < 0:
                raise ValueError("aggregate range must be non-negative")
            if not math.isclose(
                self.range,
                self.maximum - self.minimum,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("aggregate range must equal maximum - minimum")
        elif self.status is BiomechanicsFactStatus.INSUFFICIENT_SAMPLES:
            if any(value is not None for value in statistics):
                raise ValueError("insufficient aggregate statistics must be null")

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

    def __post_init__(self) -> None:
        if not isinstance(self.turn_id, str) or not self.turn_id.strip():
            raise ValueError("turn_id must be a non-empty string")
        _required_integer(self.temporal_segment_id, "temporal_segment_id", minimum=1)
        _required_integer(self.signal_run_id, "signal_run_id", minimum=1)
        feature_ids = []
        for fact in self.facts:
            if fact.scope is not BiomechanicsFactScope.TURN:
                raise ValueError("turn result may contain only TURN facts")
            if fact.turn_id != self.turn_id:
                raise ValueError("turn fact turn_id does not match result")
            if fact.temporal_segment_id != self.temporal_segment_id:
                raise ValueError("turn fact temporal_segment_id does not match result")
            if fact.signal_run_id != self.signal_run_id:
                raise ValueError("turn fact signal_run_id does not match result")
            feature_ids.append(fact.feature_id)
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("turn result feature IDs must be unique")

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
    feature_schema_version: str
    feature_registry_sha256: str
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

    def __post_init__(self) -> None:
        self.config.validate()
        if not isinstance(self.contract_version, str) or not self.contract_version.strip():
            raise ValueError("contract_version must be a non-empty string")
        if (
            not isinstance(self.feature_schema_version, str)
            or not self.feature_schema_version.strip()
        ):
            raise ValueError("feature_schema_version must be a non-empty string")
        if (
            not isinstance(self.feature_registry_sha256, str)
            or len(self.feature_registry_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.feature_registry_sha256
            )
        ):
            raise ValueError("feature_registry_sha256 must be a lowercase SHA256 hex digest")
        seen = set()
        for fact in self.frame_facts:
            if fact.timestamp_us is None:
                continue
            key = (fact.timestamp_us, fact.temporal_segment_id, fact.feature_id)
            if key in seen:
                raise ValueError(f"duplicate frame biomechanics fact: {key}")
            seen.add(key)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "feature_schema_version": self.feature_schema_version,
            "feature_registry_sha256": self.feature_registry_sha256,
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

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False, indent=indent)


def _optional_integer(value: int | None, name: str, *, minimum: int) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < minimum
    ):
        raise ValueError(f"{name} must be null or an integer >= {minimum}")


def _required_integer(value: int, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _ratio(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _optional_ratio(value: float | None, name: str) -> None:
    if value is not None:
        _ratio(value, name)
