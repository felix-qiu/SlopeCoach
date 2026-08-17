"""A6.3 provisional SportType calibration contracts.

These contracts are Python research/reference models, not production Domain Kernel
contracts. RAW_V1 remains the effective routing implementation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from ..contracts import SportEvidenceKind

SPORT_EVIDENCE_CALIBRATION_CONTRACT_VERSION = "sport-evidence-calibration-v1"
SPORT_TYPE_GT_CONTRACT_VERSION = "sport-type-gt-v1"
SPORT_TYPE_CALIBRATION_DATASET_VERSION = "sport-type-calibration-dataset-v1"
CALIBRATED_FUSION_VERSION = "calibrated-sport-fusion-v1"
SPORT_TYPE_BENCHMARK_CONTRACT_VERSION = "ski-bench-sport-type-v4"
CALIBRATION_PROFILE = "RESEARCH_DEFAULTS_A6_3"
CALIBRATED_FUSION_CONTROLS_ROUTING = False


class GroundTruthSportType(StrEnum):
    SKI = "SKI"
    SNOWBOARD = "SNOWBOARD"
    UNLABELED = "UNLABELED"
    UNCERTAIN = "UNCERTAIN"


class AnnotationSource(StrEnum):
    USER_MANUAL = "USER_MANUAL"


class IntendedTargetConfirmation(StrEnum):
    CONFIRMED = "CONFIRMED"
    UNCONFIRMED = "UNCONFIRMED"
    UNCERTAIN = "UNCERTAIN"


class CalibrationChannelStatus(StrEnum):
    ACCEPTED_RESEARCH_CALIBRATION = "ACCEPTED_RESEARCH_CALIBRATION"
    REJECTED_INSUFFICIENT_DATA = "REJECTED_INSUFFICIENT_DATA"
    REJECTED_NON_MONOTONIC_CHANNEL = "REJECTED_NON_MONOTONIC_CHANNEL"
    REJECTED_NO_BRIER_IMPROVEMENT = "REJECTED_NO_BRIER_IMPROVEMENT"
    REJECTED_NO_LOG_LOSS_IMPROVEMENT = "REJECTED_NO_LOG_LOSS_IMPROVEMENT"
    REJECTED_FIT_FAILURE = "REJECTED_FIT_FAILURE"
    CALIBRATION_CHANNEL_PROVENANCE_MISMATCH = "CALIBRATION_CHANNEL_PROVENANCE_MISMATCH"


class CalibratedFusionStatus(StrEnum):
    NOT_AVAILABLE_NO_CALIBRATION_ARTIFACT = "NOT_AVAILABLE_NO_CALIBRATION_ARTIFACT"
    NOT_AVAILABLE_NO_VALID_CALIBRATION_ARTIFACT = "NOT_AVAILABLE_NO_VALID_CALIBRATION_ARTIFACT"
    CALIBRATION_ARTIFACT_INCOMPATIBLE = "CALIBRATION_ARTIFACT_INCOMPATIBLE"
    NOT_AVAILABLE_INSUFFICIENT_CALIBRATED_CHANNELS = (
        "NOT_AVAILABLE_INSUFFICIENT_CALIBRATED_CHANNELS"
    )
    AVAILABLE_SINGLE_PRIMARY_KIND = "AVAILABLE_SINGLE_PRIMARY_KIND"
    AVAILABLE_MULTIPLE_PRIMARY_KINDS = "AVAILABLE_MULTIPLE_PRIMARY_KINDS"
    CONFLICTING_CALIBRATED_PRIMARY_EVIDENCE = "CONFLICTING_CALIBRATED_PRIMARY_EVIDENCE"


class AgreementState(StrEnum):
    NO_CALIBRATED_EVIDENCE = "NO_CALIBRATED_EVIDENCE"
    SINGLE_KIND_SKI = "SINGLE_KIND_SKI"
    SINGLE_KIND_SNOWBOARD = "SINGLE_KIND_SNOWBOARD"
    AGREE_SKI = "AGREE_SKI"
    AGREE_SNOWBOARD = "AGREE_SNOWBOARD"
    CONFLICT = "CONFLICT"
    WEAK_OR_MIXED = "WEAK_OR_MIXED"


@dataclass(frozen=True)
class SportCalibrationFitConfig:
    minimum_labeled_sources_per_class: int = 10
    preferred_labeled_sources_per_class: int = 20
    cross_validation_folds: int = 5
    l2_regularization: float = 1.0
    maximum_newton_iterations: int = 100
    convergence_tolerance: float = 1e-10
    probability_epsilon: float = 1e-6
    fusion_prior_snowboard: float = 0.5
    calibrated_conflict_probability: float = 0.80
    profile: str = CALIBRATION_PROFILE

    def __post_init__(self) -> None:
        for name in (
            "minimum_labeled_sources_per_class",
            "preferred_labeled_sources_per_class",
            "cross_validation_folds",
            "maximum_newton_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a non-bool integer")
        if self.minimum_labeled_sources_per_class < 1:
            raise ValueError("minimum_labeled_sources_per_class must be positive")
        if self.preferred_labeled_sources_per_class < self.minimum_labeled_sources_per_class:
            raise ValueError("preferred labeled source count cannot be below minimum")
        if self.cross_validation_folds < 3:
            raise ValueError("cross_validation_folds must be at least 3")
        if self.maximum_newton_iterations < 1:
            raise ValueError("maximum_newton_iterations must be positive")
        for name in (
            "l2_regularization",
            "convergence_tolerance",
            "probability_epsilon",
            "fusion_prior_snowboard",
            "calibrated_conflict_probability",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite numeric")
        if self.l2_regularization < 0 or self.convergence_tolerance <= 0:
            raise ValueError("regularization must be non-negative and tolerance positive")
        if not 0 < self.probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must be in (0, 0.5)")
        if not 0 < self.fusion_prior_snowboard < 1:
            raise ValueError("fusion prior must be in (0, 1)")
        if not 0.5 < self.calibrated_conflict_probability < 1:
            raise ValueError("conflict probability must be in (0.5, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SportTypeGroundTruth:
    clip_id: str
    source_video_id: str
    video_sha256: str
    target_sport_type: GroundTruthSportType = GroundTruthSportType.UNLABELED
    annotation_source: AnnotationSource = AnnotationSource.USER_MANUAL
    intended_target_confirmation: IntendedTargetConfirmation = (
        IntendedTargetConfirmation.UNCONFIRMED
    )
    notes: str = ""
    contract_version: str = SPORT_TYPE_GT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != SPORT_TYPE_GT_CONTRACT_VERSION:
            raise ValueError("unsupported SportType GT contract")
        if not self.clip_id.strip() or not self.source_video_id.strip():
            raise ValueError("clip_id and source_video_id must be non-empty")
        if len(self.video_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.video_sha256
        ):
            raise ValueError("video_sha256 must be a lowercase SHA256")

    @property
    def eligible_for_fitting(self) -> bool:
        return (
            self.target_sport_type in {GroundTruthSportType.SKI, GroundTruthSportType.SNOWBOARD}
            and self.annotation_source is AnnotationSource.USER_MANUAL
            and self.intended_target_confirmation is IntendedTargetConfirmation.CONFIRMED
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "clip_id": self.clip_id,
            "source_video_id": self.source_video_id,
            "video_sha256": self.video_sha256,
            "target_sport_type": self.target_sport_type.value,
            "annotation_source": self.annotation_source.value,
            "intended_target_confirmation": self.intended_target_confirmation.value,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> SportTypeGroundTruth:
        try:
            return cls(
                clip_id=str(value["clip_id"]),
                source_video_id=str(value["source_video_id"]),
                video_sha256=str(value["video_sha256"]),
                target_sport_type=GroundTruthSportType(str(value["target_sport_type"])),
                annotation_source=AnnotationSource(str(value["annotation_source"])),
                intended_target_confirmation=IntendedTargetConfirmation(
                    str(value["intended_target_confirmation"])
                ),
                notes=str(value.get("notes", "")),
                contract_version=str(value["contract_version"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("malformed SportType GT") from error


@dataclass(frozen=True)
class RawProviderSportEvidenceSummary:
    calibration_channel_id: str
    provider_name: str
    evidence_kind: SportEvidenceKind
    source_video_id: str
    video_sha256: str
    observation_count: int
    distinct_timestamp_count: int
    raw_ski_support: float | None
    raw_snowboard_support: float | None
    raw_direction: float | None
    raw_margin: float | None
    clip_count_per_source: int = 1
    source_status: str = "AVAILABLE"
    limitations: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = f"{self.evidence_kind.value}::{self.provider_name}"
        if self.calibration_channel_id != expected:
            raise ValueError("calibration_channel_id must be KIND::provider_name")
        if self.observation_count < 0 or self.distinct_timestamp_count < 0:
            raise ValueError("observation counts must be non-negative")
        values = (
            self.raw_ski_support,
            self.raw_snowboard_support,
            self.raw_direction,
            self.raw_margin,
        )
        if any(item is None for item in values) and not all(item is None for item in values):
            raise ValueError("raw summary values must be all null or all present")
        if self.raw_direction is not None:
            if not all(math.isfinite(item) for item in values):
                raise ValueError("raw summary values must be finite")
            if not -1 <= self.raw_direction <= 1:
                raise ValueError("raw_direction must be in [-1, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_channel_id": self.calibration_channel_id,
            "provider_name": self.provider_name,
            "evidence_kind": self.evidence_kind.value,
            "source_video_id": self.source_video_id,
            "video_sha256": self.video_sha256,
            "observation_count": self.observation_count,
            "distinct_timestamp_count": self.distinct_timestamp_count,
            "clip_count_per_source": self.clip_count_per_source,
            "raw_ski_support": self.raw_ski_support,
            "raw_snowboard_support": self.raw_snowboard_support,
            "raw_direction": self.raw_direction,
            "raw_margin": self.raw_margin,
            "source_status": self.source_status,
            "limitations": list(self.limitations),
            "provenance": self.provenance,
        }


def strict_json(payload: object, *, indent: int | None = None) -> str:
    return json.dumps(payload, sort_keys=True, allow_nan=False, indent=indent)
