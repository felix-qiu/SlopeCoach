"""Provisional A6 sport-resolution contracts for Python research/reference use."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum

SPORT_TYPE_CONTRACT_VERSION = "sport-type-v1"
SPORT_TYPE_REQUIRED_REASON = "SPORT_TYPE_UNKNOWN"
SPORT_TYPE_CONFIG_PROFILE = "RESEARCH_DEFAULTS_A6"


class SportType(StrEnum):
    SKI = "SKI"
    SNOWBOARD = "SNOWBOARD"
    UNKNOWN = "UNKNOWN"


class SportTypeSource(StrEnum):
    AUTO = "AUTO"
    USER = "USER"


class SportTypeResolutionStatus(StrEnum):
    RESOLVED_AUTO = "RESOLVED_AUTO"
    RESOLVED_USER = "RESOLVED_USER"
    INSUFFICIENT_PRIMARY_EVIDENCE = "INSUFFICIENT_PRIMARY_EVIDENCE"
    INSUFFICIENT_TOTAL_EVIDENCE = "INSUFFICIENT_TOTAL_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTING_PRIMARY_EVIDENCE = "CONFLICTING_PRIMARY_EVIDENCE"


class SportEvidenceKind(StrEnum):
    EQUIPMENT = "EQUIPMENT"
    VISUAL_CLASSIFIER = "VISUAL_CLASSIFIER"
    POSE_GEOMETRY = "POSE_GEOMETRY"
    TEMPORAL_MOTION = "TEMPORAL_MOTION"

    @property
    def is_primary(self) -> bool:
        return self in {self.EQUIPMENT, self.VISUAL_CLASSIFIER}


class SportEvidenceScope(StrEnum):
    FRAME = "FRAME"
    CLIP = "CLIP"


class SportEvidenceProviderStatus(StrEnum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    CONFIGURED = "CONFIGURED"
    EXECUTED_NO_EVIDENCE = "EXECUTED_NO_EVIDENCE"
    EXECUTED_WITH_EVIDENCE = "EXECUTED_WITH_EVIDENCE"
    FAILED = "FAILED"


class SportCueStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass(frozen=True)
class SportTypeConfig:
    equipment_weight: float = 1.0
    visual_classifier_weight: float = 0.80
    pose_geometry_weight: float = 0.25
    temporal_motion_weight: float = 0.25
    minimum_frame_observations_per_kind: int = 2
    minimum_primary_support: float = 0.70
    minimum_auto_support: float = 0.65
    minimum_auto_margin: float = 0.20
    primary_conflict_support: float = 0.75

    def __post_init__(self) -> None:
        for name in (
            "equipment_weight",
            "visual_classifier_weight",
            "pose_geometry_weight",
            "temporal_motion_weight",
        ):
            _finite_number(getattr(self, name), name, minimum=0.0, maximum=None, positive=True)
        if (
            isinstance(self.minimum_frame_observations_per_kind, bool)
            or not isinstance(self.minimum_frame_observations_per_kind, int)
            or self.minimum_frame_observations_per_kind < 1
        ):
            raise ValueError("minimum_frame_observations_per_kind must be a positive integer")
        for name in (
            "minimum_primary_support",
            "minimum_auto_support",
            "minimum_auto_margin",
            "primary_conflict_support",
        ):
            _finite_number(getattr(self, name), name, minimum=0.0, maximum=1.0)

    def weight_for(self, kind: SportEvidenceKind) -> float:
        return {
            SportEvidenceKind.EQUIPMENT: self.equipment_weight,
            SportEvidenceKind.VISUAL_CLASSIFIER: self.visual_classifier_weight,
            SportEvidenceKind.POSE_GEOMETRY: self.pose_geometry_weight,
            SportEvidenceKind.TEMPORAL_MOTION: self.temporal_motion_weight,
        }[kind]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SportEvidenceObservation:
    evidence_id: str
    kind: SportEvidenceKind
    provider_name: str
    ski_support: float
    snowboard_support: float
    quality: float
    scope: SportEvidenceScope
    timestamp_us: int | None = None
    temporal_segment_id: int | None = None
    reason: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.evidence_id, "evidence_id")
        _nonempty(self.provider_name, "provider_name")
        for name in ("ski_support", "snowboard_support", "quality"):
            _finite_number(getattr(self, name), name, minimum=0.0, maximum=1.0)
        if self.scope is SportEvidenceScope.FRAME and self.timestamp_us is None:
            raise ValueError("FRAME evidence requires timestamp_us")
        _optional_integer(self.timestamp_us, "timestamp_us", minimum=0)
        _optional_integer(self.temporal_segment_id, "temporal_segment_id", minimum=1)
        if self.reason is not None:
            _nonempty(self.reason, "reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "provider_name": self.provider_name,
            "timestamp_us": self.timestamp_us,
            "temporal_segment_id": self.temporal_segment_id,
            "ski_support": self.ski_support,
            "snowboard_support": self.snowboard_support,
            "quality": self.quality,
            "scope": self.scope.value,
            "reason": self.reason,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class SportEvidenceProviderResult:
    provider_name: str
    evidence_kind: SportEvidenceKind
    status: SportEvidenceProviderStatus
    observations: tuple[SportEvidenceObservation, ...] = ()
    error: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.provider_name, "provider_name")
        if any(item.kind is not self.evidence_kind for item in self.observations):
            raise ValueError("provider result observation kind mismatch")
        if any(item.provider_name != self.provider_name for item in self.observations):
            raise ValueError("provider result observation provider_name mismatch")
        if self.status is SportEvidenceProviderStatus.EXECUTED_WITH_EVIDENCE:
            if not self.observations:
                raise ValueError("EXECUTED_WITH_EVIDENCE requires observations")
        elif self.observations:
            raise ValueError(f"{self.status.value} requires empty observations")
        if self.status is SportEvidenceProviderStatus.FAILED:
            if self.error is None:
                raise ValueError("FAILED provider result requires an error")
            _nonempty(self.error, "error")
        elif self.error is not None:
            raise ValueError("only FAILED provider results may contain an error")

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_name": self.provider_name,
            "evidence_kind": self.evidence_kind.value,
            "status": self.status.value,
            "observations": [item.to_dict() for item in self.observations],
            "error": self.error,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class SportCueMeasurement:
    cue_id: str
    value: float | None
    unit: str
    status: SportCueStatus
    source_feature_ids: tuple[str, ...]
    limitations: tuple[str, ...]
    contributes_to_auto_fusion: bool = False

    def __post_init__(self) -> None:
        _nonempty(self.cue_id, "cue_id")
        _nonempty(self.unit, "unit")
        if self.contributes_to_auto_fusion:
            raise ValueError("A6 uncalibrated cues must not contribute to auto fusion")
        if self.status is SportCueStatus.AVAILABLE:
            if self.value is None:
                raise ValueError("available cue requires a value")
            _finite_number(self.value, "value", minimum=None, maximum=None)
        elif self.value is not None:
            raise ValueError("unavailable cue value must be null")
        if not self.source_feature_ids or any(not item.strip() for item in self.source_feature_ids):
            raise ValueError("source_feature_ids must contain non-empty IDs")

    def to_dict(self) -> dict[str, object]:
        return {
            "cue_id": self.cue_id,
            "value": self.value,
            "unit": self.unit,
            "status": self.status.value,
            "source_feature_ids": list(self.source_feature_ids),
            "limitations": list(self.limitations),
            "contributes_to_auto_fusion": self.contributes_to_auto_fusion,
        }


@dataclass(frozen=True)
class AutoSportTypeDecision:
    sport_type: SportType
    status: SportTypeResolutionStatus
    ski_support: float | None
    snowboard_support: float | None
    margin: float | None
    active_evidence_kinds: tuple[SportEvidenceKind, ...]
    primary_evidence_kinds: tuple[SportEvidenceKind, ...]
    evidence_observation_count: int
    ask_user_recommended: bool
    reason_codes: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("ski_support", "snowboard_support", "margin"):
            value = getattr(self, name)
            if value is not None:
                _finite_number(value, name, minimum=0.0, maximum=1.0)
        if (
            isinstance(self.evidence_observation_count, bool)
            or not isinstance(self.evidence_observation_count, int)
            or self.evidence_observation_count < 0
        ):
            raise ValueError("evidence_observation_count must be non-negative")
        if len(self.active_evidence_kinds) != len(set(self.active_evidence_kinds)):
            raise ValueError("active_evidence_kinds must be unique")
        if len(self.primary_evidence_kinds) != len(set(self.primary_evidence_kinds)):
            raise ValueError("primary_evidence_kinds must be unique")
        if any(
            not item.is_primary or item not in self.active_evidence_kinds
            for item in self.primary_evidence_kinds
        ):
            raise ValueError("primary_evidence_kinds must be active primary kinds")
        support_values = (self.ski_support, self.snowboard_support, self.margin)
        if any(value is None for value in support_values) and not all(
            value is None for value in support_values
        ):
            raise ValueError("auto support values must be all null or all non-null")
        if self.status is SportTypeResolutionStatus.RESOLVED_AUTO:
            if self.sport_type is SportType.UNKNOWN or self.ask_user_recommended:
                raise ValueError("resolved auto decision must select a sport without asking user")
        elif self.sport_type is not SportType.UNKNOWN or not self.ask_user_recommended:
            raise ValueError("unresolved auto decision must be UNKNOWN and recommend user input")

    def to_dict(self) -> dict[str, object]:
        return {
            "sport_type": self.sport_type.value,
            "status": self.status.value,
            "ski_support": self.ski_support,
            "snowboard_support": self.snowboard_support,
            "margin": self.margin,
            "active_evidence_kinds": [item.value for item in self.active_evidence_kinds],
            "primary_evidence_kinds": [item.value for item in self.primary_evidence_kinds],
            "evidence_observation_count": self.evidence_observation_count,
            "ask_user_recommended": self.ask_user_recommended,
            "reason_codes": list(self.reason_codes),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class SportTypeResult:
    effective_sport_type: SportType
    effective_source: SportTypeSource
    resolution_status: SportTypeResolutionStatus
    auto_decision: AutoSportTypeDecision
    user_selection: SportType | None
    auto_user_disagreement: bool
    provider_results: tuple[SportEvidenceProviderResult, ...]
    cue_measurements: tuple[SportCueMeasurement, ...]
    ask_user_recommended: bool
    config: SportTypeConfig
    limitations: tuple[str, ...] = field(
        default=(
            "SUPPORT_VALUES_ARE_NOT_CALIBRATED_PROBABILITIES",
            "UNCALIBRATED_CUES_DO_NOT_CONTRIBUTE_TO_AUTO_FUSION",
            "PYTHON_RESEARCH_REFERENCE_ONLY",
        )
    )
    contract_version: str = SPORT_TYPE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != SPORT_TYPE_CONTRACT_VERSION:
            raise ValueError(f"contract_version must be {SPORT_TYPE_CONTRACT_VERSION}")
        if self.user_selection is SportType.UNKNOWN:
            raise ValueError("explicit user selection must be SKI or SNOWBOARD")
        if self.effective_source is SportTypeSource.USER:
            if self.user_selection is None or self.effective_sport_type is not self.user_selection:
                raise ValueError("USER result must use the explicit user selection")
            if self.resolution_status is not SportTypeResolutionStatus.RESOLVED_USER:
                raise ValueError("USER result must have RESOLVED_USER status")
            if self.ask_user_recommended:
                raise ValueError("resolved user result must not ask user")
            expected_disagreement = (
                self.auto_decision.sport_type is not SportType.UNKNOWN
                and self.auto_decision.sport_type is not self.user_selection
            )
            if self.auto_user_disagreement is not expected_disagreement:
                raise ValueError("auto_user_disagreement is inconsistent with auto/user values")
        elif self.user_selection is not None:
            raise ValueError("AUTO result cannot retain a user selection")
        elif (
            self.effective_sport_type is not self.auto_decision.sport_type
            or self.resolution_status is not self.auto_decision.status
            or self.ask_user_recommended is not self.auto_decision.ask_user_recommended
            or self.auto_user_disagreement
        ):
            raise ValueError("AUTO result must preserve the auto decision exactly")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "effective_sport_type": self.effective_sport_type.value,
            "effective_source": self.effective_source.value,
            "resolution_status": self.resolution_status.value,
            "auto_decision": self.auto_decision.to_dict(),
            "user_selection": self.user_selection.value if self.user_selection else None,
            "auto_user_disagreement": self.auto_user_disagreement,
            "provider_results": [item.to_dict() for item in self.provider_results],
            "cue_measurements": [item.to_dict() for item in self.cue_measurements],
            "ask_user_recommended": self.ask_user_recommended,
            "config": {"profile": SPORT_TYPE_CONFIG_PROFILE, **self.config.to_dict()},
            "limitations": list(self.limitations),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False, indent=indent)


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _finite_number(
    value: float, name: str, *, minimum: float | None, maximum: float | None, positive=False
) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _optional_integer(value: int | None, name: str, *, minimum: int) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < minimum
    ):
        raise ValueError(f"{name} must be null or an integer >= {minimum}")
