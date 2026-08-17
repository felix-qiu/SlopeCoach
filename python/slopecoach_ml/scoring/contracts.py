"""Provisional A8 scorecard contracts for research and future Rust parity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum

SCORECARD_CONTRACT_VERSION = "scorecard-v1"
SCORING_POLICY_VERSION = "scoring-policy-v1"
DIAGNOSIS_DIMENSION_REGISTRY_VERSION = "diagnosis-dimension-registry-v1"
ISSUE_PRIORITY_POLICY_VERSION = "issue-priority-v1"


class ScoreDimension(StrEnum):
    BALANCE = "BALANCE"
    EDGE_CONTROL = "EDGE_CONTROL"
    STANCE = "STANCE"
    SYMMETRY = "SYMMETRY"
    TIMING = "TIMING"


class DimensionAssessmentStatus(StrEnum):
    PROVISIONAL_ISSUE_DETECTED = "PROVISIONAL_ISSUE_DETECTED"
    NO_PROVISIONAL_ISSUE_DETECTED = "NO_PROVISIONAL_ISSUE_DETECTED"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


@dataclass(frozen=True)
class ScoringPolicy:
    profile: str = "STRUCTURE_ONLY_NO_NUMERIC_SCORE_A8"
    numeric_scoring_enabled: bool = False
    version: str = SCORING_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.numeric_scoring_enabled:
            raise ValueError("A8 numeric scoring must remain disabled")


@dataclass(frozen=True)
class IssuePriorityPolicy:
    profile: str = "EVIDENCE_RECURRENCE_PRIORITY_A8"
    max_top_issues: int = 2
    version: str = ISSUE_PRIORITY_POLICY_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_top_issues, bool)
            or not isinstance(self.max_top_issues, int)
            or not 1 <= self.max_top_issues <= 2
        ):
            raise ValueError("max_top_issues must be a positive non-bool integer <= 2")


@dataclass(frozen=True)
class DimensionAssessment:
    dimension: ScoreDimension
    status: DimensionAssessmentStatus
    mapped_diagnosis_codes: tuple[str, ...]
    rule_count: int
    evaluable_rule_turn_count: int
    triggered_rule_turn_count: int
    not_triggered_rule_turn_count: int
    not_evaluable_rule_turn_count: int
    triggered_turn_ratio: float | None
    evidence_references: tuple[dict[str, object], ...]
    limitations: tuple[str, ...]
    reason: str | None = None
    score_value: None = None
    score_scale_min: None = None
    score_scale_max: None = None
    score_validation_status: str = "NOT_CALIBRATED_GT_REQUIRED"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dimension"] = self.dimension.value
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class ScoreCard:
    dimensions: tuple[DimensionAssessment, ...]
    diagnosis_contract_version: str
    diagnosis_rule_registry_sha256: str
    diagnosis_dimension_registry_sha256: str
    limitations: tuple[str, ...]
    contract_version: str = SCORECARD_CONTRACT_VERSION
    scoring_policy_version: str = SCORING_POLICY_VERSION
    numeric_scoring_enabled: bool = False
    overall_score: None = None
    overall_score_status: str = "NOT_CALIBRATED_GT_REQUIRED"

    def __post_init__(self) -> None:
        if self.numeric_scoring_enabled or self.overall_score is not None:
            raise ValueError("A8 ScoreCard cannot contain numeric scoring")
        if any(item.score_value is not None for item in self.dimensions):
            raise ValueError("A8 dimension score values must be null")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "diagnosis_contract_version": self.diagnosis_contract_version,
            "diagnosis_rule_registry_sha256": self.diagnosis_rule_registry_sha256,
            "diagnosis_dimension_registry_sha256": self.diagnosis_dimension_registry_sha256,
            "scoring_policy_version": self.scoring_policy_version,
            "numeric_scoring_enabled": self.numeric_scoring_enabled,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "overall_score": self.overall_score,
            "overall_score_status": self.overall_score_status,
            "OVERALL_SCORE_STATUS": self.overall_score_status,
            "SCORE_GT_STATUS": "NOT_AVAILABLE",
            "SCORE_CALIBRATION_STATUS": "DEFERRED_GT_REQUIRED",
            "limitations": list(self.limitations),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False, indent=indent)
