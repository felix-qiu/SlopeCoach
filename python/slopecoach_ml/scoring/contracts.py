"""Provisional A8 scorecard contracts for research and future Rust parity."""

from __future__ import annotations

import hashlib
import json
import math
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
        if self.version != ISSUE_PRIORITY_POLICY_VERSION or self.profile != (
            "EVIDENCE_RECURRENCE_PRIORITY_A8"
        ):
            raise ValueError("A8 issue priority policy version and profile are fixed")
        if (
            isinstance(self.max_top_issues, bool)
            or not isinstance(self.max_top_issues, int)
            or not 1 <= self.max_top_issues <= 2
        ):
            raise ValueError("max_top_issues must be a positive non-bool integer <= 2")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "profile": self.profile,
            "max_top_issues": self.max_top_issues,
        }


def canonical_issue_priority_policy_json(policy: IssuePriorityPolicy) -> str:
    return json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def issue_priority_policy_sha256(policy: IssuePriorityPolicy) -> str:
    return hashlib.sha256(canonical_issue_priority_policy_json(policy).encode()).hexdigest()


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

    def __post_init__(self) -> None:
        if any(
            value is not None
            for value in (self.score_value, self.score_scale_min, self.score_scale_max)
        ):
            raise ValueError("A8 dimension scores and scales must be null")
        if self.score_validation_status != "NOT_CALIBRATED_GT_REQUIRED":
            raise ValueError("A8 dimension score validation status is fixed")
        count_names = (
            "rule_count",
            "evaluable_rule_turn_count",
            "triggered_rule_turn_count",
            "not_triggered_rule_turn_count",
            "not_evaluable_rule_turn_count",
        )
        for name in count_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative non-bool integer")
        if self.evaluable_rule_turn_count != (
            self.triggered_rule_turn_count + self.not_triggered_rule_turn_count
        ):
            raise ValueError("dimension evaluable count arithmetic is inconsistent")
        if self.evaluable_rule_turn_count == 0:
            if self.triggered_turn_ratio is not None:
                raise ValueError("zero evaluable turns require a null trigger ratio")
        else:
            ratio = self.triggered_turn_ratio
            expected = self.triggered_rule_turn_count / self.evaluable_rule_turn_count
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, int | float)
                or not math.isfinite(ratio)
                or not 0 <= ratio <= 1
                or not math.isclose(ratio, expected, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError("dimension trigger ratio is inconsistent")
        if self.status is DimensionAssessmentStatus.NOT_IMPLEMENTED:
            if (
                self.rule_count != 0
                or self.mapped_diagnosis_codes
                or any(
                    (
                        self.evaluable_rule_turn_count,
                        self.triggered_rule_turn_count,
                        self.not_triggered_rule_turn_count,
                        self.not_evaluable_rule_turn_count,
                    )
                )
                or self.triggered_turn_ratio is not None
            ):
                raise ValueError("NOT_IMPLEMENTED dimension contains rule evidence")
        else:
            if self.rule_count == 0 or not self.mapped_diagnosis_codes:
                raise ValueError("implemented dimension requires mapped rules")
            if self.rule_count != len(set(self.mapped_diagnosis_codes)):
                raise ValueError("dimension rule_count must match unique mapped diagnosis codes")
            expected_status = (
                DimensionAssessmentStatus.NOT_EVALUABLE
                if self.evaluable_rule_turn_count == 0
                else DimensionAssessmentStatus.PROVISIONAL_ISSUE_DETECTED
                if self.triggered_rule_turn_count > 0
                else DimensionAssessmentStatus.NO_PROVISIONAL_ISSUE_DETECTED
            )
            if self.status is not expected_status:
                raise ValueError("dimension status is inconsistent with evaluation counts")

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
    diagnosis_config_sha256: str
    diagnosis_semantics_sha256: str
    diagnosis_semantics_provenance_version: str
    diagnosis_semantics_provenance: dict[str, object]
    limitations: tuple[str, ...]
    contract_version: str = SCORECARD_CONTRACT_VERSION
    scoring_policy_version: str = SCORING_POLICY_VERSION
    numeric_scoring_enabled: bool = False
    overall_score: None = None
    overall_score_status: str = "NOT_CALIBRATED_GT_REQUIRED"

    def __post_init__(self) -> None:
        from .registry import DIAGNOSIS_DIMENSION_REGISTRY_SHA256

        if self.contract_version != SCORECARD_CONTRACT_VERSION:
            raise ValueError("ScoreCard contract version is incompatible")
        if self.scoring_policy_version != SCORING_POLICY_VERSION:
            raise ValueError("ScoreCard scoring policy version is incompatible")
        if self.diagnosis_dimension_registry_sha256 != DIAGNOSIS_DIMENSION_REGISTRY_SHA256:
            raise ValueError("ScoreCard diagnosis dimension registry is incompatible")
        if (
            self.numeric_scoring_enabled is not False
            or self.overall_score is not None
            or self.overall_score_status != "NOT_CALIBRATED_GT_REQUIRED"
        ):
            raise ValueError("A8 ScoreCard cannot contain numeric scoring")
        expected = set(ScoreDimension)
        actual = [item.dimension for item in self.dimensions]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("ScoreCard requires exactly the five canonical dimensions")
        if any(
            item.score_value is not None
            or item.score_scale_min is not None
            or item.score_scale_max is not None
            for item in self.dimensions
        ):
            raise ValueError("A8 dimension score values must be null")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.diagnosis_contract_version,
                self.diagnosis_rule_registry_sha256,
                self.diagnosis_config_sha256,
                self.diagnosis_semantics_sha256,
                self.diagnosis_semantics_provenance_version,
            )
        ):
            raise ValueError("ScoreCard diagnosis semantics provenance is required")
        from slopecoach_ml.diagnosis import validate_diagnosis_semantics_provenance

        provenance = validate_diagnosis_semantics_provenance(self.diagnosis_semantics_provenance)
        if (
            provenance.version != self.diagnosis_semantics_provenance_version
            or provenance.diagnosis_contract_version != self.diagnosis_contract_version
            or provenance.diagnosis_rule_registry_sha256 != self.diagnosis_rule_registry_sha256
            or provenance.diagnosis_config_sha256 != self.diagnosis_config_sha256
            or provenance.diagnosis_semantics_sha256 != self.diagnosis_semantics_sha256
        ):
            raise ValueError("ScoreCard diagnosis semantics provenance is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "diagnosis_contract_version": self.diagnosis_contract_version,
            "diagnosis_rule_registry_sha256": self.diagnosis_rule_registry_sha256,
            "diagnosis_dimension_registry_sha256": self.diagnosis_dimension_registry_sha256,
            "diagnosis_config_sha256": self.diagnosis_config_sha256,
            "diagnosis_semantics_sha256": self.diagnosis_semantics_sha256,
            "diagnosis_semantics_provenance_version": self.diagnosis_semantics_provenance_version,
            "diagnosis_semantics_provenance": dict(self.diagnosis_semantics_provenance),
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
