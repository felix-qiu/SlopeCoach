"""Research/reference A9 AnalysisResult and ProductReport contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from .fingerprint import semantic_sha256, semantic_snapshot
from .registry import (
    ANALYSIS_SECTION_NAMES,
    ANALYSIS_SECTION_REGISTRY_SHA256,
    ANALYSIS_SECTION_REGISTRY_VERSION,
)

ANALYSIS_RESULT_CONTRACT_VERSION = "analysis-result-v1"
ANALYSIS_SECTION_CONTRACT_VERSION = "analysis-section-v1"
PRODUCT_REPORT_CONTRACT_VERSION = "product-report-v1"
PRODUCT_REPORT_PROJECTION_POLICY_VERSION = "product-report-projection-v1"


class AnalysisSectionStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class AnalysisQualityGateStatus(StrEnum):
    READY = "READY"
    PARTIAL_ANALYSIS = "PARTIAL_ANALYSIS"
    NOT_ANALYZABLE = "NOT_ANALYZABLE"


@dataclass(frozen=True)
class AnalysisSection:
    name: str
    status: AnalysisSectionStatus
    payload_contract_version: str | None
    payload: dict[str, object] | None
    payload_sha256: str | None = None
    reason_codes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)
    contract_version: str = ANALYSIS_SECTION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != ANALYSIS_SECTION_CONTRACT_VERSION:
            raise ValueError("ANALYSIS_SECTION_CONTRACT_INCOMPATIBLE")
        if self.name not in ANALYSIS_SECTION_NAMES:
            raise ValueError("ANALYSIS_SECTION_UNKNOWN")
        if self.status is AnalysisSectionStatus.UNAVAILABLE:
            if self.payload is not None or self.payload_sha256 is not None:
                raise ValueError("ANALYSIS_UNAVAILABLE_SECTION_HAS_PAYLOAD")
        else:
            if not isinstance(self.payload, dict) or not self.payload_contract_version:
                raise ValueError("ANALYSIS_AVAILABLE_SECTION_MISSING_PAYLOAD")
            snapshot = semantic_snapshot(self.payload)
            expected = semantic_sha256(snapshot)
            if self.payload_sha256 is not None and self.payload_sha256 != expected:
                raise ValueError("ANALYSIS_SECTION_PAYLOAD_SHA256_INVALID")
            object.__setattr__(self, "payload", snapshot)
            object.__setattr__(self, "payload_sha256", expected)
        object.__setattr__(self, "provenance", semantic_snapshot(self.provenance))

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "name": self.name,
            "status": self.status.value,
            "payload_contract_version": self.payload_contract_version,
            "payload": semantic_snapshot(self.payload),
            "payload_sha256": self.payload_sha256,
            "reason_codes": list(self.reason_codes),
            "limitations": list(self.limitations),
            "provenance": semantic_snapshot(self.provenance),
        }


@dataclass(frozen=True)
class AnalysisResult:
    quality_gate_status: AnalysisQualityGateStatus
    primary_reason_code: str | None
    sections: tuple[AnalysisSection, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    ground_truth: dict[str, str]
    semantic_provenance: dict[str, object]
    engineering_status: str = "PASS_WITH_LIMITATIONS"
    product_validation_status: str = "RESEARCH_ONLY_GT_DEFERRED"
    section_registry_version: str = ANALYSIS_SECTION_REGISTRY_VERSION
    section_registry_sha256: str = ANALYSIS_SECTION_REGISTRY_SHA256
    analysis_result_sha256: str | None = None
    contract_version: str = ANALYSIS_RESULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != ANALYSIS_RESULT_CONTRACT_VERSION:
            raise ValueError("ANALYSIS_RESULT_CONTRACT_INCOMPATIBLE")
        if (
            self.section_registry_version != ANALYSIS_SECTION_REGISTRY_VERSION
            or self.section_registry_sha256 != ANALYSIS_SECTION_REGISTRY_SHA256
        ):
            raise ValueError("ANALYSIS_SECTION_REGISTRY_INCOMPATIBLE")
        names = tuple(section.name for section in self.sections)
        if names != ANALYSIS_SECTION_NAMES:
            raise ValueError("ANALYSIS_SECTION_REGISTRY_SHAPE_INVALID")
        object.__setattr__(self, "ground_truth", semantic_snapshot(self.ground_truth))
        object.__setattr__(self, "semantic_provenance", semantic_snapshot(self.semantic_provenance))
        from .integrity import validate_analysis_result_integrity

        validate_analysis_result_integrity(self)
        expected = semantic_sha256(self.semantic_dict())
        if self.analysis_result_sha256 is not None and self.analysis_result_sha256 != expected:
            raise ValueError("ANALYSIS_RESULT_SHA256_INVALID")
        object.__setattr__(self, "analysis_result_sha256", expected)

    def semantic_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "quality_gate_status": self.quality_gate_status.value,
            "primary_reason_code": self.primary_reason_code,
            "sections": [section.to_dict() for section in self.sections],
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "ground_truth": semantic_snapshot(self.ground_truth),
            "semantic_provenance": semantic_snapshot(self.semantic_provenance),
            "engineering_status": self.engineering_status,
            "product_validation_status": self.product_validation_status,
            "section_registry_version": self.section_registry_version,
            "section_registry_sha256": self.section_registry_sha256,
            "QUALITY_GATE_IS_AVAILABILITY_NOT_PRODUCT_VALIDATION": True,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_dict(), "analysis_result_sha256": self.analysis_result_sha256}

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False, indent=indent)


@dataclass(frozen=True)
class ProductReport:
    source_analysis_result_sha256: str
    status: AnalysisQualityGateStatus
    primary_reason_code: str | None
    availability: dict[str, str]
    sport: dict[str, object] | None
    scorecard: dict[str, object] | None
    headline: str | None
    top_issues: tuple[dict[str, object], ...]
    practice_plan: tuple[dict[str, object], ...]
    evidence_summary: dict[str, object] | None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    numeric_scoring_enabled: bool = False
    overall_score: None = None
    projection_policy_version: str = PRODUCT_REPORT_PROJECTION_POLICY_VERSION
    product_report_sha256: str | None = None
    contract_version: str = PRODUCT_REPORT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PRODUCT_REPORT_CONTRACT_VERSION:
            raise ValueError("PRODUCT_REPORT_CONTRACT_INCOMPATIBLE")
        if self.projection_policy_version != PRODUCT_REPORT_PROJECTION_POLICY_VERSION:
            raise ValueError("PRODUCT_REPORT_PROJECTION_POLICY_INCOMPATIBLE")
        if self.numeric_scoring_enabled or self.overall_score is not None:
            raise ValueError("PRODUCT_REPORT_NUMERIC_SCORE_LEAKAGE")
        if len(self.top_issues) > 2 or len(self.practice_plan) > 2:
            raise ValueError("PRODUCT_REPORT_ISSUE_LIMIT_EXCEEDED")
        for name in ("availability", "sport", "scorecard", "evidence_summary"):
            object.__setattr__(self, name, semantic_snapshot(getattr(self, name)))
        object.__setattr__(self, "top_issues", tuple(semantic_snapshot(x) for x in self.top_issues))
        object.__setattr__(
            self, "practice_plan", tuple(semantic_snapshot(x) for x in self.practice_plan)
        )
        expected = semantic_sha256(self.semantic_dict())
        if self.product_report_sha256 is not None and self.product_report_sha256 != expected:
            raise ValueError("PRODUCT_REPORT_SHA256_INVALID")
        object.__setattr__(self, "product_report_sha256", expected)

    def semantic_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "projection_policy_version": self.projection_policy_version,
            "source_analysis_result_sha256": self.source_analysis_result_sha256,
            "status": self.status.value,
            "primary_reason_code": self.primary_reason_code,
            "availability": semantic_snapshot(self.availability),
            "sport": semantic_snapshot(self.sport),
            "scorecard": semantic_snapshot(self.scorecard),
            "headline": self.headline,
            "top_issues": [semantic_snapshot(x) for x in self.top_issues],
            "practice_plan": [semantic_snapshot(x) for x in self.practice_plan],
            "evidence_summary": semantic_snapshot(self.evidence_summary),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "numeric_scoring_enabled": self.numeric_scoring_enabled,
            "overall_score": self.overall_score,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_dict(), "product_report_sha256": self.product_report_sha256}

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False, indent=indent)
