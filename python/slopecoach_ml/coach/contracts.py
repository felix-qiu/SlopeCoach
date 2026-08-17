"""Controlled A8 coach contracts; language output is not a truth layer."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum

from slopecoach_ml.scoring import IssuePriorityPolicy, issue_priority_policy_sha256

DRILL_LIBRARY_VERSION = "drill-library-v1"
COACH_CONTEXT_VERSION = "coach-context-v1"
COACH_REPORT_VERSION = "coach-report-v1"
COACH_TEMPLATE_VERSION = "coach-template-zh-cn-v1"


class CoachContextStatus(StrEnum):
    NOT_ANALYZABLE_UPSTREAM = "NOT_ANALYZABLE_UPSTREAM"
    EXECUTED_NO_PROVISIONAL_ISSUES = "EXECUTED_NO_PROVISIONAL_ISSUES"
    EXECUTED_WITH_PROVISIONAL_ISSUES = "EXECUTED_WITH_PROVISIONAL_ISSUES"


@dataclass(frozen=True)
class ProvisionalIssueSummary:
    diagnosis_code: str
    dimension: str
    triggered_turn_count: int
    evaluable_turn_count: int
    triggered_turn_ratio: float
    affected_turn_ids: tuple[str, ...]
    evidence_frames: tuple[int, ...]
    feature_ids: tuple[str, ...]
    limitations: tuple[str, ...]
    validation_status: str = "UNVALIDATED_RESEARCH_RULE"
    severity: None = None
    confidence: None = None

    def __post_init__(self) -> None:
        if not self.diagnosis_code or not self.dimension:
            raise ValueError("issue diagnosis code and dimension are required")
        if self.severity is not None or self.confidence is not None:
            raise ValueError("A8 issue severity and confidence must be null")
        if self.validation_status != "UNVALIDATED_RESEARCH_RULE":
            raise ValueError("A8 issue validation status is fixed")
        for name in ("triggered_turn_count", "evaluable_turn_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive non-bool integer")
        if self.triggered_turn_count > self.evaluable_turn_count:
            raise ValueError("issue triggered turns cannot exceed evaluable turns")
        expected = self.triggered_turn_count / self.evaluable_turn_count
        ratio = self.triggered_turn_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, int | float)
            or not math.isfinite(ratio)
            or not 0 < ratio <= 1
            or not math.isclose(ratio, expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("issue trigger ratio is inconsistent")
        for name in ("affected_turn_ids", "evidence_frames", "feature_ids"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate issue evidence in {name}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledDrill:
    drill_id: str
    applicable_sport_types: tuple[str, ...]
    mapped_diagnosis_codes: tuple[str, ...]
    title_zh_cn: str
    goal_zh_cn: str
    steps_zh_cn: tuple[str, ...]
    what_to_notice_zh_cn: tuple[str, ...]
    safety_notes: tuple[str, ...]
    limitations: tuple[str, ...]
    version: str = DRILL_LIBRARY_VERSION
    practice_environment: str = "EASY_CONTROLLED_TERRAIN"
    speed_guidance: str = "LOW_TO_MODERATE_WITHIN_ABILITY"
    research_only: bool = True

    def __post_init__(self) -> None:
        if self.research_only is not True:
            raise ValueError("controlled drills must remain research-only")
        if not self.applicable_sport_types or not set(self.applicable_sport_types) <= {
            "SKI",
            "SNOWBOARD",
        }:
            raise ValueError("controlled drill sport types are invalid")
        if not self.mapped_diagnosis_codes:
            raise ValueError("controlled drill requires a diagnosis mapping")
        required_safety = {
            "PRACTICE_WITHIN_CURRENT_ABILITY",
            "STOP_IF_PAIN_OR_LOSS_OF_CONTROL",
        }
        if not required_safety <= set(self.safety_notes):
            raise ValueError("controlled drill safety metadata is incomplete")
        if self.practice_environment != "EASY_CONTROLLED_TERRAIN":
            raise ValueError("controlled drill environment is fixed")
        if self.speed_guidance != "LOW_TO_MODERATE_WITHIN_ABILITY":
            raise ValueError("controlled drill speed guidance is fixed")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CoachContext:
    status: CoachContextStatus
    sport_type: str
    sport_type_source: str
    scorecard: dict[str, object]
    top_issues: tuple[ProvisionalIssueSummary, ...]
    all_issue_summaries: tuple[ProvisionalIssueSummary, ...]
    upstream_diagnosis_status: str
    diagnosis_limitations: tuple[str, ...]
    turn_counts: dict[str, int]
    evidence_references: tuple[dict[str, object], ...]
    limitations: tuple[str, ...]
    issue_priority_policy: IssuePriorityPolicy
    contract_version: str = COACH_CONTEXT_VERSION

    def __post_init__(self) -> None:
        validate_scorecard_payload(self.scorecard)
        if not isinstance(self.issue_priority_policy, IssuePriorityPolicy):
            raise ValueError("CoachContext requires an IssuePriorityPolicy")
        if len(self.top_issues) > self.issue_priority_policy.max_top_issues:
            raise ValueError("CoachContext exceeds issue priority policy")
        all_codes = {item.diagnosis_code for item in self.all_issue_summaries}
        if any(item.diagnosis_code not in all_codes for item in self.top_issues):
            raise ValueError("CoachContext top issue is absent from all issues")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "issue_priority_policy": self.issue_priority_policy.to_dict(),
            "issue_priority_policy_sha256": issue_priority_policy_sha256(
                self.issue_priority_policy
            ),
            "status": self.status.value,
            "sport_type": self.sport_type,
            "sport_type_source": self.sport_type_source,
            "scorecard": self.scorecard,
            "diagnosis_semantics_provenance": scorecard_provenance(self.scorecard),
            "top_issues": [item.to_dict() for item in self.top_issues],
            "all_issue_summaries": [item.to_dict() for item in self.all_issue_summaries],
            "upstream_diagnosis_status": self.upstream_diagnosis_status,
            "diagnosis_limitations": list(self.diagnosis_limitations),
            "turn_counts": self.turn_counts,
            "evidence_references": list(self.evidence_references),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class CoachReport:
    status: CoachContextStatus
    headline: str
    scorecard: dict[str, object]
    top_issues: tuple[dict[str, object], ...]
    all_issue_summaries: tuple[dict[str, object], ...]
    practice_plan: tuple[dict[str, object], ...]
    evidence_summary: dict[str, object]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    template_registry_sha256: str
    drill_library_sha256: str
    issue_priority_policy: IssuePriorityPolicy
    language: str = "zh-CN"
    contract_version: str = COACH_REPORT_VERSION
    template_version: str = COACH_TEMPLATE_VERSION
    drill_library_version: str = DRILL_LIBRARY_VERSION

    def __post_init__(self) -> None:
        validate_scorecard_payload(self.scorecard)
        if not isinstance(self.issue_priority_policy, IssuePriorityPolicy):
            raise ValueError("CoachReport requires an IssuePriorityPolicy")
        if self.language != "zh-CN":
            raise ValueError("A8 CoachReport language must be zh-CN")
        if (
            len(self.top_issues) > self.issue_priority_policy.max_top_issues
            or len(self.top_issues) > 2
        ):
            raise ValueError("CoachReport exceeds top issue limit")
        if len(self.practice_plan) > len(self.top_issues) or len(self.practice_plan) > 2:
            raise ValueError("CoachReport exceeds practice plan limit")
        top_codes = {item.get("diagnosis_code") for item in self.top_issues}
        if any(
            item.get("severity") is not None or item.get("confidence") is not None
            for item in self.top_issues
        ):
            raise ValueError("CoachReport issue severity and confidence must be null")
        if any(item.get("diagnosis_code") not in top_codes for item in self.practice_plan):
            raise ValueError("practice plan references a non-top issue")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "coach_context_version": COACH_CONTEXT_VERSION,
            "issue_priority_policy": self.issue_priority_policy.to_dict(),
            "issue_priority_policy_sha256": issue_priority_policy_sha256(
                self.issue_priority_policy
            ),
            "language": self.language,
            "status": self.status.value,
            "headline": self.headline,
            "scorecard": self.scorecard,
            "diagnosis_semantics_provenance": scorecard_provenance(self.scorecard),
            "top_issues": list(self.top_issues),
            "all_issue_summaries": list(self.all_issue_summaries),
            "practice_plan": list(self.practice_plan),
            "evidence_summary": self.evidence_summary,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "template_version": self.template_version,
            "template_registry_sha256": self.template_registry_sha256,
            "drill_library_version": self.drill_library_version,
            "drill_library_sha256": self.drill_library_sha256,
            "LLM_COACH_STATUS": "NOT_IMPLEMENTED",
            "COACH_VALIDATION_STATUS": "STRUCTURAL_TEMPLATE_VALIDATION_ONLY",
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False, indent=indent)


def validate_scorecard_payload(scorecard: dict[str, object]) -> None:
    if not isinstance(scorecard, dict):
        raise ValueError("Coach contract requires a serialized ScoreCard")
    if scorecard.get("numeric_scoring_enabled") is not False:
        raise ValueError("Coach contract rejects numeric score leakage")
    if scorecard.get("overall_score") is not None:
        raise ValueError("Coach contract rejects numeric score leakage")
    dimensions = scorecard.get("dimensions")
    if (
        not isinstance(dimensions, list)
        or {item.get("dimension") for item in dimensions if isinstance(item, dict)}
        != {"BALANCE", "EDGE_CONTROL", "STANCE", "SYMMETRY", "TIMING"}
        or len(dimensions) != 5
    ):
        raise ValueError("Coach contract requires exactly five ScoreCard dimensions")
    for dimension in dimensions:
        if not isinstance(dimension, dict) or any(
            dimension.get(field) is not None
            for field in ("score_value", "score_scale_min", "score_scale_max")
        ):
            raise ValueError("Coach contract rejects numeric score leakage")


def scorecard_provenance(scorecard: dict[str, object]) -> dict[str, object]:
    provenance = scorecard.get("diagnosis_semantics_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Coach contract requires Diagnosis semantics provenance")
    aliases = {
        "version": "diagnosis_semantics_provenance_version",
        "diagnosis_contract_version": "diagnosis_contract_version",
        "diagnosis_rule_registry_sha256": "diagnosis_rule_registry_sha256",
        "diagnosis_config_sha256": "diagnosis_config_sha256",
        "diagnosis_semantics_sha256": "diagnosis_semantics_sha256",
    }
    if any(provenance.get(key) != scorecard.get(alias) for key, alias in aliases.items()):
        raise ValueError("Coach contract Diagnosis semantics provenance is inconsistent")
    return dict(provenance)
