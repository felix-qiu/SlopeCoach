"""Controlled A8 coach contracts; language output is not a truth layer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum

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
    contract_version: str = COACH_CONTEXT_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "issue_priority_policy_version": "issue-priority-v1",
            "issue_priority_profile": "EVIDENCE_RECURRENCE_PRIORITY_A8",
            "status": self.status.value,
            "sport_type": self.sport_type,
            "sport_type_source": self.sport_type_source,
            "scorecard": self.scorecard,
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
    language: str = "zh-CN"
    contract_version: str = COACH_REPORT_VERSION
    template_version: str = COACH_TEMPLATE_VERSION
    drill_library_version: str = DRILL_LIBRARY_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "coach_context_version": COACH_CONTEXT_VERSION,
            "issue_priority_policy_version": "issue-priority-v1",
            "issue_priority_profile": "EVIDENCE_RECURRENCE_PRIORITY_A8",
            "language": self.language,
            "status": self.status.value,
            "headline": self.headline,
            "scorecard": self.scorecard,
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
