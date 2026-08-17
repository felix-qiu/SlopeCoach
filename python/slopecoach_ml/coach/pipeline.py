"""Controlled downstream bridge from DiagnosisResult to deterministic CoachReport."""

from __future__ import annotations

from slopecoach_ml.scoring import IssuePriorityPolicy, build_scorecard

from .contracts import CoachContext, CoachContextStatus, CoachReport
from .drills import DRILL_LIBRARY_SHA256, drill_for_diagnosis
from .issues import build_issue_summaries, prioritize_issues
from .templates import (
    COACH_TEMPLATE_REGISTRY_SHA256,
    LANGUAGE_POLICY,
    render_headline,
    render_issue_template,
)

_NOT_ANALYZABLE = {
    "NOT_ANALYZABLE_SPORT_TYPE_UNKNOWN",
    "NOT_ANALYZABLE_NO_QUALIFIED_TURNS",
    "NOT_ANALYZABLE_INSUFFICIENT_DIAGNOSIS_EVIDENCE",
}


def build_coach_context(
    diagnosis_result,
    *,
    issue_policy: IssuePriorityPolicy | None = None,
    diagnosis_semantics_provenance=None,
    trusted_current_process: bool = False,
) -> CoachContext:
    settings = issue_policy or IssuePriorityPolicy()
    payload = (
        diagnosis_result.to_dict() if hasattr(diagnosis_result, "to_dict") else diagnosis_result
    )
    scorecard = build_scorecard(
        diagnosis_result,
        diagnosis_semantics_provenance=diagnosis_semantics_provenance,
        trusted_current_process=trusted_current_process,
    ).to_dict()
    upstream = str(payload.get("status"))
    if upstream in _NOT_ANALYZABLE:
        status = CoachContextStatus.NOT_ANALYZABLE_UPSTREAM
        all_issues = ()
        top_issues = ()
    else:
        all_issues = build_issue_summaries(payload)
        top_issues = prioritize_issues(all_issues, settings)
        status = (
            CoachContextStatus.EXECUTED_WITH_PROVISIONAL_ISSUES
            if top_issues
            else CoachContextStatus.EXECUTED_NO_PROVISIONAL_ISSUES
        )
    evaluations = payload.get("rule_evaluations", ())
    turn_ids = {item.get("turn_id") for item in evaluations if item.get("turn_id") is not None}
    evaluable_turn_ids = {
        item.get("turn_id")
        for item in evaluations
        if item.get("status") in {"TRIGGERED", "NOT_TRIGGERED"}
    }
    evidence = tuple(
        reference
        for dimension in scorecard["dimensions"]
        for reference in dimension["evidence_references"]
    )
    upstream_limitations = tuple(payload.get("limitations", ()))
    limitations = tuple(dict.fromkeys((*scorecard["limitations"], *upstream_limitations)))
    if status is CoachContextStatus.EXECUTED_NO_PROVISIONAL_ISSUES:
        limitations = (*limitations, "NO_TRIGGER_DOES_NOT_MEAN_GOOD_FORM")
    return CoachContext(
        status=status,
        sport_type=str(payload.get("sport_type", "UNKNOWN")),
        sport_type_source=str(payload.get("sport_type_source", "AUTO")),
        scorecard=scorecard,
        top_issues=top_issues,
        all_issue_summaries=all_issues,
        upstream_diagnosis_status=upstream,
        diagnosis_limitations=upstream_limitations,
        turn_counts={
            "turn_count": len(turn_ids),
            "evaluable_turn_count": len(evaluable_turn_ids),
        },
        evidence_references=evidence,
        limitations=limitations,
        issue_priority_policy=settings,
    )


def build_coach_report(
    diagnosis_result,
    *,
    issue_policy: IssuePriorityPolicy | None = None,
    diagnosis_semantics_provenance=None,
    trusted_current_process: bool = False,
) -> CoachReport:
    context = (
        diagnosis_result
        if isinstance(diagnosis_result, CoachContext)
        else build_coach_context(
            diagnosis_result,
            issue_policy=issue_policy,
            diagnosis_semantics_provenance=diagnosis_semantics_provenance,
            trusted_current_process=trusted_current_process,
        )
    )
    headline = render_headline(
        status=context.status.value,
        issue_count=len(context.top_issues),
        upstream_status=context.upstream_diagnosis_status,
    )
    if context.status is CoachContextStatus.NOT_ANALYZABLE_UPSTREAM:
        practice_plan = ()
    elif context.top_issues:
        practice_plan = _practice_plan(context)
    else:
        practice_plan = ()
    return CoachReport(
        status=context.status,
        headline=headline,
        scorecard=context.scorecard,
        top_issues=tuple(item.to_dict() for item in context.top_issues),
        all_issue_summaries=tuple(item.to_dict() for item in context.all_issue_summaries),
        practice_plan=practice_plan,
        evidence_summary={
            "turn_counts": context.turn_counts,
            "evidence_references": list(context.evidence_references),
        },
        warnings=LANGUAGE_POLICY.controlled_warnings,
        limitations=context.limitations,
        template_registry_sha256=COACH_TEMPLATE_REGISTRY_SHA256,
        drill_library_sha256=DRILL_LIBRARY_SHA256,
        issue_priority_policy=context.issue_priority_policy,
    )


def _practice_plan(context: CoachContext) -> tuple[dict[str, object], ...]:
    items = []
    used_drills = set()
    for issue in context.top_issues:
        drill = drill_for_diagnosis(issue.diagnosis_code, context.sport_type)
        if drill.drill_id in used_drills:
            continue
        used_drills.add(drill.drill_id)
        rendered = render_issue_template(issue)
        items.append(
            {
                "priority_rank": len(items) + 1,
                "diagnosis_code": issue.diagnosis_code,
                "dimension": issue.dimension,
                "drill": drill.to_dict(),
                "template_id": rendered["template_id"],
                "title": rendered["title"],
                "why_this_focus": rendered["explanation"],
                "evidence": rendered["evidence"],
                "limitation": rendered["limitation"],
                "evidence_reference": {
                    "affected_turn_ids": list(issue.affected_turn_ids),
                    "evidence_frames": list(issue.evidence_frames),
                    "feature_ids": list(issue.feature_ids),
                },
            }
        )
    return tuple(items)
