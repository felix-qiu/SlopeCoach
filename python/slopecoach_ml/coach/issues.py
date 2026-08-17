"""Evidence-recurrence issue summaries and deterministic top-two policy."""

from __future__ import annotations

from slopecoach_ml.scoring import (
    DIAGNOSIS_DIMENSION_REGISTRY,
    IssuePriorityPolicy,
)

from .contracts import ProvisionalIssueSummary


def build_issue_summaries(diagnosis_result) -> tuple[ProvisionalIssueSummary, ...]:
    payload = (
        diagnosis_result.to_dict() if hasattr(diagnosis_result, "to_dict") else diagnosis_result
    )
    diagnosed_codes = {
        item.get("diagnosis_code")
        for item in payload.get("diagnoses", ())
        if item.get("evaluation_status") == "TRIGGERED"
    }
    summaries = []
    for mapping in DIAGNOSIS_DIMENSION_REGISTRY:
        code = mapping.diagnosis_code
        if code not in diagnosed_codes:
            continue
        evaluations = [
            item
            for item in payload.get("rule_evaluations", ())
            if item.get("diagnosis_code") == code
        ]
        evaluable = [item for item in evaluations if item.get("status") != "NOT_EVALUABLE"]
        triggered = [item for item in evaluable if item.get("status") == "TRIGGERED"]
        if not triggered:
            continue
        turns = tuple(dict.fromkeys(str(item.get("turn_id")) for item in triggered))
        frames = tuple(
            dict.fromkeys(frame for item in triggered for frame in item.get("evidence_frames", ()))
        )
        features = tuple(
            dict.fromkeys(
                str(feature.get("feature_id"))
                for item in triggered
                for feature in item.get("feature_evidence", ())
            )
        )
        summaries.append(
            ProvisionalIssueSummary(
                diagnosis_code=code,
                dimension=mapping.dimension.value,
                triggered_turn_count=len(triggered),
                evaluable_turn_count=len(evaluable),
                triggered_turn_ratio=len(triggered) / len(evaluable),
                affected_turn_ids=turns,
                evidence_frames=frames,
                feature_ids=features,
                limitations=tuple(mapping.limitations),
            )
        )
    return tuple(summaries)


def prioritize_issues(
    issues: tuple[ProvisionalIssueSummary, ...],
    policy: IssuePriorityPolicy | None = None,
) -> tuple[ProvisionalIssueSummary, ...]:
    settings = policy or IssuePriorityPolicy()
    registry_order = {
        item.diagnosis_code: index for index, item in enumerate(DIAGNOSIS_DIMENSION_REGISTRY)
    }
    ordered = sorted(
        issues,
        key=lambda item: (
            -item.triggered_turn_ratio,
            -item.triggered_turn_count,
            registry_order[item.diagnosis_code],
        ),
    )
    return tuple(ordered[: settings.max_top_issues])
