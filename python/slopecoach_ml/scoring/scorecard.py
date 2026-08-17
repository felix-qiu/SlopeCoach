"""Build a structure-only scorecard from immutable DiagnosisResult facts."""

from __future__ import annotations

from slopecoach_ml.diagnosis import DIAGNOSIS_RULE_REGISTRY_SHA256

from .contracts import (
    DimensionAssessment,
    DimensionAssessmentStatus,
    ScoreCard,
    ScoreDimension,
    ScoringPolicy,
)
from .registry import DIAGNOSIS_DIMENSION_REGISTRY, DIAGNOSIS_DIMENSION_REGISTRY_SHA256

SCORECARD_LIMITATIONS = (
    "NUMERIC_SCORE_NOT_CALIBRATED",
    "NO_DIAGNOSIS_GT",
    "NO_TURN_GT",
    "A7_RULES_PROVISIONAL",
    "IMAGE_SPACE_2D_ONLY",
    "NO_GOOD_FORM_INFERENCE",
    "PYTHON_RESEARCH_REFERENCE_ONLY",
)


def build_scorecard(diagnosis_result, policy: ScoringPolicy | None = None) -> ScoreCard:
    settings = policy or ScoringPolicy()
    payload = _payload(diagnosis_result)
    evaluations = tuple(payload.get("rule_evaluations", ()))
    assessments = []
    for dimension in ScoreDimension:
        mappings = tuple(
            item for item in DIAGNOSIS_DIMENSION_REGISTRY if item.dimension is dimension
        )
        codes = tuple(item.diagnosis_code for item in mappings)
        if not mappings:
            assessments.append(
                DimensionAssessment(
                    dimension=dimension,
                    status=DimensionAssessmentStatus.NOT_IMPLEMENTED,
                    mapped_diagnosis_codes=(),
                    rule_count=0,
                    evaluable_rule_turn_count=0,
                    triggered_rule_turn_count=0,
                    not_triggered_rule_turn_count=0,
                    not_evaluable_rule_turn_count=0,
                    triggered_turn_ratio=None,
                    evidence_references=(),
                    limitations=("NO_CURRENT_DIAGNOSIS_RULE_FOR_DIMENSION",),
                    reason="NO_CURRENT_DIAGNOSIS_RULE_FOR_DIMENSION",
                )
            )
            continue
        items = [item for item in evaluations if item.get("diagnosis_code") in codes]
        _validate_evaluations(items)
        triggered = sum(item["status"] == "TRIGGERED" for item in items)
        not_triggered = sum(item["status"] == "NOT_TRIGGERED" for item in items)
        not_evaluable = sum(item["status"] == "NOT_EVALUABLE" for item in items)
        evaluable = triggered + not_triggered
        if evaluable == 0:
            status = DimensionAssessmentStatus.NOT_EVALUABLE
        elif triggered:
            status = DimensionAssessmentStatus.PROVISIONAL_ISSUE_DETECTED
        else:
            status = DimensionAssessmentStatus.NO_PROVISIONAL_ISSUE_DETECTED
        evidence = tuple(_evidence_reference(item) for item in items)
        limitations = tuple(dict.fromkeys(x for mapping in mappings for x in mapping.limitations))
        assessments.append(
            DimensionAssessment(
                dimension=dimension,
                status=status,
                mapped_diagnosis_codes=codes,
                rule_count=len(mappings),
                evaluable_rule_turn_count=evaluable,
                triggered_rule_turn_count=triggered,
                not_triggered_rule_turn_count=not_triggered,
                not_evaluable_rule_turn_count=not_evaluable,
                triggered_turn_ratio=triggered / evaluable if evaluable else None,
                evidence_references=evidence,
                limitations=limitations,
            )
        )
    return ScoreCard(
        dimensions=tuple(assessments),
        diagnosis_contract_version=str(payload.get("contract_version", "diagnosis-v1")),
        diagnosis_rule_registry_sha256=DIAGNOSIS_RULE_REGISTRY_SHA256,
        diagnosis_dimension_registry_sha256=DIAGNOSIS_DIMENSION_REGISTRY_SHA256,
        scoring_policy_version=settings.version,
        limitations=SCORECARD_LIMITATIONS,
    )


def _payload(result) -> dict[str, object]:
    return result.to_dict() if hasattr(result, "to_dict") else result


def _validate_evaluations(items) -> None:
    allowed = {"TRIGGERED", "NOT_TRIGGERED", "NOT_EVALUABLE"}
    if any(item.get("status") not in allowed for item in items):
        raise ValueError("invalid diagnosis evaluation status")


def _evidence_reference(item) -> dict[str, object]:
    return {
        "diagnosis_code": item.get("diagnosis_code"),
        "turn_id": item.get("turn_id"),
        "status": item.get("status"),
        "evidence_frames": list(item.get("evidence_frames", ())),
        "feature_ids": [fact.get("feature_id") for fact in item.get("feature_evidence", ())],
    }
