"""Fail-closed cross-section integrity validation for A9."""

from __future__ import annotations

from slopecoach_ml.coach import (
    COACH_TEMPLATE_REGISTRY_SHA256,
    DRILL_LIBRARY_SHA256,
    scorecard_provenance,
    validate_scorecard_payload,
)
from slopecoach_ml.diagnosis import (
    DIAGNOSIS_RULE_REGISTRY_SHA256,
    validate_diagnosis_semantics_provenance,
    validate_diagnosis_truth_consistency,
)
from slopecoach_ml.scoring import (
    DIAGNOSIS_DIMENSION_REGISTRY_SHA256,
    SCORING_POLICY_VERSION,
    IssuePriorityPolicy,
    issue_priority_policy_sha256,
)

_PRIMARY_REASON_PRECEDENCE = (
    "TARGET_IDENTITY_UNCERTAIN",
    "INPUT_NOT_ANALYZABLE",
    "SPORT_TYPE_UNKNOWN",
    "NO_QUALIFIED_TURNS",
    "INSUFFICIENT_DIAGNOSIS_EVIDENCE",
    "SOURCE_IDENTITY_UNAVAILABLE",
    "TARGET_IDENTITY_SUMMARY_UNAVAILABLE",
    "DOWNSTREAM_SECTION_UNAVAILABLE",
)


def derive_quality_gate(sections) -> tuple[str, tuple[str, ...], str | None]:
    """Derive the sole canonical A9 top-level availability decision."""
    by_name = {section.name: section for section in sections}

    def payload(name):
        section = by_name[name]
        return section.payload if section.status.value != "UNAVAILABLE" else None

    source = payload("SOURCE")
    target = payload("TARGET_IDENTITY")
    sport = payload("SPORT_TYPE")
    turns = payload("TURNS")
    diagnosis_section = payload("DIAGNOSIS")
    scorecard = payload("SCORECARD")
    coach = payload("COACH")
    blockers = []
    if target is not None and target.get("safe_for_analysis") is False:
        blockers.append("TARGET_IDENTITY_UNCERTAIN")
    if any("INPUT_NOT_ANALYZABLE" in section.reason_codes for section in sections):
        blockers.append("INPUT_NOT_ANALYZABLE")
    if sport is not None and sport.get("effective_sport_type") not in {"SKI", "SNOWBOARD"}:
        blockers.append("SPORT_TYPE_UNKNOWN")
    if turns is not None and turns.get("qualified_turn_count") == 0:
        blockers.append("NO_QUALIFIED_TURNS")
    diagnosis = diagnosis_section.get("diagnosis_result") if diagnosis_section is not None else None
    if diagnosis is not None and diagnosis.get("status") == (
        "NOT_ANALYZABLE_INSUFFICIENT_DIAGNOSIS_EVIDENCE"
    ):
        blockers.append("INSUFFICIENT_DIAGNOSIS_EVIDENCE")
    if source is None:
        blockers.append("SOURCE_IDENTITY_UNAVAILABLE")
    if target is None:
        blockers.append("TARGET_IDENTITY_SUMMARY_UNAVAILABLE")
    if any(value is None for value in (diagnosis_section, scorecard, coach)):
        blockers.append("DOWNSTREAM_SECTION_UNAVAILABLE")
    blockers_tuple = tuple(dict.fromkeys(blockers))
    primary = next(
        (reason for reason in _PRIMARY_REASON_PRECEDENCE if reason in blockers_tuple), None
    )
    if {"TARGET_IDENTITY_UNCERTAIN", "INPUT_NOT_ANALYZABLE"} & set(blockers_tuple):
        quality = "NOT_ANALYZABLE"
    else:
        ready_diagnosis = diagnosis is not None and diagnosis.get("status") in {
            "EXECUTED_NO_PROVISIONAL_RULES_TRIGGERED",
            "EXECUTED_WITH_PROVISIONAL_DIAGNOSES",
        }
        quality = (
            "READY"
            if source is not None
            and target is not None
            and target.get("safe_for_analysis") is True
            and sport is not None
            and sport.get("effective_sport_type") in {"SKI", "SNOWBOARD"}
            and ready_diagnosis
            and scorecard is not None
            and coach is not None
            else "PARTIAL_ANALYSIS"
        )
    return quality, blockers_tuple, primary


def _available(result, name):
    section = next(section for section in result.sections if section.name == name)
    return section.payload if section.status.value != "UNAVAILABLE" else None


def validate_analysis_result_integrity(result) -> None:
    expected_quality, expected_blockers, expected_primary = derive_quality_gate(result.sections)
    if (
        result.quality_gate_status.value != expected_quality
        or result.blockers != expected_blockers
        or result.primary_reason_code != expected_primary
    ):
        raise ValueError("ANALYSIS_QUALITY_GATE_CONTRACT_INCONSISTENT")
    diagnosis_section = _available(result, "DIAGNOSIS")
    scorecard = _available(result, "SCORECARD")
    coach = _available(result, "COACH")
    sport = _available(result, "SPORT_TYPE")

    diagnosis = None
    provenance = None
    if diagnosis_section is not None:
        diagnosis = diagnosis_section.get("diagnosis_result")
        provenance_payload = diagnosis_section.get("diagnosis_semantics_provenance")
        validate_diagnosis_truth_consistency(diagnosis)
        provenance = validate_diagnosis_semantics_provenance(provenance_payload)
        if provenance.diagnosis_rule_registry_sha256 != DIAGNOSIS_RULE_REGISTRY_SHA256:
            raise ValueError("DIAGNOSIS_RULE_REGISTRY_INCOMPATIBLE")
        if provenance.diagnosis_config != diagnosis.get("config"):
            raise ValueError("DIAGNOSIS_CONFIG_PROVENANCE_MISMATCH")

    if (
        sport is not None
        and diagnosis is not None
        and (
            sport.get("effective_sport_type") != diagnosis.get("sport_type")
            or sport.get("effective_source") != diagnosis.get("sport_type_source")
        )
    ):
        raise ValueError("ANALYSIS_SPORT_TYPE_CONTRACT_INCONSISTENT")

    if scorecard is not None:
        validate_scorecard_payload(scorecard)
        if (
            scorecard.get("diagnosis_dimension_registry_sha256")
            != (DIAGNOSIS_DIMENSION_REGISTRY_SHA256)
            or scorecard.get("scoring_policy_version") != SCORING_POLICY_VERSION
        ):
            raise ValueError("ANALYSIS_SCORECARD_CONTRACT_INCONSISTENT")
        card_provenance = scorecard_provenance(scorecard)
        if provenance is not None and card_provenance != provenance.to_dict():
            raise ValueError("ANALYSIS_SECTION_PROVENANCE_MISMATCH")

    if coach is not None:
        if scorecard is None or coach.get("scorecard") != scorecard:
            raise ValueError("ANALYSIS_SCORECARD_COACH_CONTRACT_INCONSISTENT")
        if provenance is not None and coach.get("diagnosis_semantics_provenance") != (
            provenance.to_dict()
        ):
            raise ValueError("ANALYSIS_SECTION_PROVENANCE_MISMATCH")
        try:
            policy = IssuePriorityPolicy(**coach["issue_priority_policy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ANALYSIS_COACH_POLICY_PROVENANCE_MISSING") from exc
        if (
            coach.get("issue_priority_policy_sha256") != issue_priority_policy_sha256(policy)
            or coach.get("drill_library_sha256") != DRILL_LIBRARY_SHA256
            or coach.get("template_registry_sha256") != COACH_TEMPLATE_REGISTRY_SHA256
        ):
            raise ValueError("ANALYSIS_COACH_POLICY_PROVENANCE_INVALID")
        triggered = {
            item.get("diagnosis_code")
            for item in (diagnosis or {}).get("diagnoses", ())
            if item.get("evaluation_status") == "TRIGGERED"
        }
        top_codes = {item.get("diagnosis_code") for item in coach.get("top_issues", ())}
        if not top_codes <= triggered:
            raise ValueError("ANALYSIS_COACH_ISSUE_DIAGNOSIS_INCONSISTENT")
        if any(
            item.get("diagnosis_code") not in top_codes for item in coach.get("practice_plan", ())
        ):
            raise ValueError("ANALYSIS_COACH_PRACTICE_ISSUE_INCONSISTENT")

    identities = set()
    for section in result.sections:
        if section.payload:
            for key in ("source_video_id", "source_video_sha256"):
                value = section.payload.get(key)
                if value:
                    identities.add((key, value))
    for key in ("source_video_id", "source_video_sha256"):
        if len({value for name, value in identities if name == key}) > 1:
            raise ValueError("ANALYSIS_SOURCE_IDENTITY_MISMATCH")
