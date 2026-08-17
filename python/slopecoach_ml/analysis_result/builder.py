"""Assembly-only builder for the A9 machine semantic result boundary."""

from __future__ import annotations

from slopecoach_ml.coach import (
    COACH_REPORT_VERSION,
    COACH_TEMPLATE_REGISTRY_SHA256,
    DRILL_LIBRARY_SHA256,
)
from slopecoach_ml.diagnosis import DIAGNOSIS_CONTRACT_VERSION
from slopecoach_ml.scoring import (
    DIAGNOSIS_DIMENSION_REGISTRY_SHA256,
    SCORECARD_CONTRACT_VERSION,
)

from .contracts import (
    AnalysisQualityGateStatus,
    AnalysisResult,
    AnalysisSection,
    AnalysisSectionStatus,
)
from .registry import ANALYSIS_SECTION_NAMES, ANALYSIS_SECTION_REGISTRY_SHA256

GROUND_TRUTH_DEFAULTS = {
    "TARGET_IDENTITY_GT_ANNOTATION_STATUS": "DEFERRED",
    "SPORT_TYPE_GT_STATUS": "NOT_AVAILABLE",
    "TURN_SEGMENTATION_GT_STATUS": "NOT_AVAILABLE",
    "DIAGNOSIS_GT_STATUS": "NOT_AVAILABLE",
    "SCORE_GT_STATUS": "NOT_AVAILABLE",
}

_DEFAULT_UNAVAILABLE = {
    "SOURCE": "SOURCE_IDENTITY_NOT_EMBEDDED_IN_SOURCE_ARTIFACT",
    "TARGET_IDENTITY": "TARGET_IDENTITY_SUMMARY_NOT_EMBEDDED",
    "SPORT_TYPE": "SPORT_TYPE_SUMMARY_NOT_EMBEDDED",
    "TURNS": "TURN_SUMMARY_NOT_EMBEDDED",
    "BIOMECHANICS": "BIOMECHANICS_SUMMARY_NOT_EMBEDDED",
    "DIAGNOSIS": "DIAGNOSIS_RESULT_NOT_EMBEDDED",
    "SCORECARD": "SCORECARD_NOT_AVAILABLE",
    "COACH": "COACH_REPORT_NOT_AVAILABLE",
}

_PAYLOAD_VERSIONS = {
    "SOURCE": "source-summary-v1",
    "TARGET_IDENTITY": "target-identity-summary-v1",
    "SPORT_TYPE": "sport-type-summary-v1",
    "TURNS": "turn-summary-v1",
    "BIOMECHANICS": "biomechanics-summary-v1",
    "DIAGNOSIS": DIAGNOSIS_CONTRACT_VERSION,
    "SCORECARD": SCORECARD_CONTRACT_VERSION,
    "COACH": COACH_REPORT_VERSION,
}

_COMPACT_FIELDS = {
    "SOURCE": {
        "source_video_id",
        "source_video_sha256",
        "duration_us",
        "width_px",
        "height_px",
        "limitations",
    },
    "TARGET_IDENTITY": {
        "state",
        "safe_for_analysis",
        "selection_mode",
        "coverage",
        "TARGET_IDENTITY_GT_ANNOTATION_STATUS",
        "source_video_id",
        "source_video_sha256",
        "limitations",
    },
    "SPORT_TYPE": {
        "effective_sport_type",
        "effective_source",
        "resolution_status",
        "CALIBRATED_FUSION_CONTROLS_ROUTING",
        "source_video_id",
        "source_video_sha256",
        "limitations",
    },
    "TURNS": {
        "turn_candidate_count",
        "qualified_turn_count",
        "valid_turn_count",
        "partial_turn_count",
        "rejected_turn_count",
        "complete_diagnosis_eligible_turn_count",
        "rejection_reason_counts",
        "TURN_SEGMENTATION_GT_STATUS",
        "source_video_id",
        "source_video_sha256",
        "limitations",
    },
    "BIOMECHANICS": {
        "feature_registry_count",
        "feature_registry_sha256",
        "frame_feature_count",
        "temporal_feature_count",
        "turn_feature_count",
        "frame_fact_count",
        "turn_feature_group_count",
        "availability_summary",
        "coverage_summary",
        "source_video_id",
        "source_video_sha256",
        "limitations",
    },
}

_PRIMARY_PRECEDENCE = (
    "TARGET_IDENTITY_UNCERTAIN",
    "INPUT_NOT_ANALYZABLE",
    "SPORT_TYPE_UNKNOWN",
    "NO_QUALIFIED_TURNS",
    "INSUFFICIENT_DIAGNOSIS_EVIDENCE",
    "SOURCE_IDENTITY_UNAVAILABLE",
    "TARGET_IDENTITY_SUMMARY_UNAVAILABLE",
    "DOWNSTREAM_SECTION_UNAVAILABLE",
)


def build_analysis_result(
    *,
    source=None,
    target_identity=None,
    sport_type=None,
    turns=None,
    biomechanics=None,
    diagnosis=None,
    diagnosis_semantics_provenance=None,
    scorecard=None,
    coach=None,
    unavailable_reasons: dict[str, str] | None = None,
) -> AnalysisResult:
    """Build without recomputing upstream truth or mutating caller-owned payloads."""
    payloads = {
        "SOURCE": _compact("SOURCE", source),
        "TARGET_IDENTITY": _compact("TARGET_IDENTITY", target_identity),
        "SPORT_TYPE": _compact("SPORT_TYPE", sport_type),
        "TURNS": _compact("TURNS", turns),
        "BIOMECHANICS": _compact("BIOMECHANICS", biomechanics),
        "DIAGNOSIS": (
            {
                "diagnosis_result": diagnosis,
                "diagnosis_semantics_provenance": diagnosis_semantics_provenance,
            }
            if diagnosis is not None
            else None
        ),
        "SCORECARD": scorecard,
        "COACH": coach,
    }
    reasons = {**_DEFAULT_UNAVAILABLE, **(unavailable_reasons or {})}
    sections = tuple(
        _section(name, payloads[name], reasons[name]) for name in ANALYSIS_SECTION_NAMES
    )
    blockers = _blockers(payloads)
    quality = _quality(payloads, blockers)
    primary = next((reason for reason in _PRIMARY_PRECEDENCE if reason in blockers), None)
    limitations = _merge_limitations(sections)
    warnings = tuple(dict.fromkeys((coach or {}).get("warnings", ())))
    provenance = _semantic_provenance(
        diagnosis_semantics_provenance, scorecard, coach, biomechanics
    )
    return AnalysisResult(
        quality_gate_status=quality,
        primary_reason_code=primary,
        sections=sections,
        blockers=blockers,
        warnings=warnings,
        limitations=limitations,
        ground_truth=GROUND_TRUTH_DEFAULTS,
        semantic_provenance=provenance,
    )


def _compact(name, payload):
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{name}_SUMMARY_INVALID")
    compact = {key: value for key, value in payload.items() if key in _COMPACT_FIELDS[name]}
    required = {
        "SOURCE": bool(compact.get("source_video_id") or compact.get("source_video_sha256")),
        "TARGET_IDENTITY": isinstance(compact.get("safe_for_analysis"), bool)
        and bool(compact.get("state")),
        "SPORT_TYPE": bool(compact.get("effective_sport_type"))
        and bool(compact.get("effective_source")),
        "TURNS": isinstance(compact.get("qualified_turn_count"), int)
        and not isinstance(compact.get("qualified_turn_count"), bool),
        "BIOMECHANICS": bool(compact.get("feature_registry_sha256"))
        or isinstance(compact.get("feature_registry_count"), int),
    }
    return compact if required[name] else None


def _section(name, payload, unavailable_reason):
    if payload is None:
        return AnalysisSection(
            name=name,
            status=AnalysisSectionStatus.UNAVAILABLE,
            payload_contract_version=None,
            payload=None,
            reason_codes=(unavailable_reason,),
            limitations=(unavailable_reason,),
        )
    limitations = tuple(payload.get("limitations", ()))
    return AnalysisSection(
        name=name,
        status=AnalysisSectionStatus.AVAILABLE,
        payload_contract_version=_PAYLOAD_VERSIONS[name],
        payload=payload,
        limitations=limitations,
    )


def _blockers(payloads) -> tuple[str, ...]:
    blockers = []
    target = payloads["TARGET_IDENTITY"]
    sport = payloads["SPORT_TYPE"]
    turns = payloads["TURNS"]
    diagnosis = payloads["DIAGNOSIS"]
    if target is not None and target.get("safe_for_analysis") is False:
        blockers.append("TARGET_IDENTITY_UNCERTAIN")
    if sport is not None and sport.get("effective_sport_type") not in {"SKI", "SNOWBOARD"}:
        blockers.append("SPORT_TYPE_UNKNOWN")
    if turns is not None and turns.get("qualified_turn_count") == 0:
        blockers.append("NO_QUALIFIED_TURNS")
    if diagnosis is not None and diagnosis["diagnosis_result"].get("status") == (
        "NOT_ANALYZABLE_INSUFFICIENT_DIAGNOSIS_EVIDENCE"
    ):
        blockers.append("INSUFFICIENT_DIAGNOSIS_EVIDENCE")
    if payloads["SOURCE"] is None:
        blockers.append("SOURCE_IDENTITY_UNAVAILABLE")
    if target is None:
        blockers.append("TARGET_IDENTITY_SUMMARY_UNAVAILABLE")
    if any(payloads[name] is None for name in ("DIAGNOSIS", "SCORECARD", "COACH")):
        blockers.append("DOWNSTREAM_SECTION_UNAVAILABLE")
    return tuple(dict.fromkeys(blockers))


def _quality(payloads, blockers):
    if "TARGET_IDENTITY_UNCERTAIN" in blockers or "INPUT_NOT_ANALYZABLE" in blockers:
        return AnalysisQualityGateStatus.NOT_ANALYZABLE
    diagnosis = payloads["DIAGNOSIS"]
    ready_diagnosis = diagnosis is not None and diagnosis["diagnosis_result"].get("status") in {
        "EXECUTED_NO_PROVISIONAL_RULES_TRIGGERED",
        "EXECUTED_WITH_PROVISIONAL_DIAGNOSES",
    }
    if (
        payloads["SOURCE"] is not None
        and payloads["TARGET_IDENTITY"] is not None
        and payloads["TARGET_IDENTITY"].get("safe_for_analysis") is True
        and payloads["SPORT_TYPE"] is not None
        and payloads["SPORT_TYPE"].get("effective_sport_type") in {"SKI", "SNOWBOARD"}
        and ready_diagnosis
        and payloads["SCORECARD"] is not None
        and payloads["COACH"] is not None
    ):
        return AnalysisQualityGateStatus.READY
    return AnalysisQualityGateStatus.PARTIAL_ANALYSIS


def _merge_limitations(sections):
    return tuple(
        dict.fromkeys(limitation for section in sections for limitation in section.limitations)
    )


def _semantic_provenance(provenance, scorecard, coach, biomechanics):
    provenance = provenance or {}
    scorecard = scorecard or {}
    coach = coach or {}
    biomechanics = biomechanics or {}
    return {
        "diagnosis_rule_registry_sha256": provenance.get("diagnosis_rule_registry_sha256"),
        "diagnosis_config_sha256": provenance.get("diagnosis_config_sha256"),
        "diagnosis_semantics_sha256": provenance.get("diagnosis_semantics_sha256"),
        "diagnosis_dimension_registry_sha256": scorecard.get("diagnosis_dimension_registry_sha256"),
        "issue_priority_policy_sha256": coach.get("issue_priority_policy_sha256"),
        "drill_library_sha256": coach.get("drill_library_sha256"),
        "coach_template_registry_sha256": coach.get("template_registry_sha256"),
        "analysis_section_registry_sha256": ANALYSIS_SECTION_REGISTRY_SHA256,
        "feature_registry_sha256": biomechanics.get("feature_registry_sha256"),
        "expected_diagnosis_dimension_registry_sha256": (DIAGNOSIS_DIMENSION_REGISTRY_SHA256),
        "expected_drill_library_sha256": DRILL_LIBRARY_SHA256,
        "expected_coach_template_registry_sha256": COACH_TEMPLATE_REGISTRY_SHA256,
    }
