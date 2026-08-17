"""Artifact-only A9 AnalysisResult and ProductReport benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path

from slopecoach_ml.analysis_result import build_analysis_result, build_product_report
from slopecoach_ml.coach import build_coach_context, build_coach_report
from slopecoach_ml.scoring import build_scorecard

from .diagnosis_compatibility import validate_diagnosis_artifact_compatibility

ANALYSIS_RESULT_BENCHMARK_VERSION = "ski-bench-analysis-result-v1"


def benchmark_analysis_result_artifact(path: str | Path) -> dict[str, object]:
    started_total = time.perf_counter()
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))

    started = time.perf_counter()
    diagnosis, provenance, compatibility = validate_diagnosis_artifact_compatibility(artifact)
    compatibility_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scorecard = build_scorecard(diagnosis, diagnosis_semantics_provenance=provenance).to_dict()
    scorecard_seconds = time.perf_counter() - started

    started = time.perf_counter()
    context = build_coach_context(diagnosis, diagnosis_semantics_provenance=provenance)
    coach = build_coach_report(context).to_dict()
    coach_seconds = time.perf_counter() - started

    sport = _sport_summary(artifact)
    turns = _turn_summary(artifact)
    started = time.perf_counter()
    analysis_result = build_analysis_result(
        source=None,
        target_identity=None,
        sport_type=sport,
        turns=turns,
        biomechanics=None,
        diagnosis=diagnosis,
        diagnosis_semantics_provenance=provenance,
        scorecard=scorecard,
        coach=coach,
    )
    analysis_seconds = time.perf_counter() - started

    started = time.perf_counter()
    product_report = build_product_report(analysis_result)
    product_seconds = time.perf_counter() - started
    sections = {section.name: section.status.value for section in analysis_result.sections}
    report = {
        "benchmark_contract_version": ANALYSIS_RESULT_BENCHMARK_VERSION,
        "input_artifact_contract": artifact.get("benchmark_contract_version"),
        "input_compatibility": compatibility,
        "REAL_A9_MODEL_RERUN": False,
        "analysis_result": analysis_result.to_dict(),
        "product_report": product_report.to_dict(),
        "analysis_result_sha256": analysis_result.analysis_result_sha256,
        "product_report_sha256": product_report.product_report_sha256,
        "section_availability_summary": sections,
        "quality_gate": analysis_result.quality_gate_status.value,
        "primary_reason_code": analysis_result.primary_reason_code,
        "fingerprints": analysis_result.semantic_provenance,
        "ground_truth": analysis_result.ground_truth,
        "validation": {
            "A9_ANALYSIS_RESULT_ENGINEERING_VALIDATION": "PASS_WITH_LIMITATIONS",
            "A9_PRODUCT_REPORT_ENGINEERING_VALIDATION": "PASS_WITH_LIMITATIONS",
            "A9_REAL_PRODUCT_ACCURACY_VALIDATION": "NOT_VALIDATED_GT_REQUIRED",
            "A9_NUMERIC_SCORE_VALIDATION": "NOT_VALIDATED_GT_REQUIRED",
            "A9_PRODUCT_VALIDATION": "RESEARCH_ONLY_GT_DEFERRED",
        },
        "performance": {
            "compatibility_validation_seconds": compatibility_seconds,
            "scorecard_build_seconds": scorecard_seconds,
            "coach_build_seconds": coach_seconds,
            "analysis_result_build_seconds": analysis_seconds,
            "product_report_build_seconds": product_seconds,
            "total_seconds": time.perf_counter() - started_total,
        },
    }
    json.dumps(report, sort_keys=True, allow_nan=False)
    return report


def _sport_summary(artifact):
    value = artifact.get("sport_input")
    if not isinstance(value, dict):
        return None
    if not {"effective_sport_type", "effective_source"} <= set(value):
        return None
    return {
        "effective_sport_type": value["effective_sport_type"],
        "effective_source": value["effective_source"],
        "resolution_status": "EXPLICIT_LEGACY_ARTIFACT_FACT",
        "CALIBRATED_FUSION_CONTROLS_ROUTING": False,
        "limitations": list(artifact.get("diagnosis_result", {}).get("limitations", ())),
    }


def _turn_summary(artifact):
    if "qualified_turn_count" not in artifact:
        return None
    count_fields = (
        "turn_candidate_count",
        "qualified_turn_count",
        "valid_turn_count",
        "partial_turn_count",
        "rejected_turn_count",
        "complete_diagnosis_eligible_turn_count",
    )
    counts = {name: artifact.get(name) for name in count_fields}
    for value in counts.values():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError("A9_TURN_SUMMARY_INCONSISTENT")
    reasons_explicit = "rejection_reason_counts" in artifact
    raw_reasons = artifact.get("rejection_reason_counts")
    if reasons_explicit:
        if not isinstance(raw_reasons, dict) or any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in raw_reasons.items()
        ):
            raise ValueError("A9_TURN_SUMMARY_INCONSISTENT")
        rejection_reasons = dict(raw_reasons)
    else:
        rejection_reasons = None
    core_fields = count_fields[:5]
    complete = all(counts[name] is not None for name in core_fields)
    if complete and (
        counts["turn_candidate_count"]
        != counts["valid_turn_count"] + counts["partial_turn_count"] + counts["rejected_turn_count"]
        or counts["qualified_turn_count"]
        != counts["valid_turn_count"] + counts["partial_turn_count"]
    ):
        raise ValueError("A9_TURN_SUMMARY_INCONSISTENT")
    if (
        complete
        and reasons_explicit
        and sum(rejection_reasons.values()) != counts["rejected_turn_count"]
    ):
        raise ValueError("A9_TURN_SUMMARY_INCONSISTENT")
    limitations = ["NO_TURN_GT"]
    if not all(counts[name] is not None for name in count_fields) or not reasons_explicit:
        limitations.append("LEGACY_ARTIFACT_INCOMPLETE_TURN_SUMMARY")
    ground_truth = artifact.get("ground_truth")
    turn_gt = (
        ground_truth.get("TURN_SEGMENTATION_GT_STATUS") if isinstance(ground_truth, dict) else None
    )
    return {
        **counts,
        "rejection_reason_counts": rejection_reasons,
        "TURN_SEGMENTATION_GT_STATUS": turn_gt or "NOT_AVAILABLE",
        "limitations": limitations,
    }
