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
    qualified = artifact.get("qualified_turn_count")
    partial = artifact.get("partial_or_noneligible_turn_count")
    return {
        "turn_candidate_count": None,
        "qualified_turn_count": qualified,
        "valid_turn_count": None,
        "partial_turn_count": partial,
        "rejected_turn_count": None,
        "complete_diagnosis_eligible_turn_count": artifact.get(
            "complete_diagnosis_eligible_turn_count"
        ),
        "rejection_reason_counts": dict(artifact.get("blocker_counts", {})),
        "TURN_SEGMENTATION_GT_STATUS": "NOT_AVAILABLE",
        "limitations": ["NO_TURN_GT", "LEGACY_ARTIFACT_COMPACT_TURN_SUMMARY"],
    }
