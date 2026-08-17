"""Artifact-only A8 scorecard and controlled-coach benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path

from slopecoach_ml.coach import (
    COACH_TEMPLATE_REGISTRY_SHA256,
    DRILL_LIBRARY_SHA256,
    build_coach_context,
    build_coach_report,
    build_issue_summaries,
    prioritize_issues,
)
from slopecoach_ml.scoring import (
    DIAGNOSIS_DIMENSION_REGISTRY_SHA256,
    build_scorecard,
)

SCORING_COACH_BENCHMARK_VERSION = "ski-bench-scoring-coach-v1"


def benchmark_scoring_coach_artifact(path: str | Path) -> dict[str, object]:
    started_total = time.perf_counter()
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("benchmark_contract_version") != "ski-bench-diagnosis-v1":
        raise ValueError("ARTIFACT_MISSING_DIAGNOSIS_RESULT")
    diagnosis = artifact.get("diagnosis_result")
    if not isinstance(diagnosis, dict) or diagnosis.get("contract_version") != "diagnosis-v1":
        raise ValueError("ARTIFACT_MISSING_DIAGNOSIS_RESULT")

    started = time.perf_counter()
    scorecard = build_scorecard(diagnosis).to_dict()
    scorecard_seconds = time.perf_counter() - started
    started = time.perf_counter()
    all_issues = build_issue_summaries(diagnosis)
    top_issues = prioritize_issues(all_issues)
    issue_seconds = time.perf_counter() - started
    started = time.perf_counter()
    context = build_coach_context(diagnosis)
    context_seconds = time.perf_counter() - started
    started = time.perf_counter()
    coach = build_coach_report(context).to_dict()
    template_seconds = time.perf_counter() - started

    report = {
        "benchmark_contract_version": SCORING_COACH_BENCHMARK_VERSION,
        "input_kind": "EXISTING_DIAGNOSIS_ARTIFACT",
        "input_diagnosis_contract_version": diagnosis["contract_version"],
        "scorecard_contract_version": scorecard["contract_version"],
        "scoring_policy_version": scorecard["scoring_policy_version"],
        "issue_priority_policy_version": "issue-priority-v1",
        "coach_context_version": "coach-context-v1",
        "coach_report_version": coach["contract_version"],
        "coach_template_version": coach["template_version"],
        "drill_library_version": coach["drill_library_version"],
        "scorecard": scorecard,
        "dimension_assessment_summary": {
            item["dimension"]: item["status"] for item in scorecard["dimensions"]
        },
        "numeric_scoring_status": "DISABLED_NOT_CALIBRATED_GT_REQUIRED",
        "top_issue_count": len(top_issues),
        "top_issue_codes": [item.diagnosis_code for item in top_issues],
        "all_issue_count": len(all_issues),
        "practice_item_count": len(coach["practice_plan"]),
        "drill_ids": [item["drill"]["drill_id"] for item in coach["practice_plan"]],
        "template_ids": [item["template_id"] for item in coach["practice_plan"]],
        "coach_report": coach,
        "fingerprints": {
            "diagnosis_rule_registry_sha256": scorecard["diagnosis_rule_registry_sha256"],
            "diagnosis_dimension_registry_sha256": DIAGNOSIS_DIMENSION_REGISTRY_SHA256,
            "drill_library_sha256": DRILL_LIBRARY_SHA256,
            "coach_template_registry_sha256": COACH_TEMPLATE_REGISTRY_SHA256,
        },
        "limitations": list(context.limitations),
        "ground_truth": {
            "SCORE_GT_STATUS": "NOT_AVAILABLE",
            "DIAGNOSIS_GT_STATUS": "NOT_AVAILABLE",
            "TURN_SEGMENTATION_GT_STATUS": "NOT_AVAILABLE",
            "score_mae": None,
            "score_rmse": None,
            "score_correlation": None,
        },
        "validation": {
            "A8_SCORECARD_ENGINEERING_VALIDATION": "PASS_WITH_LIMITATIONS",
            "A8_COACH_ENGINEERING_VALIDATION": "PASS_WITH_LIMITATIONS",
            "A8_NUMERIC_SCORE_VALIDATION": "NOT_VALIDATED_GT_REQUIRED",
            "A8_COACH_EFFECTIVENESS_VALIDATION": "NOT_VALIDATED_GT_REQUIRED",
            "A8_PRODUCT_VALIDATION": "BLOCKED_BY_GT_AND_SCORE_CALIBRATION",
        },
        "performance": {
            "scorecard_build_seconds": scorecard_seconds,
            "issue_prioritization_seconds": issue_seconds,
            "coach_context_build_seconds": context_seconds,
            "template_render_seconds": template_seconds,
            "total_seconds": time.perf_counter() - started_total,
        },
    }
    json.dumps(report, allow_nan=False)
    return report
