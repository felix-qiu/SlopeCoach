"""Synthetic structure-only A8 ScoreCard Golden runner."""

from __future__ import annotations

import json
import math
from pathlib import Path

from slopecoach_ml.diagnosis import (
    DiagnosisRuleConfig,
    build_diagnosis_semantics_provenance,
)

from .registry import DIAGNOSIS_DIMENSION_REGISTRY_SHA256
from .scorecard import build_scorecard

_CODES = (
    "LIMITED_KNEE_FLEXION_MODULATION_2D",
    "BILATERAL_KNEE_ASYMMETRY_2D",
    "KNEE_FLEXION_TIMING_OFFSET_2D",
)


def diagnosis_from_golden_case(case: dict[str, object]) -> dict[str, object]:
    evaluations = []
    diagnoses = []
    statuses = case.get("rule_statuses", {})
    for code_index, code in enumerate(_CODES):
        for turn_index, status in enumerate(statuses.get(code, [])):
            turn_id = f"turn-{turn_index + 1}"
            evaluation = {
                "diagnosis_code": code,
                "turn_id": turn_id,
                "turn_apex_timestamp_us": (turn_index + 1) * 1_000_000,
                "status": status,
                "triggered": status == "TRIGGERED",
                "phase": "APEX_RELATIVE_TIMING" if code_index == 2 else "FULL_TURN",
                "reason_codes": ["SYNTHETIC_MISSING_EVIDENCE"] if status == "NOT_EVALUABLE" else [],
                "evidence_frames": [turn_index * 100 + 1, turn_index * 100 + 2]
                if status != "NOT_EVALUABLE"
                else [],
                "feature_evidence": []
                if status == "NOT_EVALUABLE"
                else [{"feature_id": f"feature-{code_index + 1}"}],
                "limitations": ["IMAGE_SPACE_2D_ONLY"],
            }
            evaluations.append(evaluation)
            if status == "TRIGGERED":
                diagnoses.append(
                    {
                        "diagnosis_code": code,
                        "evaluation_status": "TRIGGERED",
                        "affected_turn_ids": [turn_id],
                        "severity": None,
                        "confidence": None,
                    }
                )
    source = case.get("sport_type_source", "USER")
    limitation = (
        "SPORT_TYPE_USER_SELECTED_ROUTING_NOT_GT"
        if source == "USER"
        else "AUTO_SPORT_TYPE_NOT_PRODUCT_VALIDATED"
    )
    config = DiagnosisRuleConfig().to_dict()
    provenance = build_diagnosis_semantics_provenance(config).to_dict()
    return {
        "contract_version": "diagnosis-v1",
        "status": case.get(
            "upstream_status",
            "EXECUTED_WITH_PROVISIONAL_DIAGNOSES"
            if diagnoses
            else "EXECUTED_NO_PROVISIONAL_RULES_TRIGGERED",
        ),
        "sport_type": case.get("sport_type", "SKI"),
        "sport_type_source": source,
        "rule_evaluations": evaluations,
        "diagnoses": diagnoses,
        "blockers": [],
        "config": config,
        "diagnosis_semantics_provenance": provenance,
        "limitations": ["IMAGE_SPACE_2D_ONLY", limitation],
    }


def run_scorecard_golden(path: str | Path) -> dict[str, object]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    results = []
    for case in fixture["cases"]:
        card = build_scorecard(diagnosis_from_golden_case(case)).to_dict()
        dimensions = {item["dimension"]: item for item in card["dimensions"]}
        expected = case["expected_dimension_statuses"]
        passed = (
            all(dimensions[key]["status"] == value for key, value in expected.items())
            and all(item["score_value"] is None for item in card["dimensions"])
            and card["overall_score"] is None
            and card["numeric_scoring_enabled"] is False
        )
        for dimension, ratio in case.get("expected_trigger_ratios", {}).items():
            passed = passed and math.isclose(
                dimensions[dimension]["triggered_turn_ratio"], ratio, abs_tol=1e-12
            )
        results.append(
            {
                "case_id": case["case_id"],
                "passed": passed,
                "dimension_statuses": {key: value["status"] for key, value in dimensions.items()},
            }
        )
    return {
        "fixture_contract_version": fixture["contract_version"],
        "scorecard_contract_version": "scorecard-v1",
        "scoring_policy_version": "scoring-policy-v1",
        "diagnosis_dimension_registry_version": "diagnosis-dimension-registry-v1",
        "diagnosis_dimension_registry_sha256": DIAGNOSIS_DIMENSION_REGISTRY_SHA256,
        "golden_passed": all(item["passed"] for item in results),
        "cases": results,
    }
