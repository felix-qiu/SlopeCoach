"""Artifact-only A7 diagnosis benchmark; never reruns upstream models."""

from __future__ import annotations

import json
from pathlib import Path

from slopecoach_ml.diagnosis import (
    DIAGNOSIS_BENCHMARK_CONTRACT_VERSION,
    DIAGNOSIS_RULE_REGISTRY_SHA256,
    RULE_REGISTRY,
    DiagnosisRuleConfig,
    diagnose_biomechanics,
)


def benchmark_diagnosis_artifact(
    path: str | Path, *, sport_type: str = "auto"
) -> dict[str, object]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if "biomechanics_result" not in artifact or "turn_segments" not in artifact:
        raise ValueError("ARTIFACT_MISSING_DIAGNOSIS_INPUT_FACTS")
    if sport_type == "auto":
        sport = artifact.get("sport_type")
        if not sport:
            raise ValueError("ARTIFACT_MISSING_EFFECTIVE_SPORT_TYPE")
    else:
        sport = {
            "effective_sport_type": sport_type.upper(),
            "effective_source": "USER",
        }
    result = diagnose_biomechanics(
        sport_type_result=sport,
        biomechanics_result=artifact["biomechanics_result"],
        turn_segments=artifact["turn_segments"],
    ).to_dict()
    evaluations = result["rule_evaluations"]
    per_rule = []
    for definition in RULE_REGISTRY:
        items = [
            item
            for item in evaluations
            if item["diagnosis_code"] == definition.diagnosis_code.value
        ]
        per_rule.append(
            {
                "diagnosis_code": definition.diagnosis_code.value,
                "eligible_turn_count": len(items),
                "evaluable_turn_count": sum(item["status"] != "NOT_EVALUABLE" for item in items),
                "triggered_turn_count": sum(item["status"] == "TRIGGERED" for item in items),
                "not_triggered_turn_count": sum(
                    item["status"] == "NOT_TRIGGERED" for item in items
                ),
                "not_evaluable_turn_count": sum(
                    item["status"] == "NOT_EVALUABLE" for item in items
                ),
            }
        )
    turns = artifact["turn_segments"]
    complete = sum(
        item.get("status") == "VALID"
        and item.get("start_timestamp_us") is not None
        and item.get("end_timestamp_us") is not None
        for item in turns
    )
    return {
        "benchmark_contract_version": DIAGNOSIS_BENCHMARK_CONTRACT_VERSION,
        "input_kind": "EXISTING_BIOMECHANICS_ARTIFACT",
        "sport_input": sport,
        "qualified_turn_count": len(turns),
        "complete_diagnosis_eligible_turn_count": complete,
        "partial_or_noneligible_turn_count": len(turns) - complete,
        "per_rule": per_rule,
        "total_provisional_diagnoses": len(result["diagnoses"]),
        "blocker_counts": {
            reason: result["blockers"].count(reason) for reason in sorted(set(result["blockers"]))
        },
        "diagnosis_rule_registry_sha256": DIAGNOSIS_RULE_REGISTRY_SHA256,
        "diagnosis_config": DiagnosisRuleConfig().to_dict(),
        "diagnosis_result": result,
        "ground_truth": {
            "DIAGNOSIS_GT_STATUS": "NOT_AVAILABLE",
            "TURN_SEGMENTATION_GT_STATUS": "NOT_AVAILABLE",
            "diagnosis_precision": None,
            "diagnosis_recall": None,
            "diagnosis_f1": None,
            "diagnosis_agreement": None,
        },
        "validation": {
            "A7_ENGINEERING_VALIDATION": "PASS_WITH_LIMITATIONS",
            "A7_REAL_DIAGNOSIS_VALIDATION": "NOT_VALIDATED_NO_DIAGNOSIS_GT",
            "A7_PRODUCT_VALIDATION": "BLOCKED_BY_TURN_AND_DIAGNOSIS_GT",
        },
    }
