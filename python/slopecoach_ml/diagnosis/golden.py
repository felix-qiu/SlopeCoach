"""Independent synthetic A7 diagnosis Golden runner."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .pipeline import diagnose_biomechanics


def run_diagnosis_golden(path: str | Path) -> dict[str, object]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    results = []
    for case in fixture["cases"]:
        timestamps = case["timestamps_us"]
        facts = []
        for timestamp, knee, asymmetry in zip(
            timestamps, case["knee_angles_deg"], case["knee_differences_deg"], strict=True
        ):
            for feature_id, value in (
                ("bilateral_knee_mean_angle_2d_deg", knee),
                ("bilateral_knee_abs_difference_2d_deg", asymmetry),
            ):
                facts.append(
                    {
                        "feature_id": feature_id,
                        "unit": "deg",
                        "value": value,
                        "status": "AVAILABLE",
                        "timestamp_us": timestamp,
                        "temporal_segment_id": 1,
                        "signal_run_id": 1,
                        "support_confidence": 0.9,
                        "observed_joint_count": 6,
                        "interpolated_joint_count": 0,
                    }
                )
        turn = {
            "turn_id": case["case_id"] + "-turn",
            "temporal_segment_id": 1,
            "signal_run_id": 1,
            "start_timestamp_us": timestamps[0] if case["turn_status"] == "VALID" else None,
            "apex_timestamp_us": timestamps[2],
            "end_timestamp_us": timestamps[-1] if case["turn_status"] == "VALID" else None,
            "status": case["turn_status"],
        }
        turn_features = {
            "turn_id": turn["turn_id"],
            "temporal_segment_id": 1,
            "signal_run_id": 1,
            "facts": [
                {
                    "feature_id": "minimum_mean_knee_angle_phase_offset",
                    "value": case["phase_offset"],
                    "status": "AVAILABLE"
                    if case["turn_status"] == "VALID"
                    else "TURN_BOUNDARY_UNAVAILABLE",
                },
                {
                    "feature_id": "minimum_mean_knee_angle_timestamp_us",
                    "value": timestamps[2],
                    "status": "AVAILABLE"
                    if case["turn_status"] == "VALID"
                    else "TURN_BOUNDARY_UNAVAILABLE",
                },
            ],
        }
        result = diagnose_biomechanics(
            sport_type_result={
                "effective_sport_type": case["sport_type"],
                "effective_source": "USER",
            },
            biomechanics_result={"frame_facts": facts, "turn_features": [turn_features]},
            turn_segments=[turn],
        ).to_dict()
        codes = [item["diagnosis_code"] for item in result["diagnoses"]]
        expected = case["expected"]
        statistics = {
            item["diagnosis_code"]: item["feature_evidence"][0]["value"]
            for item in result["rule_evaluations"]
            if item["feature_evidence"]
        }
        passed = (
            result["status"] == expected["result_status"]
            and codes == expected["diagnosis_codes"]
            and all(
                math.isclose(statistics[key], value, abs_tol=1e-12)
                for key, value in expected.get("statistics", {}).items()
            )
            and all(
                item["evidence_frames"] == expected["evidence_timestamps_us"]
                for item in result["rule_evaluations"]
                if item["status"] in {"TRIGGERED", "NOT_TRIGGERED"}
            )
        )
        results.append(
            {
                "case_id": case["case_id"],
                "passed": passed,
                "result_status": result["status"],
                "diagnosis_codes": codes,
            }
        )
    return {
        "fixture_contract_version": fixture["contract_version"],
        "golden_passed": all(item["passed"] for item in results),
        "cases": results,
    }
