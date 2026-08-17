"""Human-authored synthetic A9 Golden runner."""

from __future__ import annotations

import json
from pathlib import Path

from slopecoach_ml.coach import build_coach_report
from slopecoach_ml.scoring import build_scorecard
from slopecoach_ml.scoring.golden import diagnosis_from_golden_case

from .builder import build_analysis_result
from .projection import build_product_report


def build_golden_case(case: dict[str, object]):
    diagnosis_case = dict(case)
    sport = case.get("sport_type") or {}
    diagnosis_case["sport_type"] = sport.get("effective_sport_type", "SKI")
    diagnosis_case["sport_type_source"] = sport.get("effective_source", "AUTO")
    diagnosis = diagnosis_from_golden_case(diagnosis_case)
    provenance = diagnosis["diagnosis_semantics_provenance"]
    scorecard = build_scorecard(diagnosis, diagnosis_semantics_provenance=provenance).to_dict()
    coach = build_coach_report(diagnosis, diagnosis_semantics_provenance=provenance).to_dict()
    result = build_analysis_result(
        source=case.get("source"),
        target_identity=case.get("target_identity"),
        sport_type=case.get("sport_type"),
        turns=case.get("turns"),
        biomechanics=case.get("biomechanics"),
        diagnosis=diagnosis,
        diagnosis_semantics_provenance=provenance,
        scorecard=scorecard,
        coach=coach,
    )
    return result, build_product_report(result)


def run_analysis_result_golden(path: str | Path) -> dict[str, object]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for case in fixture["cases"]:
        result, product = build_golden_case(case)
        expected = case["expected"]
        passed = (
            result.quality_gate_status.value == expected["status"]
            and result.primary_reason_code == expected["primary_reason_code"]
            and product.status.value == expected["status"]
            and len(product.top_issues) == expected["top_issue_count"]
            and len(product.practice_plan) == expected["practice_item_count"]
            and product.numeric_scoring_enabled is False
            and product.overall_score is None
        )
        cases.append(
            {
                "case_id": case["case_id"],
                "passed": passed,
                "analysis_result": result.to_dict(),
                "product_report": product.to_dict(),
            }
        )
    return {
        "golden_contract_version": fixture["contract_version"],
        "golden_passed": all(case["passed"] for case in cases),
        "cases": cases,
    }
