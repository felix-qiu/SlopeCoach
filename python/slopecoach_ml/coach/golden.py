"""Synthetic controlled-coach A8 Golden runner."""

from __future__ import annotations

import json
from pathlib import Path

from slopecoach_ml.scoring.golden import diagnosis_from_golden_case

from .drills import DRILL_LIBRARY_SHA256
from .pipeline import build_coach_report
from .templates import COACH_TEMPLATE_REGISTRY_SHA256


def run_coach_golden(path: str | Path) -> dict[str, object]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    results = []
    forbidden = ("GOOD_FORM", "满分", "严重", "完美", "准确率")
    for case in fixture["cases"]:
        report = build_coach_report(diagnosis_from_golden_case(case)).to_dict()
        user_text = json.dumps(
            {
                "headline": report["headline"],
                "practice_plan": report["practice_plan"],
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        top_codes = [item["diagnosis_code"] for item in report["top_issues"]]
        drill_ids = [item["drill"]["drill_id"] for item in report["practice_plan"]]
        expected = case["expected"]
        passed = (
            report["status"] == expected["status"]
            and top_codes == expected["top_issue_codes"]
            and drill_ids == expected["drill_ids"]
            and not any(term in user_text for term in forbidden)
            and all(
                item["severity"] is None and item["confidence"] is None
                for item in report["all_issue_summaries"]
            )
        )
        if report["practice_plan"]:
            passed = passed and all(
                "研究性" in item["why_this_focus"] and "2D" in item["why_this_focus"]
                for item in report["practice_plan"]
            )
        results.append(
            {
                "case_id": case["case_id"],
                "passed": passed,
                "status": report["status"],
                "top_issue_codes": top_codes,
                "drill_ids": drill_ids,
            }
        )
    return {
        "fixture_contract_version": fixture["contract_version"],
        "coach_report_contract_version": "coach-report-v1",
        "coach_context_version": "coach-context-v1",
        "coach_template_version": "coach-template-zh-cn-v1",
        "coach_template_registry_sha256": COACH_TEMPLATE_REGISTRY_SHA256,
        "drill_library_version": "drill-library-v1",
        "drill_library_sha256": DRILL_LIBRARY_SHA256,
        "golden_passed": all(item["passed"] for item in results),
        "cases": results,
    }
