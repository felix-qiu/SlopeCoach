"""A8.1 deterministic diagnosis and priority provenance Golden."""

from __future__ import annotations

import json
from pathlib import Path

from slopecoach_ml.diagnosis import build_diagnosis_semantics_provenance

from .contracts import IssuePriorityPolicy, issue_priority_policy_sha256


def run_a8_provenance_golden(path: str | Path) -> dict[str, object]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    semantics = [
        build_diagnosis_semantics_provenance(case["diagnosis_config"])
        for case in fixture["diagnosis_semantics_cases"]
    ]
    policies = [IssuePriorityPolicy(**case["policy"]) for case in fixture["policy_cases"]]
    semantic_results = [
        {
            "case_id": case["case_id"],
            "diagnosis_config_sha256": result.diagnosis_config_sha256,
            "diagnosis_semantics_sha256": result.diagnosis_semantics_sha256,
            "passed": result.to_dict()
            == build_diagnosis_semantics_provenance(case["diagnosis_config"]).to_dict(),
        }
        for case, result in zip(fixture["diagnosis_semantics_cases"], semantics, strict=True)
    ]
    policy_results = [
        {
            "case_id": case["case_id"],
            "policy": policy.to_dict(),
            "issue_priority_policy_sha256": issue_priority_policy_sha256(policy),
            "passed": policy.to_dict() == case["expected_policy"],
        }
        for case, policy in zip(fixture["policy_cases"], policies, strict=True)
    ]
    relationships_passed = (
        semantics[0].diagnosis_config_sha256 != semantics[1].diagnosis_config_sha256
        and semantics[0].diagnosis_semantics_sha256 != semantics[1].diagnosis_semantics_sha256
        and issue_priority_policy_sha256(policies[0]) != issue_priority_policy_sha256(policies[1])
    )
    return {
        "fixture_contract_version": fixture["contract_version"],
        "provenance_version": semantics[0].version,
        "semantic_cases": semantic_results,
        "policy_cases": policy_results,
        "expected_inequality_relationships_passed": relationships_passed,
        "golden_passed": relationships_passed
        and all(item["passed"] for item in (*semantic_results, *policy_results)),
    }
