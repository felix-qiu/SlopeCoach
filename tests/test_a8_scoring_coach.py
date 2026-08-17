from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from slopecoach_ml.benchmark import benchmark_scoring_coach_artifact
from slopecoach_ml.benchmark.diagnosis import benchmark_diagnosis_artifact
from slopecoach_ml.diagnosis import DIAGNOSIS_RULE_REGISTRY_SHA256
from slopecoach_ml.coach import (
    COACH_TEMPLATE_REGISTRY_SHA256,
    DRILL_LIBRARY,
    DRILL_LIBRARY_SHA256,
    build_coach_report,
    canonical_drill_library_json,
    canonical_template_registry_json,
    drill_for_diagnosis,
    run_coach_golden,
)
from slopecoach_ml.scoring import (
    DIAGNOSIS_DIMENSION_REGISTRY,
    DIAGNOSIS_DIMENSION_REGISTRY_SHA256,
    IssuePriorityPolicy,
    ScoringPolicy,
    build_scorecard,
    canonical_dimension_registry_json,
    run_scorecard_golden,
)
from slopecoach_ml.scoring.golden import diagnosis_from_golden_case

ROOT = Path(__file__).resolve().parents[1]
CODES = [item.diagnosis_code for item in DIAGNOSIS_DIMENSION_REGISTRY]


def _case(statuses, **extra):
    return {
        "case_id": "test",
        "rule_statuses": dict(zip(CODES, statuses, strict=True)),
        **extra,
    }


def _dimensions(card):
    return {item["dimension"]: item for item in card["dimensions"]}


def test_scorecard_and_coach_goldens():
    assert run_scorecard_golden(ROOT / "fixtures/golden_scorecard_001.json")[
        "golden_passed"
    ]
    assert run_coach_golden(ROOT / "fixtures/golden_coach_001.json")["golden_passed"]


@pytest.mark.parametrize(
    "statuses",
    [
        (["NOT_TRIGGERED"], ["NOT_TRIGGERED"], ["NOT_TRIGGERED"]),
        (["TRIGGERED"], ["TRIGGERED"], ["TRIGGERED"]),
        (["NOT_EVALUABLE"], ["NOT_EVALUABLE"], ["NOT_EVALUABLE"]),
    ],
)
def test_every_numeric_score_is_null(statuses):
    card = build_scorecard(diagnosis_from_golden_case(_case(statuses))).to_dict()
    assert card["numeric_scoring_enabled"] is False
    assert card["overall_score"] is None
    assert all(item["score_value"] is None for item in card["dimensions"])
    json.dumps(card, allow_nan=False)


def test_balance_and_edge_are_not_invented_and_missing_is_visible():
    card = build_scorecard(
        diagnosis_from_golden_case(
            _case((["TRIGGERED"], ["NOT_TRIGGERED"], ["NOT_EVALUABLE"]))
        )
    ).to_dict()
    dimensions = _dimensions(card)
    assert dimensions["BALANCE"]["status"] == "NOT_IMPLEMENTED"
    assert dimensions["EDGE_CONTROL"]["status"] == "NOT_IMPLEMENTED"
    assert dimensions["TIMING"]["status"] == "NOT_EVALUABLE"
    assert dimensions["TIMING"]["status"] != "NO_PROVISIONAL_ISSUE_DETECTED"


def test_no_trigger_is_not_good_form_and_has_no_drill():
    report = build_coach_report(
        diagnosis_from_golden_case(
            _case((["NOT_TRIGGERED"], ["NOT_TRIGGERED"], ["NOT_TRIGGERED"]))
        )
    ).to_dict()
    assert report["status"] == "EXECUTED_NO_PROVISIONAL_ISSUES"
    assert report["top_issues"] == []
    assert report["practice_plan"] == []
    assert "NO_TRIGGER_DOES_NOT_MEAN_GOOD_FORM" in report["limitations"]
    assert all(
        term not in report["headline"] for term in ("GOOD_FORM", "技术很好", "没有问题")
    )


def test_top_two_tie_order_recurrence_and_null_severity_confidence():
    tied = diagnosis_from_golden_case(
        _case((["TRIGGERED"], ["TRIGGERED"], ["TRIGGERED"]))
    )
    first = build_coach_report(tied).to_dict()
    second = build_coach_report(tied).to_dict()
    assert [item["diagnosis_code"] for item in first["top_issues"]] == CODES[:2]
    assert len(first["all_issue_summaries"]) == 3
    assert len(first["practice_plan"]) == 2
    assert first == second
    assert all(
        item["severity"] is None and item["confidence"] is None
        for item in first["all_issue_summaries"]
    )
    recurrence = diagnosis_from_golden_case(
        _case(
            (
                ["TRIGGERED", "NOT_TRIGGERED", "NOT_TRIGGERED", "NOT_TRIGGERED"],
                ["TRIGGERED", "TRIGGERED", "TRIGGERED", "NOT_TRIGGERED"],
                ["NOT_TRIGGERED"] * 4,
            )
        )
    )
    assert (
        build_coach_report(recurrence).to_dict()["top_issues"][0]["diagnosis_code"]
        == CODES[1]
    )


def test_diagnosis_is_read_only_and_coach_cannot_invent_issue():
    diagnosis = diagnosis_from_golden_case(
        _case((["TRIGGERED"], ["NOT_TRIGGERED"], ["NOT_TRIGGERED"]))
    )
    original = copy.deepcopy(diagnosis)
    build_scorecard(diagnosis)
    build_coach_report(diagnosis)
    assert diagnosis == original
    diagnosis["diagnoses"] = []
    diagnosis["tempting_raw_feature"] = "knee range below threshold"
    with pytest.raises(ValueError, match="DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT"):
        build_coach_report(diagnosis)


def test_upstream_not_analyzable_and_limitations_are_preserved():
    diagnosis = diagnosis_from_golden_case(
        _case(
            ([], [], []),
            upstream_status="NOT_ANALYZABLE_NO_QUALIFIED_TURNS",
            sport_type_source="AUTO",
        )
    )
    report = build_coach_report(diagnosis).to_dict()
    assert report["status"] == "NOT_ANALYZABLE_UPSTREAM"
    assert report["top_issues"] == report["practice_plan"] == []
    assert "AUTO_SPORT_TYPE_NOT_PRODUCT_VALIDATED" in report["limitations"]
    assert report["scorecard"]["overall_score"] is None


def test_drill_contract_safety_applicability_and_fingerprints():
    assert len(DRILL_LIBRARY) == 3
    for drill in DRILL_LIBRARY:
        assert drill.practice_environment == "EASY_CONTROLLED_TERRAIN"
        assert "PRACTICE_WITHIN_CURRENT_ABILITY" in drill.safety_notes
        assert "STOP_IF_PAIN_OR_LOSS_OF_CONTROL" in drill.safety_notes
    with pytest.raises(ValueError, match="not applicable"):
        drill_for_diagnosis(CODES[0], "UNKNOWN")
    assert (
        hashlib.sha256(canonical_drill_library_json().encode()).hexdigest()
        == DRILL_LIBRARY_SHA256
    )
    assert (
        hashlib.sha256(canonical_template_registry_json().encode()).hexdigest()
        == COACH_TEMPLATE_REGISTRY_SHA256
    )
    assert (
        hashlib.sha256(canonical_dimension_registry_json().encode()).hexdigest()
        == DIAGNOSIS_DIMENSION_REGISTRY_SHA256
    )
    changed = replace(DIAGNOSIS_DIMENSION_REGISTRY[0], limitations=("CHANGED",))
    assert (
        canonical_dimension_registry_json((changed, *DIAGNOSIS_DIMENSION_REGISTRY[1:]))
        != canonical_dimension_registry_json()
    )


@pytest.mark.parametrize("value", [True, False, 0, 3])
def test_issue_priority_policy_is_strict(value):
    with pytest.raises(ValueError):
        IssuePriorityPolicy(max_top_issues=value)


def test_scoring_policy_cannot_enable_numeric_score():
    with pytest.raises(ValueError):
        ScoringPolicy(numeric_scoring_enabled=True)


def test_artifact_only_benchmark_and_missing_contract(tmp_path):
    diagnosis = diagnosis_from_golden_case(
        _case(([], [], []), upstream_status="NOT_ANALYZABLE_NO_QUALIFIED_TURNS")
    )
    artifact = tmp_path / "a7.json"
    artifact.write_text(
        json.dumps(
            {
                "benchmark_contract_version": "ski-bench-diagnosis-v1",
                "diagnosis_rule_registry_sha256": DIAGNOSIS_RULE_REGISTRY_SHA256,
                "diagnosis_config": diagnosis["config"],
                "diagnosis_semantics_provenance": diagnosis[
                    "diagnosis_semantics_provenance"
                ],
                "diagnosis_result": diagnosis,
            }
        ),
        encoding="utf-8",
    )
    report = benchmark_scoring_coach_artifact(artifact)
    assert report["benchmark_contract_version"] == "ski-bench-scoring-coach-v1"
    assert report["top_issue_count"] == report["practice_item_count"] == 0
    assert report["ground_truth"]["score_mae"] is None
    assert report["scorecard"]["overall_score"] is None
    json.dumps(report, allow_nan=False)
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="DIAGNOSIS_BENCHMARK_CONTRACT_INCOMPATIBLE"):
        benchmark_scoring_coach_artifact(invalid)


def test_a7_rejected_turn_counts_and_blocker_occurrences(tmp_path):
    rejected = [
        {
            "turn_id": f"rejected-{index}",
            "status": "REJECTED_LOW_PROMINENCE",
            "apex_timestamp_us": index,
            "start_timestamp_us": None,
            "end_timestamp_us": None,
        }
        for index in range(10)
    ]
    path = tmp_path / "rejected.json"
    path.write_text(
        json.dumps(
            {
                "biomechanics_result": {"frame_facts": [], "turn_features": []},
                "turn_segments": rejected,
            }
        ),
        encoding="utf-8",
    )
    report = benchmark_diagnosis_artifact(path, sport_type="ski")
    assert report["turn_candidate_count"] == 10
    assert report["qualified_turn_count"] == 0
    assert report["rejected_turn_count"] == 10

    partial = {
        "turn_id": "partial",
        "status": "PARTIAL",
        "apex_timestamp_us": 100,
        "start_timestamp_us": None,
        "end_timestamp_us": 200,
        "temporal_segment_id": 1,
        "signal_run_id": 1,
    }
    path.write_text(
        json.dumps(
            {
                "biomechanics_result": {"frame_facts": [], "turn_features": []},
                "turn_segments": [partial],
            }
        ),
        encoding="utf-8",
    )
    report = benchmark_diagnosis_artifact(path, sport_type="ski")
    assert report["blocker_counts"]["TURN_BOUNDARY_UNAVAILABLE"] == 3
    timing = next(
        item
        for item in report["diagnosis_result"]["rule_evaluations"]
        if item["diagnosis_code"] == "KNEE_FLEXION_TIMING_OFFSET_2D"
    )
    assert timing["phase"] == "APEX_RELATIVE_TIMING"
    assert report["diagnosis_result"]["DIAGNOSIS_SEVERITY_STATUS"] == "NOT_CALIBRATED"
    assert report["diagnosis_result"]["SEVERITY_STATUS"] == "NOT_CALIBRATED"
    provenance = report["diagnosis_semantics_provenance"]
    assert provenance["diagnosis_config"] == report["diagnosis_config"]
    assert provenance["diagnosis_config"] == report["diagnosis_result"]["config"]
    assert (
        provenance["diagnosis_rule_registry_sha256"]
        == report["diagnosis_rule_registry_sha256"]
    )
