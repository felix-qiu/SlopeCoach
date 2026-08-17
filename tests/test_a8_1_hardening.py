from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from slopecoach_ml.benchmark import benchmark_scoring_coach_artifact
from slopecoach_ml.coach import (
    COACH_TEMPLATE_REGISTRY_SHA256,
    LANGUAGE_POLICY,
    build_coach_context,
    build_coach_report,
    canonical_template_registry_json,
    run_coach_golden,
)
from slopecoach_ml.diagnosis import (
    DIAGNOSIS_RULE_REGISTRY_SHA256,
    DiagnosisResult,
    DiagnosisResultStatus,
    DiagnosisRuleConfig,
    build_diagnosis_semantics_provenance,
    diagnosis_config_sha256,
    validate_diagnosis_truth_consistency,
)
from slopecoach_ml.scoring import (
    IssuePriorityPolicy,
    build_scorecard,
    canonical_issue_priority_policy_json,
    issue_priority_policy_sha256,
    run_a8_provenance_golden,
)
from slopecoach_ml.scoring.golden import diagnosis_from_golden_case

ROOT = Path(__file__).resolve().parents[1]
CODES = (
    "LIMITED_KNEE_FLEXION_MODULATION_2D",
    "BILATERAL_KNEE_ASYMMETRY_2D",
    "KNEE_FLEXION_TIMING_OFFSET_2D",
)


def _diagnosis(statuses=None):
    statuses = statuses or (["TRIGGERED"], ["TRIGGERED"], ["TRIGGERED"])
    return diagnosis_from_golden_case(
        {
            "case_id": "hardening",
            "rule_statuses": dict(zip(CODES, statuses, strict=True)),
        }
    )


def _artifact(diagnosis=None, *, explicit=True):
    diagnosis = copy.deepcopy(diagnosis or _diagnosis())
    artifact = {
        "benchmark_contract_version": "ski-bench-diagnosis-v1",
        "diagnosis_rule_registry_sha256": DIAGNOSIS_RULE_REGISTRY_SHA256,
        "diagnosis_config": copy.deepcopy(diagnosis["config"]),
        "diagnosis_result": diagnosis,
        "qualified_turn_count": 1,
    }
    if explicit:
        artifact["diagnosis_semantics_provenance"] = copy.deepcopy(
            diagnosis["diagnosis_semantics_provenance"]
        )
    return artifact


def _write(tmp_path, artifact):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
    return path


def test_a8_1_provenance_golden_and_deterministic_fingerprints():
    result = run_a8_provenance_golden(ROOT / "fixtures/golden_a8_1_provenance_001.json")
    assert result["golden_passed"]
    default = DiagnosisRuleConfig()
    first = build_diagnosis_semantics_provenance(default)
    second = build_diagnosis_semantics_provenance(default)
    custom = build_diagnosis_semantics_provenance(
        replace(default, limited_knee_flexion_range_deg=11.5)
    )
    assert first == second
    assert first.diagnosis_config_sha256 != custom.diagnosis_config_sha256
    assert first.diagnosis_semantics_sha256 != custom.diagnosis_semantics_sha256
    assert diagnosis_config_sha256(default) == first.diagnosis_config_sha256
    json.dumps(first.to_dict(), allow_nan=False)


def test_explicit_and_legacy_artifacts_preserve_source_provenance(tmp_path):
    for explicit, origin in (
        (True, "EXPLICIT_SEMANTIC_PROVENANCE"),
        (False, "LEGACY_EXPLICIT_FIELDS_DERIVED"),
    ):
        artifact = _artifact(explicit=explicit)
        report = benchmark_scoring_coach_artifact(_write(tmp_path, artifact))
        provenance = report["diagnosis_semantics_provenance"]
        card = report["scorecard"]
        coach = report["coach_report"]
        assert report["input_compatibility"]["status"] == "COMPATIBLE"
        assert report["input_compatibility"]["diagnosis_provenance_origin"] == origin
        assert (
            card["diagnosis_rule_registry_sha256"]
            == provenance["diagnosis_rule_registry_sha256"]
        )
        assert card["diagnosis_config_sha256"] == provenance["diagnosis_config_sha256"]
        assert (
            card["diagnosis_semantics_sha256"]
            == provenance["diagnosis_semantics_sha256"]
        )
        assert (
            coach["diagnosis_semantics_provenance"]["diagnosis_semantics_sha256"]
            == provenance["diagnosis_semantics_sha256"]
        )
        assert coach["diagnosis_semantics_provenance"] == provenance


def test_custom_config_is_compatible_and_not_relabelled(tmp_path):
    diagnosis = _diagnosis()
    diagnosis["config"]["limited_knee_flexion_range_deg"] = 11.5
    provenance = build_diagnosis_semantics_provenance(diagnosis["config"]).to_dict()
    diagnosis["diagnosis_semantics_provenance"] = provenance
    artifact = _artifact(diagnosis)
    artifact["diagnosis_semantics_provenance"] = provenance
    report = benchmark_scoring_coach_artifact(_write(tmp_path, artifact))
    default = build_diagnosis_semantics_provenance(DiagnosisRuleConfig())
    assert (
        report["scorecard"]["diagnosis_config_sha256"]
        != default.diagnosis_config_sha256
    )
    assert (
        report["scorecard"]["diagnosis_semantics_sha256"]
        == provenance["diagnosis_semantics_sha256"]
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda artifact: artifact.update(benchmark_contract_version="old"),
            "DIAGNOSIS_BENCHMARK_CONTRACT_INCOMPATIBLE",
        ),
        (
            lambda artifact: artifact["diagnosis_result"].update(
                contract_version="old"
            ),
            "DIAGNOSIS_CONTRACT_INCOMPATIBLE",
        ),
        (
            lambda artifact: artifact.update(
                diagnosis_rule_registry_sha256="old-registry"
            ),
            "DIAGNOSIS_RULE_REGISTRY_PROVENANCE_MISMATCH",
        ),
        (
            lambda artifact: artifact["diagnosis_config"].update(
                limited_knee_flexion_range_deg=11.0
            ),
            "DIAGNOSIS_CONFIG_PROVENANCE_MISMATCH",
        ),
        (
            lambda artifact: artifact["diagnosis_semantics_provenance"].update(
                diagnosis_config_sha256="0" * 64
            ),
            "DIAGNOSIS_CONFIG_SHA256_INVALID",
        ),
        (
            lambda artifact: artifact["diagnosis_semantics_provenance"].update(
                diagnosis_semantics_sha256="0" * 64
            ),
            "DIAGNOSIS_SEMANTICS_SHA256_INVALID",
        ),
        (
            lambda artifact: (
                artifact.pop("diagnosis_semantics_provenance"),
                artifact.pop("diagnosis_config"),
            ),
            "DIAGNOSIS_SEMANTIC_PROVENANCE_MISSING",
        ),
    ],
)
def test_artifact_compatibility_fails_closed(tmp_path, mutation, error):
    artifact = _artifact()
    mutation(artifact)
    with pytest.raises(ValueError, match=error):
        benchmark_scoring_coach_artifact(_write(tmp_path, artifact))


def test_old_registry_never_becomes_current_registry(tmp_path):
    artifact = _artifact(explicit=False)
    artifact["diagnosis_rule_registry_sha256"] = "old-registry"
    with pytest.raises(ValueError, match="DIAGNOSIS_RULE_REGISTRY_INCOMPATIBLE"):
        benchmark_scoring_coach_artifact(_write(tmp_path, artifact))


def test_bare_dict_fails_but_explicit_dict_and_typed_config_work():
    diagnosis = _diagnosis()
    provenance = diagnosis.pop("diagnosis_semantics_provenance")
    with pytest.raises(ValueError, match="DIAGNOSIS_SEMANTIC_PROVENANCE_MISSING"):
        build_scorecard(diagnosis)
    card = build_scorecard(diagnosis, diagnosis_semantics_provenance=provenance)
    assert card.diagnosis_semantics_sha256 == provenance["diagnosis_semantics_sha256"]
    typed = DiagnosisResult(
        DiagnosisResultStatus.NOT_ANALYZABLE_NO_QUALIFIED_TURNS,
        "SKI",
        "USER",
        (),
        (),
        ("NO_QUALIFIED_TURNS",),
        DiagnosisRuleConfig(limited_knee_flexion_range_deg=11.5),
        ("PYTHON_RESEARCH_REFERENCE_ONLY",),
    )
    typed_card = build_scorecard(typed)
    assert typed_card.diagnosis_config_sha256 == diagnosis_config_sha256(typed.config)


@pytest.mark.parametrize("mode", ["missing_diagnosis", "false_diagnosis", "duplicate"])
def test_truth_consistency_fails_closed(mode):
    diagnosis = _diagnosis((["TRIGGERED"], ["NOT_TRIGGERED"], ["NOT_TRIGGERED"]))
    if mode == "missing_diagnosis":
        diagnosis["diagnoses"] = []
    elif mode == "false_diagnosis":
        diagnosis["rule_evaluations"][0]["status"] = "NOT_TRIGGERED"
    else:
        diagnosis["diagnoses"].append(copy.deepcopy(diagnosis["diagnoses"][0]))
    with pytest.raises(ValueError, match="DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT"):
        validate_diagnosis_truth_consistency(diagnosis)
    with pytest.raises(ValueError, match="DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT"):
        build_scorecard(diagnosis)


def test_complete_priority_policy_survives_context_report_and_benchmark(tmp_path):
    policy = IssuePriorityPolicy(max_top_issues=1)
    context = build_coach_context(_diagnosis(), issue_policy=policy)
    report = build_coach_report(context).to_dict()
    assert len(context.top_issues) == len(report["top_issues"]) == 1
    assert len(report["practice_plan"]) == 1
    assert context.to_dict()["issue_priority_policy"] == policy.to_dict()
    assert report["issue_priority_policy"] == policy.to_dict()
    assert report["issue_priority_policy_sha256"] == issue_priority_policy_sha256(
        policy
    )
    benchmark = benchmark_scoring_coach_artifact(
        _write(tmp_path, _artifact()), issue_policy=policy
    )
    assert benchmark["issue_priority_policy"] == policy.to_dict()
    assert benchmark["top_issue_count"] == benchmark["practice_item_count"] == 1
    default = IssuePriorityPolicy()
    assert issue_priority_policy_sha256(policy) != issue_priority_policy_sha256(default)
    assert (
        json.loads(canonical_issue_priority_policy_json(policy))["max_top_issues"] == 1
    )


def test_language_fingerprint_covers_all_static_language_not_runtime_values():
    old_sha = "4b7af86d4364b516cca265e6ff23b3f0f2704b80393f531157ba518a1fd7d549"
    assert COACH_TEMPLATE_REGISTRY_SHA256 != old_sha
    variants = (
        replace(LANGUAGE_POLICY, no_qualified_turns_headline="changed"),
        replace(LANGUAGE_POLICY, evidence_template="changed {triggered_turn_count}"),
        replace(
            LANGUAGE_POLICY,
            issue_templates=(
                replace(LANGUAGE_POLICY.issue_templates[0], title="changed"),
                *LANGUAGE_POLICY.issue_templates[1:],
            ),
        ),
        replace(LANGUAGE_POLICY, controlled_warnings=("changed",)),
    )
    for variant in variants:
        sha = hashlib.sha256(
            canonical_template_registry_json(variant).encode()
        ).hexdigest()
        assert sha != COACH_TEMPLATE_REGISTRY_SHA256
    first = build_coach_report(_diagnosis()).to_dict()
    second = build_coach_report(
        _diagnosis((["TRIGGERED", "TRIGGERED"], ["NOT_TRIGGERED"], ["NOT_TRIGGERED"]))
    ).to_dict()
    assert first["template_registry_sha256"] == second["template_registry_sha256"]
    assert first["headline"] == "这段视频有 2 个可以优先关注的动作信号"
    assert (
        build_coach_report(
            diagnosis_from_golden_case(
                {
                    "case_id": "none",
                    "upstream_status": "NOT_ANALYZABLE_NO_QUALIFIED_TURNS",
                    "rule_statuses": {},
                }
            )
        ).headline
        == "当前没有足够的完整转弯证据生成动作建议"
    )


def test_dimension_and_scorecard_negative_invariants():
    card = build_scorecard(_diagnosis())
    dimension = next(
        item for item in card.dimensions if item.dimension.value == "STANCE"
    )
    for kwargs in (
        {"score_value": 82},
        {"score_scale_min": 0},
        {"score_scale_max": 100},
        {"evaluable_rule_turn_count": 2},
        {"triggered_turn_ratio": 0.2},
        {"triggered_turn_ratio": math.nan},
        {"triggered_turn_ratio": math.inf},
    ):
        with pytest.raises(ValueError):
            replace(dimension, **kwargs)
    with pytest.raises(ValueError):
        replace(card, overall_score=80)
    with pytest.raises(ValueError):
        replace(card, numeric_scoring_enabled=True)
    with pytest.raises(ValueError):
        replace(card, dimensions=card.dimensions[:-1])
    with pytest.raises(ValueError):
        replace(card, dimensions=(*card.dimensions[:-1], card.dimensions[0]))


def test_issue_negative_invariants_and_evidence_duplicates():
    issue = build_coach_context(_diagnosis()).all_issue_summaries[0]
    for kwargs in (
        {"severity": "HIGH"},
        {"confidence": 0.9},
        {"triggered_turn_count": True},
        {"evaluable_turn_count": 0},
        {"triggered_turn_count": issue.evaluable_turn_count + 1},
        {"triggered_turn_ratio": 0.5},
        {"triggered_turn_ratio": math.nan},
        {"triggered_turn_ratio": math.inf},
        {"affected_turn_ids": ("turn-1", "turn-1")},
    ):
        with pytest.raises(ValueError):
            replace(issue, **kwargs)


def test_coach_and_drill_negative_invariants():
    context = build_coach_context(_diagnosis())
    bad_card = copy.deepcopy(context.scorecard)
    bad_card["dimensions"][0]["score_value"] = 100
    with pytest.raises(ValueError, match="numeric score leakage"):
        replace(context, scorecard=bad_card)
    with pytest.raises(ValueError, match="exceeds issue priority policy"):
        replace(context, top_issues=context.all_issue_summaries)
    report = build_coach_report(context)
    with pytest.raises(ValueError):
        replace(report, language="en-US")
    with pytest.raises(ValueError):
        replace(
            report,
            practice_plan=(
                *report.practice_plan,
                {"diagnosis_code": "NOT_A_TOP_ISSUE"},
            ),
        )
    drill = report.practice_plan[0]["drill"]
    from slopecoach_ml.coach import ControlledDrill

    with pytest.raises(ValueError):
        ControlledDrill(**{**drill, "safety_notes": ("KEEP_CLEAR_SPACE_AROUND",)})


def test_existing_goldens_remain_behavior_compatible():
    result = run_coach_golden(ROOT / "fixtures/golden_coach_001.json")
    assert result["golden_passed"]
    assert result["coach_template_registry_sha256"] == COACH_TEMPLATE_REGISTRY_SHA256
