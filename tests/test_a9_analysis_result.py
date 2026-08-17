from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from slopecoach_ml.analysis_result import (
    ANALYSIS_RESULT_CONTRACT_VERSION,
    ANALYSIS_SECTION_NAMES,
    ANALYSIS_SECTION_REGISTRY_SHA256,
    PRODUCT_REPORT_CONTRACT_VERSION,
    AnalysisSection,
    AnalysisSectionStatus,
    build_analysis_result,
    build_golden_case,
    build_product_report,
    run_analysis_result_golden,
)
from slopecoach_ml.analysis_result.fingerprint import semantic_sha256
from slopecoach_ml.analysis_result import projection
from slopecoach_ml.benchmark import benchmark_analysis_result_artifact
from slopecoach_ml.coach import build_coach_report
from slopecoach_ml.diagnosis import (
    DIAGNOSIS_RULE_REGISTRY_SHA256,
    build_diagnosis_semantics_provenance,
    validate_diagnosis_truth_consistency,
)
from slopecoach_ml.scoring import build_scorecard
from slopecoach_ml.scoring.golden import diagnosis_from_golden_case

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/golden_analysis_result_001.json"
CODES = (
    "LIMITED_KNEE_FLEXION_MODULATION_2D",
    "BILATERAL_KNEE_ASYMMETRY_2D",
    "KNEE_FLEXION_TIMING_OFFSET_2D",
)


def _case(case_id="ready-two-issues"):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return next(case for case in fixture["cases"] if case["case_id"] == case_id)


def _inputs(case_id="ready-two-issues"):
    case = _case(case_id)
    sport = case["sport_type"]
    diagnosis_case = {
        **case,
        "sport_type": sport["effective_sport_type"],
        "sport_type_source": sport["effective_source"],
    }
    diagnosis = diagnosis_from_golden_case(diagnosis_case)
    provenance = diagnosis["diagnosis_semantics_provenance"]
    scorecard = build_scorecard(
        diagnosis, diagnosis_semantics_provenance=provenance
    ).to_dict()
    coach = build_coach_report(
        diagnosis, diagnosis_semantics_provenance=provenance
    ).to_dict()
    return case, diagnosis, provenance, scorecard, coach


def _build(values):
    case, diagnosis, provenance, scorecard, coach = values
    return build_analysis_result(
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


def test_contract_versions_registry_and_golden_cases():
    report = run_analysis_result_golden(FIXTURE)
    assert report["golden_passed"]
    assert ANALYSIS_RESULT_CONTRACT_VERSION == "analysis-result-v1"
    assert PRODUCT_REPORT_CONTRACT_VERSION == "product-report-v1"
    assert ANALYSIS_SECTION_NAMES == (
        "SOURCE",
        "TARGET_IDENTITY",
        "SPORT_TYPE",
        "TURNS",
        "BIOMECHANICS",
        "DIAGNOSIS",
        "SCORECARD",
        "COACH",
    )
    assert len(ANALYSIS_SECTION_REGISTRY_SHA256) == 64


def test_raw_dict_config_and_registry_provenance_fail_closed():
    _, diagnosis, provenance, _, _ = _inputs()
    diagnosis["config"]["limited_knee_flexion_range_deg"] = 11.5
    with pytest.raises(ValueError, match="DIAGNOSIS_CONFIG_PROVENANCE_MISMATCH"):
        build_scorecard(diagnosis, diagnosis_semantics_provenance=provenance)
    old = build_diagnosis_semantics_provenance(
        diagnosis["config"], diagnosis_rule_registry_sha256="old-registry"
    ).to_dict()
    with pytest.raises(ValueError, match="DIAGNOSIS_RULE_REGISTRY_INCOMPATIBLE"):
        build_scorecard(diagnosis, diagnosis_semantics_provenance=old)


def test_unknown_code_and_diagnosis_entry_invariants_fail_closed():
    _, diagnosis, _, _, _ = _inputs()
    diagnosis["rule_evaluations"][0]["diagnosis_code"] = "FOO"
    diagnosis["diagnoses"][0]["diagnosis_code"] = "FOO"
    with pytest.raises(ValueError, match="DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT"):
        validate_diagnosis_truth_consistency(diagnosis)
    for field, value in (
        ("provisional", False),
        ("validation_status", "VALIDATED"),
        ("severity", "HIGH"),
        ("confidence", 0.9),
        ("affected_turn_ids", []),
    ):
        _, item, _, _, _ = _inputs()
        item["diagnoses"][0][field] = value
        with pytest.raises(ValueError, match="DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT"):
            validate_diagnosis_truth_consistency(item)


def test_scorecard_serialized_and_object_invariants_fail_closed():
    _, _, _, scorecard, coach = _inputs()
    tampered = copy.deepcopy(scorecard)
    tampered["diagnosis_semantics_provenance"]["diagnosis_config_sha256"] = "0" * 64
    tampered["diagnosis_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="DIAGNOSIS_CONFIG_SHA256_INVALID"):
        replace(build_coach_report(_inputs()[1]), scorecard=tampered)
    card = build_scorecard(_inputs()[1])
    with pytest.raises(ValueError):
        replace(card, diagnosis_dimension_registry_sha256="old")
    with pytest.raises(ValueError):
        replace(card, scoring_policy_version="old")
    stance = next(x for x in card.dimensions if x.dimension.value == "STANCE")
    with pytest.raises(ValueError, match="rule_count"):
        replace(stance, rule_count=2)
    assert coach["scorecard"] == scorecard


def test_section_shape_and_payload_sha_tamper_rejected():
    with pytest.raises(ValueError, match="MISSING_PAYLOAD"):
        AnalysisSection("SOURCE", AnalysisSectionStatus.AVAILABLE, "source-v1", None)
    with pytest.raises(ValueError, match="HAS_PAYLOAD"):
        AnalysisSection("SOURCE", AnalysisSectionStatus.UNAVAILABLE, None, {"x": 1})
    with pytest.raises(ValueError, match="UNKNOWN"):
        AnalysisSection("RANDOM_SECTION", AnalysisSectionStatus.UNAVAILABLE, None, None)
    with pytest.raises(ValueError, match="PAYLOAD_SHA256_INVALID"):
        AnalysisSection(
            "SOURCE",
            AnalysisSectionStatus.AVAILABLE,
            "source-v1",
            {"source_video_id": "x"},
            "0" * 64,
        )
    result = _build(_inputs())
    with pytest.raises(ValueError, match="REGISTRY_SHAPE_INVALID"):
        replace(result, sections=result.sections[:-1], analysis_result_sha256=None)


def test_cross_section_sport_provenance_scorecard_coach_and_issue_tamper():
    values = _inputs()
    values[0]["sport_type"]["effective_sport_type"] = "SNOWBOARD"
    with pytest.raises(ValueError, match="SPORT_TYPE_CONTRACT_INCONSISTENT"):
        _build(values)

    case, diagnosis, provenance, scorecard, coach = _inputs()
    bad_card = copy.deepcopy(scorecard)
    bad_card["limitations"].append("SEMANTIC_CHANGE")
    with pytest.raises(ValueError, match="SCORECARD_COACH_CONTRACT_INCONSISTENT"):
        _build((case, diagnosis, provenance, bad_card, coach))

    bad_coach = copy.deepcopy(coach)
    bad_coach["top_issues"][0]["diagnosis_code"] = "FOO"
    with pytest.raises(ValueError, match="COACH_ISSUE_DIAGNOSIS_INCONSISTENT"):
        _build((case, diagnosis, provenance, scorecard, bad_coach))

    custom = copy.deepcopy(provenance)
    custom["diagnosis_semantics_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        _build((case, diagnosis, custom, scorecard, coach))


def test_source_identity_mismatch_rejected():
    values = _inputs()
    values[0]["sport_type"]["source_video_id"] = "other"
    with pytest.raises(ValueError, match="ANALYSIS_SOURCE_IDENTITY_MISMATCH"):
        _build(values)


def test_inputs_immutable_deterministic_and_fingerprint_cascade():
    values = _inputs()
    original = copy.deepcopy(values)
    first = _build(values)
    second = _build(values)
    product1 = build_product_report(first)
    product2 = build_product_report(second)
    assert values == original
    assert first.to_json() == second.to_json()
    assert first.analysis_result_sha256 == second.analysis_result_sha256
    assert product1.product_report_sha256 == product2.product_report_sha256

    changed = _inputs()
    changed[3]["limitations"].append("SEMANTIC_CHANGE")
    changed[4]["scorecard"] = copy.deepcopy(changed[3])
    altered = _build(changed)
    altered_product = build_product_report(altered)
    assert altered.analysis_result_sha256 != first.analysis_result_sha256
    assert altered_product.product_report_sha256 != product1.product_report_sha256

    semantic = _inputs()
    semantic[1]["config"]["limited_knee_flexion_range_deg"] = 11.5
    semantic_provenance = build_diagnosis_semantics_provenance(
        semantic[1]["config"]
    ).to_dict()
    semantic_scorecard = build_scorecard(
        semantic[1], diagnosis_semantics_provenance=semantic_provenance
    ).to_dict()
    semantic_coach = build_coach_report(
        semantic[1], diagnosis_semantics_provenance=semantic_provenance
    ).to_dict()
    semantic_result = _build(
        (
            semantic[0],
            semantic[1],
            semantic_provenance,
            semantic_scorecard,
            semantic_coach,
        )
    )
    assert semantic_result.analysis_result_sha256 != first.analysis_result_sha256
    assert (
        build_product_report(semantic_result).product_report_sha256
        != product1.product_report_sha256
    )


def test_product_report_is_pure_projection_and_has_no_raw_fact_leak():
    values = _inputs()
    values[0]["source"]["local_artifact_path"] = "/tmp/private.mp4"
    values[0]["sport_type"]["frame_facts"] = [{"secret": True}]
    result = _build(values)
    product = build_product_report(result)
    source = inspect.getsource(projection)
    for forbidden_import in (
        "biomechanics.knee_angle",
        "turns.detector",
        "sport_type.fusion",
        "diagnosis.pipeline",
    ):
        assert forbidden_import not in source
    rendered = json.dumps(product.to_dict(), allow_nan=False)
    assert "/tmp/private.mp4" not in result.to_json()
    for forbidden in (
        "frame_facts",
        "pose_frames",
        "raw_keypoints",
        "turn_signal_samples",
        "detector_logits",
        "clip_logits",
        "raw_model_tensor",
    ):
        assert forbidden not in rendered.lower()
    assert product.status == result.quality_gate_status
    assert (
        product.headline
        == next(x.payload for x in result.sections if x.name == "COACH")["headline"]
    )
    assert product.scorecard["overall_score"] is None
    assert all(x["score_value"] is None for x in product.scorecard["dimensions"])


def test_unsafe_target_suppresses_product_recommendations():
    result, product = build_golden_case(_case("not-analyzable-target"))
    assert result.quality_gate_status.value == "NOT_ANALYZABLE"
    assert product.scorecard is None
    assert product.top_issues == ()
    assert product.practice_plan == ()


def test_runtime_and_local_path_are_outside_semantic_fingerprints():
    result = _build(_inputs())
    product = build_product_report(result)
    runtime_a = {"total_seconds": 0.1, "local_artifact_path": "/tmp/a.json"}
    runtime_b = {"total_seconds": 9.9, "local_artifact_path": "/other/b.json"}
    assert runtime_a != runtime_b
    assert result.analysis_result_sha256 == semantic_sha256(result.semantic_dict())
    assert product.product_report_sha256 == semantic_sha256(product.semantic_dict())

    values = _inputs()
    values[0]["source"] = {"local_artifact_path": "/tmp/not-an-identity.mp4"}
    missing_source = _build(values)
    source_section = next(x for x in missing_source.sections if x.name == "SOURCE")
    assert source_section.status.value == "UNAVAILABLE"
    assert missing_source.quality_gate_status.value == "PARTIAL_ANALYSIS"


def test_real_artifact_benchmark_is_partial_without_model_rerun():
    path = ROOT / "artifacts/benchmarks/a7/ski_test_001_artifact_only.json"
    report = benchmark_analysis_result_artifact(path)
    repeated = benchmark_analysis_result_artifact(path)
    assert report["REAL_A9_MODEL_RERUN"] is False
    assert report["quality_gate"] == "PARTIAL_ANALYSIS"
    assert report["primary_reason_code"] == "NO_QUALIFIED_TURNS"
    assert report["analysis_result"]["sections"][0]["status"] == "UNAVAILABLE"
    assert report["product_report"]["top_issues"] == []
    assert report["product_report"]["practice_plan"] == []
    assert report["product_report"]["overall_score"] is None
    assert report["analysis_result_sha256"] == repeated["analysis_result_sha256"]
    assert report["product_report_sha256"] == repeated["product_report_sha256"]
    assert report["performance"] != repeated["performance"]
    assert report["fingerprints"]["diagnosis_rule_registry_sha256"] == (
        DIAGNOSIS_RULE_REGISTRY_SHA256
    )
