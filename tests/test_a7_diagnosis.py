from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from slopecoach_ml.diagnosis import (
    DIAGNOSIS_RULE_REGISTRY_SHA256,
    RULE_REGISTRY,
    DiagnosisRuleConfig,
    canonical_rule_registry_json,
    diagnose_biomechanics,
    run_diagnosis_golden,
)
from slopecoach_ml.diagnosis.evidence import collect_turn_feature_evidence
from slopecoach_ml.benchmark.diagnosis import benchmark_diagnosis_artifact
from slopecoach_ml.sport_type.calibration.artifact import (
    artifact_fingerprint,
    fit_calibration_artifact,
)
from slopecoach_ml.sport_type.calibration.contracts import SportCalibrationFitConfig
from slopecoach_ml.sport_type.calibration.dataset import semantic_dataset_payload
from slopecoach_ml.sport_type.calibration.dataset import build_calibration_dataset
from slopecoach_ml.sport_type.calibration.fusion import apply_calibrated_fusion


TIMESTAMPS = (0, 250_000, 500_000, 750_000, 1_000_000)


def _fact(
    feature, timestamp, value, *, segment=1, run=1, available=True, interpolated=0
):
    return {
        "feature_id": feature,
        "unit": "deg",
        "value": value if available else None,
        "status": "AVAILABLE" if available else "INSUFFICIENT_EVIDENCE",
        "timestamp_us": timestamp,
        "temporal_segment_id": segment,
        "signal_run_id": run,
        "support_confidence": 0.8 if available else None,
        "observed_joint_count": 6 - interpolated,
        "interpolated_joint_count": interpolated,
    }


def _inputs(knees=(90, 80, 70, 80, 90), diffs=(2, 3, 4, 3, 2), offset=0.0):
    frame_facts = []
    for timestamp, knee, diff in zip(TIMESTAMPS, knees, diffs, strict=True):
        frame_facts.extend(
            (
                _fact("bilateral_knee_mean_angle_2d_deg", timestamp, knee),
                _fact("bilateral_knee_abs_difference_2d_deg", timestamp, diff),
            )
        )
    turn = {
        "turn_id": "turn-1",
        "temporal_segment_id": 1,
        "signal_run_id": 1,
        "start_timestamp_us": 0,
        "apex_timestamp_us": 500_000,
        "end_timestamp_us": 1_000_000,
        "status": "VALID",
    }
    turn_features = {
        "turn_id": "turn-1",
        "temporal_segment_id": 1,
        "signal_run_id": 1,
        "facts": [
            {
                "feature_id": "minimum_mean_knee_angle_phase_offset",
                "value": offset,
                "status": "AVAILABLE",
            },
            {
                "feature_id": "minimum_mean_knee_angle_timestamp_us",
                "value": 500_000,
                "status": "AVAILABLE",
            },
        ],
    }
    return frame_facts, turn, turn_features


def _run(
    *,
    sport="SKI",
    source="USER",
    knees=(90, 80, 70, 80, 90),
    diffs=(2, 3, 4, 3, 2),
    offset=0.0,
    mutate=None,
):
    facts, turn, turn_features = _inputs(knees, diffs, offset)
    if mutate:
        mutate(facts, turn, turn_features)
    return diagnose_biomechanics(
        sport_type_result={"effective_sport_type": sport, "effective_source": source},
        biomechanics_result={"frame_facts": facts, "turn_features": [turn_features]},
        turn_segments=[turn],
    ).to_dict()


def _codes(result):
    return [item["diagnosis_code"] for item in result["diagnoses"]]


def test_golden_and_exact_three_rule_registry():
    root = Path(__file__).resolve().parents[1]
    assert run_diagnosis_golden(root / "fixtures/golden_diagnosis_001.json")[
        "golden_passed"
    ]
    assert [item.diagnosis_code.value for item in RULE_REGISTRY] == [
        "LIMITED_KNEE_FLEXION_MODULATION_2D",
        "BILATERAL_KNEE_ASYMMETRY_2D",
        "KNEE_FLEXION_TIMING_OFFSET_2D",
    ]
    assert all(
        code not in canonical_rule_registry_json()
        for code in ("BACK_SEAT", "LATE_EDGE", "GOOD_FORM")
    )


def test_rule_threshold_boundaries_and_no_good_form():
    range_boundary = _run(knees=(80, 83, 92, 86, 82))
    assert not _codes(range_boundary)
    assert range_boundary["status"] == "EXECUTED_NO_PROVISIONAL_RULES_TRIGGERED"
    asymmetry = _run(diffs=(10, 10, 10, 10, 10))
    assert _codes(asymmetry) == ["BILATERAL_KNEE_ASYMMETRY_2D"]
    timing = _run(offset=0.20)
    assert "KNEE_FLEXION_TIMING_OFFSET_2D" not in _codes(timing)
    assert "GOOD_FORM" not in json.dumps(range_boundary)


def test_multiframe_and_missing_are_not_evaluable():
    def only_one(facts, _turn, _features):
        for fact in facts:
            if (
                fact["feature_id"] == "bilateral_knee_mean_angle_2d_deg"
                and fact["timestamp_us"] != 500_000
            ):
                fact.update(
                    value=None, status="INSUFFICIENT_EVIDENCE", support_confidence=None
                )

    result = _run(knees=(80, 80, 80, 80, 80), mutate=only_one)
    evaluation = next(
        item
        for item in result["rule_evaluations"]
        if item["diagnosis_code"] == "LIMITED_KNEE_FLEXION_MODULATION_2D"
    )
    assert evaluation["status"] == "NOT_EVALUABLE"
    assert evaluation["reason_codes"] == ["INSUFFICIENT_FEATURE_SAMPLES"]
    assert not result["diagnoses"]


def test_window_segment_and_run_isolation_and_interpolation_counts():
    def add_foreign(facts, _turn, _features):
        facts.extend(
            [
                _fact("bilateral_knee_mean_angle_2d_deg", 1_000_001, 0),
                _fact("bilateral_knee_mean_angle_2d_deg", 500_000, 0, segment=2),
                _fact("bilateral_knee_mean_angle_2d_deg", 600_000, 0, run=2),
            ]
        )
        facts[0]["interpolated_joint_count"] = 1
        facts[0]["observed_joint_count"] = 5

    result = _run(mutate=add_foreign)
    evaluation = result["rule_evaluations"][0]
    assert evaluation["feature_evidence"][0]["value"] == 20
    assert evaluation["evidence_frames"] == list(TIMESTAMPS)
    assert evaluation["feature_evidence"][0]["interpolated_sample_count"] == 1


def test_exact_turn_boundary_and_cross_turn_isolation():
    turn = {
        "turn_id": "window",
        "temporal_segment_id": 1,
        "signal_run_id": 1,
        "start_timestamp_us": 100,
        "apex_timestamp_us": 300,
        "end_timestamp_us": 500,
        "status": "VALID",
    }
    facts = [
        _fact("bilateral_knee_mean_angle_2d_deg", timestamp, value)
        for timestamp, value in (
            (99, 0),
            (100, 80),
            (200, 81),
            (300, 82),
            (400, 83),
            (500, 84),
            (501, 0),
        )
    ]
    evidence = collect_turn_feature_evidence(
        turn=turn, frame_facts=facts, feature_id="bilateral_knee_mean_angle_2d_deg"
    )
    assert evidence["evidence_timestamps_us"] == [100, 200, 300, 400, 500]

    first_facts, first_turn, first_features = _inputs((80, 81, 82, 83, 84))
    second_facts = [
        {**fact, "timestamp_us": fact["timestamp_us"] + 2_000_000}
        for fact in _inputs((90, 80, 70, 80, 90))[0]
    ]
    second_turn = {
        **first_turn,
        "turn_id": "turn-2",
        "start_timestamp_us": 2_000_000,
        "apex_timestamp_us": 2_500_000,
        "end_timestamp_us": 3_000_000,
    }
    second_features = {**first_features, "turn_id": "turn-2"}
    result = diagnose_biomechanics(
        sport_type_result={"effective_sport_type": "SKI", "effective_source": "USER"},
        biomechanics_result={
            "frame_facts": first_facts + second_facts,
            "turn_features": [first_features, second_features],
        },
        turn_segments=[first_turn, second_turn],
    ).to_dict()
    limited = [
        item["affected_turn_ids"][0]
        for item in result["diagnoses"]
        if item["diagnosis_code"] == "LIMITED_KNEE_FLEXION_MODULATION_2D"
    ]
    assert limited == ["turn-1"]


def test_partial_unknown_user_and_auto_routing_policy():
    partial = _run(
        mutate=lambda _facts, turn, _features: turn.update(
            status="PARTIAL", start_timestamp_us=None
        )
    )
    assert partial["status"] == "NOT_ANALYZABLE_INSUFFICIENT_DIAGNOSIS_EVIDENCE"
    assert all(
        item["status"] == "NOT_EVALUABLE" for item in partial["rule_evaluations"]
    )
    unknown = _run(sport="UNKNOWN")
    assert unknown["status"] == "NOT_ANALYZABLE_SPORT_TYPE_UNKNOWN"
    assert not unknown["diagnoses"]
    calibrated_only = diagnose_biomechanics(
        sport_type_result={
            "effective_sport_type": "UNKNOWN",
            "effective_source": "AUTO",
            "calibrated_fusion_result": {
                "sport_type": "SNOWBOARD",
                "probability": 0.99,
            },
        },
        biomechanics_result={"frame_facts": [], "turn_features": []},
        turn_segments=[],
    ).to_dict()
    assert calibrated_only["status"] == "NOT_ANALYZABLE_SPORT_TYPE_UNKNOWN"
    assert (
        "SPORT_TYPE_USER_SELECTED_ROUTING_NOT_GT"
        in _run(sport="SNOWBOARD")["limitations"]
    )
    assert "AUTO_SPORT_TYPE_NOT_PRODUCT_VALIDATED" in _run(source="AUTO")["limitations"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"minimum_turn_feature_samples": True},
        {"minimum_turn_feature_coverage": False},
        {"limited_knee_flexion_range_deg": math.nan},
        {"knee_asymmetry_median_deg": math.inf},
        {"knee_flexion_phase_offset_abs": 0},
    ],
)
def test_diagnosis_config_strictness(overrides):
    with pytest.raises(ValueError):
        replace(DiagnosisRuleConfig(), **overrides)


def test_registry_fingerprint_is_deterministic_and_semantic():
    canonical = canonical_rule_registry_json()
    assert (
        hashlib.sha256(canonical.encode()).hexdigest() == DIAGNOSIS_RULE_REGISTRY_SHA256
    )
    modified = replace(RULE_REGISTRY[0], operator="<=")
    assert (
        canonical_rule_registry_json(registry=(modified, *RULE_REGISTRY[1:]))
        != canonical
    )


@pytest.mark.parametrize(
    "field",
    [
        "minimum_labeled_sources_per_class",
        "preferred_labeled_sources_per_class",
        "cross_validation_folds",
        "maximum_newton_iterations",
    ],
)
def test_a63_integer_config_rejects_bool(field):
    with pytest.raises(ValueError):
        replace(SportCalibrationFitConfig(), **{field: True})


def test_a63_semantic_fingerprints_ignore_runtime_performance():
    dataset = {
        "dataset_id": "x",
        "performance": {"dataset_extraction_seconds": 1.0},
        "x": 1,
    }
    other = {**dataset, "performance": {"dataset_extraction_seconds": 99.0}}
    assert semantic_dataset_payload(dataset) == semantic_dataset_payload(other)
    artifact = {
        "calibration_artifact_sha256": "",
        "performance": {"cross_validation_seconds": 1.0},
        "x": 1,
    }
    other_artifact = {**artifact, "performance": {"cross_validation_seconds": 99.0}}
    assert artifact_fingerprint(artifact) == artifact_fingerprint(other_artifact)


def test_a63_dataset_readiness_uses_config_and_explicit_source_id(
    tmp_path, monkeypatch
):
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    artifact_paths = []
    for source, label, content in (
        ("source-ski", "SKI", b"ski"),
        ("source-snow", "SNOWBOARD", b"snow"),
    ):
        video = tmp_path / f"{source}.mp4"
        video.write_bytes(content)
        video_sha = hashlib.sha256(content).hexdigest()
        artifact = tmp_path / f"{source}.json"
        artifact.write_text(
            json.dumps(
                {
                    "benchmark_contract_version": "ski-bench-sport-type-v4",
                    "source_video_id": source,
                    "source_video_id_origin": "EXPLICIT",
                    "video": {"path": str(video), "sha256": video_sha},
                    "equipment_models": [
                        {
                            "provider_name": "provider",
                            "model_id": "model",
                            "checkpoint_sha256": "c" * 64,
                        }
                    ],
                    "visual_models": [],
                    "sport_type": {
                        "provider_results": [
                            {
                                "provider_name": "provider",
                                "evidence_kind": "EQUIPMENT",
                                "observations": [
                                    {
                                        "ski_support": 0.8 if label == "SKI" else 0.1,
                                        "snowboard_support": 0.1
                                        if label == "SKI"
                                        else 0.8,
                                        "quality": 1.0,
                                        "timestamp_us": 0,
                                    }
                                ],
                            }
                        ],
                        "auto_decision": None,
                    },
                }
            ),
            encoding="utf-8",
        )
        artifact_paths.append(artifact)
        (annotation_dir / f"{source}.json").write_text(
            json.dumps(
                {
                    "contract_version": "sport-type-gt-v1",
                    "clip_id": source,
                    "source_video_id": source,
                    "video_sha256": video_sha,
                    "target_sport_type": label,
                    "annotation_source": "USER_MANUAL",
                    "intended_target_confirmation": "CONFIRMED",
                    "notes": "",
                }
            ),
            encoding="utf-8",
        )
    clock = iter((1.0, 2.0, 10.0, 99.0, 100.0, 101.0))
    monkeypatch.setattr(
        "slopecoach_ml.sport_type.calibration.dataset.time.perf_counter",
        lambda: next(clock),
    )
    research = SportCalibrationFitConfig(
        minimum_labeled_sources_per_class=1,
        preferred_labeled_sources_per_class=1,
    )
    first = build_calibration_dataset(artifact_paths, annotation_dir, research)
    second = build_calibration_dataset(artifact_paths, annotation_dir, research)
    assert first["status"] == "READY_FOR_CALIBRATION_FIT"
    assert first["dataset_id"] == second["dataset_id"]
    assert {item["source_video_id_origin"] for item in first["clips"]} == {"EXPLICIT"}
    default = build_calibration_dataset(artifact_paths, annotation_dir)
    assert default["status"] == "INSUFFICIENT_LABELED_SPORT_TYPE_GT"


def test_a63_provenance_mismatch_and_null_direction_are_safe():
    annotations = {
        source: {
            "target_sport_type": label,
            "annotation_source": "USER_MANUAL",
            "intended_target_confirmation": "CONFIRMED",
            "video_sha256": char * 64,
        }
        for source, label, char in (("ski", "SKI", "a"), ("snow", "SNOWBOARD", "b"))
    }
    samples = []
    for source, direction, char, checkpoint in (
        ("ski", -0.8, "a", "1"),
        ("snow", 0.8, "b", "2"),
    ):
        samples.append(
            {
                "calibration_channel_id": "EQUIPMENT::provider",
                "provider_name": "provider",
                "evidence_kind": "EQUIPMENT",
                "source_video_id": source,
                "video_sha256": char * 64,
                "raw_direction": direction,
                "provenance": {
                    "model_id": "model",
                    "checkpoint_sha256": checkpoint * 64,
                },
            }
        )
    dataset = {
        "contract_version": "sport-type-calibration-dataset-v1",
        "dataset_id": "dataset",
        "annotations": annotations,
        "source_samples": samples,
        "per_channel_usable_source_counts": {"EQUIPMENT::provider": 2},
        "labeled_source_count": 2,
        "ski_labeled_source_count": 1,
        "snowboard_labeled_source_count": 1,
    }
    artifact = fit_calibration_artifact(
        dataset,
        SportCalibrationFitConfig(
            minimum_labeled_sources_per_class=1, preferred_labeled_sources_per_class=1
        ),
    )
    assert (
        artifact["channels"][0]["status"] == "CALIBRATION_CHANNEL_PROVENANCE_MISMATCH"
    )
    valid = {
        "contract_version": "sport-evidence-calibration-v1",
        "calibration_artifact_sha256": "",
        "status": "RESEARCH_CALIBRATION_AVAILABLE",
        "fit_config": {"probability_epsilon": 1e-6},
        "channels": [],
        "fusion": {
            "fusion_prior_snowboard": 0.5,
            "calibrated_conflict_probability": 0.8,
        },
    }
    valid["calibration_artifact_sha256"] = artifact_fingerprint(valid)
    result = apply_calibrated_fusion(
        [{"calibration_channel_id": "EQUIPMENT::provider", "raw_direction": None}],
        valid,
    )
    assert result["status"] == "NOT_AVAILABLE_INSUFFICIENT_CALIBRATED_CHANNELS"


def test_artifact_only_diagnosis_benchmark(tmp_path):
    facts, turn, turn_features = _inputs()
    artifact = tmp_path / "biomechanics.json"
    artifact.write_text(
        json.dumps(
            {
                "biomechanics_result": {
                    "frame_facts": facts,
                    "turn_features": [turn_features],
                },
                "turn_segments": [turn],
            }
        ),
        encoding="utf-8",
    )
    report = benchmark_diagnosis_artifact(artifact, sport_type="ski")
    assert report["benchmark_contract_version"] == "ski-bench-diagnosis-v1"
    assert report["qualified_turn_count"] == 1
    assert report["ground_truth"]["diagnosis_f1"] is None
