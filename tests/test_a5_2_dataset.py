from __future__ import annotations

import json
from dataclasses import replace

import pytest

from slopecoach_ml.benchmark.real_dataset import (
    BiomechanicsDatasetValidationConfig,
    RealDatasetClip,
    RealDatasetManifest,
    aggregate_biomechanics_dataset,
    execute_biomechanics_dataset,
    real_data_validation_status,
    write_dataset_contact_sheet,
)
from slopecoach_ml.biomechanics import (
    FEATURE_REGISTRY_SHA256,
    FRAME_FEATURE_REGISTRY_V1,
    TEMPORAL_FEATURE_REGISTRY_V1,
    TURN_FEATURE_REGISTRY_V1,
    BiomechanicsFactStatus,
    FeatureAggregate,
    TurnBiomechanicsResult,
)


def clip(clip_id="clip-a", source="video-a", **kwargs):
    values = {
        "clip_id": clip_id,
        "source_video_id": source,
        "path": f"videos/{clip_id}.mp4",
        "enabled": True,
        "mirror_policy": "NON_MIRRORED",
    }
    values.update(kwargs)
    return RealDatasetClip(**values)


def manifest(clips):
    return RealDatasetManifest("dataset", "test", tuple(clips))


def report(*, trusted=10, available=10, knee_value=90, turns=0):
    frame_facts = []
    coverage = {}
    aggregates = []
    for definition in FRAME_FEATURE_REGISTRY_V1:
        count = (
            available if definition.feature_id == "left_knee_angle_2d_deg" else trusted
        )
        coverage[definition.feature_id] = {
            "total_trusted_frames": trusted,
            "available_frame_count": count,
            "coverage_ratio": count / trusted if trusted else 0.0,
            "status_reason_counts": (
                {"REQUIRED_JOINT_OUT_OF_FRAME": trusted - count}
                if trusted - count
                else {}
            ),
        }
        values = (
            [knee_value] if definition.feature_id == "left_knee_angle_2d_deg" else [1.0]
        )
        frame_facts.extend(
            {
                "feature_id": definition.feature_id,
                "status": "AVAILABLE",
                "value": values[0],
            }
            for _ in range(count)
        )
        if trusted:
            aggregates.append(
                {
                    "feature_id": definition.feature_id,
                    "temporal_segment_id": 1,
                    "status": "AVAILABLE",
                    "median": values[0],
                    "range": 0.0,
                }
            )
    for definition in TEMPORAL_FEATURE_REGISTRY_V1:
        aggregates.append(
            {
                "feature_id": definition.feature_id,
                "temporal_segment_id": 1,
                "status": "AVAILABLE" if trusted else "INSUFFICIENT_SAMPLES",
                "median": 2.0 if trusted else None,
                "range": 0.0 if trusted else None,
            }
        )
    turn_results = []
    for number in range(turns):
        facts = [
            {
                "feature_id": definition.feature_id,
                "status": "AVAILABLE"
                if number != turns - 1
                else "INSUFFICIENT_EVIDENCE",
                "value": 1 if number != turns - 1 else None,
            }
            for definition in TURN_FEATURE_REGISTRY_V1
        ]
        turn_results.append({"turn_id": f"turn-{number}", "facts": facts})
    return {
        "benchmark_contract_version": "ski-bench-biomechanics-v2",
        "feature_registry_sha256": FEATURE_REGISTRY_SHA256,
        "video": {"duration_seconds": 10.0},
        "sampling": {"sampled_frame_count": 20},
        "identity_input": {
            "identity_locked_frame_count": trusted,
            "identity_unsafe_frame_count": 20 - trusted,
        },
        "upstream_conditions": {
            "raw_detection_count": 20,
            "raw_candidate_density": 1.0,
            "required_joint_visibility_ratio": 0.9,
        },
        "temporal_input": {"temporal_segment_count": 1 if trusted else 0},
        "turn_input": {
            "qualified_turn_count": turns,
            "turn_status": "EXECUTED_PROVISIONAL_CANDIDATES"
            if turns
            else "EXECUTED_NO_QUALIFIED_TURN_CANDIDATES",
        },
        "turn_signal_summary": {
            "valid_sample_count": trusted,
            "valid_signal_run_count": 1 if trusted else 0,
            "longest_valid_signal_run_duration_us": 1_000_000 if trusted else 0,
            "signal_value_span": 0.5 if trusted else None,
            "median_absolute_signal_delta": 0.1 if trusted else None,
        },
        "turn_segments": [
            {
                "status": "VALID",
                "start_timestamp_us": 0,
                "end_timestamp_us": 1_000_000,
                "duration_us": 1_000_000,
            }
            for _ in range(turns)
        ],
        "frame_biomechanics": {
            "trusted_frame_count": trusted,
            "feature_coverage": coverage,
        },
        "temporal_segment_features": aggregates,
        "turn_biomechanics": turn_results,
        "biomechanics_result": {"frame_facts": frame_facts},
        "performance": {
            "detector_total_seconds": 1.0,
            "pose_total_seconds": 2.0,
            "tracking_identity_total_seconds": 0.1,
            "temporal_total_seconds": 0.01,
            "turn_total_seconds": 0.01,
            "biomechanics_frame_total_seconds": 0.001,
            "biomechanics_aggregation_total_seconds": 0.001,
            "biomechanics_turn_total_seconds": 0.001,
            "total_seconds": 3.2,
        },
        "validation": {
            "REAL_BIOMECHANICS_STATUS": "EXECUTED_FRAME_AND_SEGMENT_FEATURES_NO_TURNS"
        },
    }


def record(current, payload):
    return {
        "clip_id": current.clip_id,
        "source_video_id": current.source_video_id,
        "execution_status": "SUCCESS",
        "sha256": "0" * 64,
        "report": payload,
    }


@pytest.mark.parametrize(
    "count,expected",
    [
        (0, "NOT_ANALYZABLE_NO_REAL_CLIPS"),
        (1, "INSUFFICIENT_DATASET_SINGLE_CLIP"),
        (3, "LIMITED_MULTICLIP_EVIDENCE"),
        (5, "MULTICLIP_ENGINEERING_EVIDENCE"),
    ],
)
def test_dataset_evidence_levels(count, expected):
    assert real_data_validation_status(count) == expected


@pytest.mark.parametrize(
    "threshold,count,expected",
    [
        (10, 5, "LIMITED_MULTICLIP_EVIDENCE"),
        (10, 9, "LIMITED_MULTICLIP_EVIDENCE"),
        (10, 10, "MULTICLIP_ENGINEERING_EVIDENCE"),
        (2, 1, "INSUFFICIENT_DATASET_SINGLE_CLIP"),
        (2, 2, "MULTICLIP_ENGINEERING_EVIDENCE"),
    ],
)
def test_dataset_evidence_level_uses_configured_threshold(threshold, count, expected):
    config = BiomechanicsDatasetValidationConfig(
        minimum_multiclip_source_videos=threshold
    )
    assert real_data_validation_status(count, config) == expected


@pytest.mark.parametrize("threshold", [0, 1, True, 2.0, "5"])
def test_multiclip_threshold_requires_integer_at_least_two(threshold):
    with pytest.raises(ValueError):
        BiomechanicsDatasetValidationConfig(
            minimum_multiclip_source_videos=threshold
        ).validate()


def test_aggregator_forwards_validation_config_threshold():
    clips = [clip(f"clip-{index}", f"video-{index}") for index in range(5)]
    records = [record(item, report()) for item in clips]
    limited = aggregate_biomechanics_dataset(
        manifest(clips),
        records,
        BiomechanicsDatasetValidationConfig(minimum_multiclip_source_videos=10),
    )
    assert (
        limited["validation"]["A5_2_REAL_DATA_VALIDATION"]
        == "LIMITED_MULTICLIP_EVIDENCE"
    )
    default = aggregate_biomechanics_dataset(
        manifest(clips),
        records,
        BiomechanicsDatasetValidationConfig(minimum_multiclip_source_videos=5),
    )
    assert (
        default["validation"]["A5_2_REAL_DATA_VALIDATION"]
        == "MULTICLIP_ENGINEERING_EVIDENCE"
    )


def test_independent_sources_and_duplicate_subclips_do_not_inflate_status():
    clips = [clip(f"clip-{index}", "same-video") for index in range(5)]
    result = aggregate_biomechanics_dataset(
        manifest(clips), [record(item, report()) for item in clips]
    )
    assert result["dataset"]["enabled_clip_count"] == 5
    assert result["dataset"]["independent_source_video_count"] == 1
    assert (
        result["validation"]["A5_2_REAL_DATA_VALIDATION"]
        == "INSUFFICIENT_DATASET_SINGLE_CLIP"
    )
    distinct = [clip("a", "video-1"), clip("b", "video-1"), clip("c", "video-2")]
    result = aggregate_biomechanics_dataset(
        manifest(distinct), [record(item, report()) for item in distinct]
    )
    assert result["dataset"]["independent_source_video_count"] == 2


def test_non_evaluable_clip_excluded_and_macro_micro_differ():
    a, b, c = clip("a", "a"), clip("b", "b"), clip("c", "c")
    result = aggregate_biomechanics_dataset(
        manifest([a, b, c]),
        [
            record(a, report(trusted=0, available=0)),
            record(b, report(trusted=10, available=10)),
            record(c, report(trusted=100, available=50)),
        ],
    )
    feature = next(
        x
        for x in result["frame_feature_robustness"]
        if x["feature_id"] == "left_knee_angle_2d_deg"
    )
    assert result["dataset"]["feature_evaluable_clip_count"] == 2
    assert result["dataset"]["upstream_non_evaluable_clip_count"] == 1
    assert feature["macro_coverage_mean"] == pytest.approx(0.75)
    assert feature["micro_coverage_ratio"] == pytest.approx(60 / 110)


def test_turn_denominator_failure_matrix_and_strict_json():
    clips = [clip(str(i), str(i)) for i in range(5)]
    reports = [
        report(turns=0),
        report(turns=0),
        report(turns=2),
        report(turns=2),
        report(turns=0),
    ]
    for turn in reports[2]["turn_biomechanics"]:
        turn["facts"][0].update(status="AVAILABLE", value=1)
    result = aggregate_biomechanics_dataset(
        manifest(clips), [record(c, r) for c, r in zip(clips, reports, strict=True)]
    )
    turn = result["turn_feature_robustness"][0]
    assert turn["eligible_turn_count"] == 4 and turn["available_turn_fact_count"] == 3
    assert turn["coverage_ratio"] == pytest.approx(3 / 4)
    matrix = result["failure_reason_matrix"]["dataset_status_totals"]
    assert matrix["REQUIRED_JOINT_OUT_OF_FRAME"] == 0
    json.dumps(result, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    "feature,value,key",
    [
        ("left_knee_angle_2d_deg", 181, "knee_angle_domain_violations"),
        ("signed_lateral_body_proxy", 1.5, "signed_lateral_proxy_domain_violations"),
        (
            "shoulder_hip_axis_difference_2d_deg",
            100,
            "axis_difference_domain_violations",
        ),
    ],
)
def test_mathematical_contract_violations(feature, value, key):
    current = clip()
    payload = report()
    fact = next(
        x
        for x in payload["biomechanics_result"]["frame_facts"]
        if x["feature_id"] == feature
    )
    fact["value"] = value
    result = aggregate_biomechanics_dataset(
        manifest([current]), [record(current, payload)]
    )
    assert result["contract_checks"][key] == 1
    status = next(
        x for x in result["frame_feature_robustness"] if x["feature_id"] == feature
    )
    assert status["robustness_status"] == "CONTRACT_FAILURE"
    assert status["retention_recommendation"] == "REVIEW_IMPLEMENTATION"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: replace(c, source_video_id=""),
        lambda c: replace(c, path=""),
        lambda c: replace(c, sample_fps=True),
        lambda c: replace(c, sample_fps=-1),
        lambda c: replace(c, expected_sha256="bad"),
        lambda c: replace(c, enabled=1),
    ],
)
def test_manifest_clip_safety(mutation):
    with pytest.raises(ValueError):
        mutation(clip()).validate()


def test_manifest_contract_and_duplicate_clip_safety():
    with pytest.raises(ValueError):
        RealDatasetManifest("dataset", "", (clip(), clip())).validate()
    data = {"contract_version": "wrong", "dataset_id": "x", "clips": []}
    with pytest.raises(ValueError):
        RealDatasetManifest.from_dict(data)


def test_execution_isolates_missing_sha_and_benchmark_failures(tmp_path):
    good_path = tmp_path / "good.mp4"
    bad_path = tmp_path / "bad.mp4"
    good_path.write_bytes(b"good")
    bad_path.write_bytes(b"bad")
    clips = [
        clip("missing", "missing", path=str(tmp_path / "missing.mp4")),
        clip("sha", "sha", path=str(bad_path), expected_sha256="0" * 64),
        clip("failed", "failed", path=str(good_path)),
    ]

    def fail_benchmark(current, path, debug_path):
        raise RuntimeError("MODEL_RUNTIME_FAILED: test")

    result = execute_biomechanics_dataset(manifest(clips), fail_benchmark)
    statuses = {
        item["clip_id"]: item["execution_status"] for item in result["per_clip"]
    }
    assert statuses == {
        "missing": "VIDEO_NOT_FOUND",
        "sha": "SHA_MISMATCH",
        "failed": "MODEL_RUNTIME_FAILED",
    }
    assert (
        result["validation"]["A5_2_REAL_DATA_VALIDATION"]
        == "INSUFFICIENT_DATASET_SINGLE_CLIP"
    )


def test_dataset_contact_sheet_frame_cap_validation(tmp_path):
    with pytest.raises(ValueError):
        write_dataset_contact_sheet({"per_clip": []}, tmp_path, frames_per_clip=3)


def test_a5_1_required_ids_and_aggregate_algebra():
    valid = FeatureAggregate(
        "x",
        1,
        "ratio",
        10,
        8,
        0.8,
        2.0,
        1.0,
        3.0,
        2.0,
        0.8,
        7,
        1,
        0.7,
        0.1,
        BiomechanicsFactStatus.AVAILABLE,
    )
    assert valid.temporal_segment_id == 1
    with pytest.raises(ValueError):
        replace(valid, temporal_segment_id=None)
    with pytest.raises(ValueError):
        replace(valid, support_ratio=0.7)
    with pytest.raises(ValueError):
        replace(valid, range=1.5)
    with pytest.raises(ValueError):
        TurnBiomechanicsResult("turn", None, 1, "POSITIVE_PHASE", ())
    with pytest.raises(ValueError):
        TurnBiomechanicsResult("turn", 1, None, "POSITIVE_PHASE", ())
