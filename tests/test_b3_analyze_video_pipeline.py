from __future__ import annotations

import copy
import json

from slopecoach_ml.product import assemble_analyze_video_product, select_user_sport_type

TIMESTAMPS = (0, 250_000, 500_000, 750_000, 1_000_000)


def _fact(feature, timestamp, value):
    return {
        "feature_id": feature,
        "unit": "deg",
        "value": value,
        "status": "AVAILABLE",
        "timestamp_us": timestamp,
        "temporal_segment_id": 1,
        "signal_run_id": 1,
        "support_confidence": 0.8,
        "observed_joint_count": 6,
        "interpolated_joint_count": 0,
    }


def _report(*, qualified: bool, safe: bool = True):
    frame_facts = []
    for timestamp, knee in zip(TIMESTAMPS, (80, 81, 82, 81, 80), strict=True):
        frame_facts.extend(
            (
                _fact("bilateral_knee_mean_angle_2d_deg", timestamp, knee),
                _fact("bilateral_knee_abs_difference_2d_deg", timestamp, 2),
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
                "value": 0.0,
                "status": "AVAILABLE",
            },
            {
                "feature_id": "minimum_mean_knee_angle_timestamp_us",
                "value": 500_000,
                "status": "AVAILABLE",
            },
        ],
    }
    return {
        "benchmark_contract_version": "ski-bench-biomechanics-v2",
        "feature_schema_version": "biomechanics-feature-schema-v1",
        "feature_registry_sha256": (
            "2777c3fbf7513e7537122f897f1901e61baf7eeddcee927937decb7476953048"
        ),
        "video": {
            "path": "synthetic.mp4",
            "duration_seconds": 1.0,
            "width_px": 1920,
            "height_px": 1080,
        },
        "models": {
            "detector": {"model_id": "detector-test"},
            "pose": {"model_id": "pose-test"},
        },
        "runtime": {"device": "cpu", "warmup_frames": 0},
        "performance": {"total_seconds": 0.1},
        "identity_input": {
            "identity_locked_frame_count": 5 if safe else 0,
            "identity_unsafe_frame_count": 0 if safe else 5,
        },
        "frame_biomechanics": {"trusted_frame_count": 5 if safe else 0},
        "turn_segments": [turn] if qualified else [],
        "biomechanics_result": {
            "contract_version": "temporal-biomechanics-v2",
            "frame_facts": frame_facts if safe else [],
            "temporal_segment_features": [],
            "turn_features": [turn_features] if qualified else [],
            "feature_coverage": {},
            "limitations": ["IMAGE_SPACE_2D_ONLY_NOT_PHYSICAL_3D"],
        },
    }


def _section(payload, name):
    return next(
        section
        for section in payload["analysis_result"]["sections"]
        if section["name"] == name
    )


def test_no_qualified_turns_is_partial_without_downstream_fabrication():
    payload = assemble_analyze_video_product(
        video="synthetic.mp4",
        sport_type=select_user_sport_type("SKI"),
        biomechanics_report=_report(qualified=False),
    )
    product = payload["product_report"]
    assert product["contract_version"] == "product-report-v1"
    assert product["status"] == "PARTIAL_ANALYSIS"
    assert product["primary_reason_code"] == "NO_QUALIFIED_TURNS"
    assert _section(payload, "DIAGNOSIS")["status"] == "UNAVAILABLE"
    assert _section(payload, "SCORECARD")["status"] == "UNAVAILABLE"
    assert _section(payload, "COACH")["status"] == "UNAVAILABLE"
    assert product["scorecard"] is None
    assert product["headline"] is None
    assert product["practice_plan"] == []


def test_unsafe_target_is_not_analyzable_and_suppresses_downstream():
    payload = assemble_analyze_video_product(
        video="synthetic.mp4",
        sport_type=select_user_sport_type("SNOWBOARD"),
        biomechanics_report=_report(qualified=False, safe=False),
    )
    assert payload["product_report"]["status"] == "NOT_ANALYZABLE"
    assert payload["product_report"]["primary_reason_code"] == (
        "TARGET_IDENTITY_UNCERTAIN"
    )
    assert payload["product_report"]["practice_plan"] == []


def test_qualified_turn_flows_through_diagnosis_scorecard_coach_and_a9():
    payload = assemble_analyze_video_product(
        video="synthetic.mp4",
        sport_type=select_user_sport_type("SKI"),
        biomechanics_report=_report(qualified=True),
    )
    product = payload["product_report"]
    diagnosis = _section(payload, "DIAGNOSIS")["payload"]["diagnosis_result"]
    assert product["status"] == "READY"
    assert diagnosis["status"] == "EXECUTED_WITH_PROVISIONAL_DIAGNOSES"
    assert _section(payload, "SCORECARD")["status"] == "AVAILABLE"
    assert _section(payload, "COACH")["status"] == "AVAILABLE"
    assert product["scorecard"]["overall_score"] is None
    assert all(
        item["score_value"] is None for item in product["scorecard"]["dimensions"]
    )
    assert product["top_issues"]
    assert product["practice_plan"]
    assert payload["sport_type"]["effective_source"] == "USER"
    assert payload["sport_type"]["resolution_status"] == "RESOLVED_USER"
    assert payload["automatic_sport_type_research"]["executed"] is False


def test_product_report_sha_is_deterministic_and_input_is_not_mutated():
    report = _report(qualified=True)
    original = copy.deepcopy(report)
    first = assemble_analyze_video_product(
        video="synthetic.mp4",
        sport_type=select_user_sport_type("SNOWBOARD"),
        biomechanics_report=report,
    )
    second = assemble_analyze_video_product(
        video="synthetic.mp4",
        sport_type=select_user_sport_type("SNOWBOARD"),
        biomechanics_report=report,
    )
    assert report == original
    assert (
        first["analysis_result"]["analysis_result_sha256"]
        == second["analysis_result"]["analysis_result_sha256"]
    )
    assert (
        first["product_report"]["product_report_sha256"]
        == second["product_report"]["product_report_sha256"]
    )
    assert first["pipeline_provenance"]["models"] == report["models"]
    json.dumps(first, sort_keys=True, allow_nan=False)
