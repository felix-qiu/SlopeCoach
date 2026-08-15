from __future__ import annotations

import json

from slopecoach_ml.benchmark import sport_type as module
from slopecoach_ml.biomechanics import FEATURE_REGISTRY_SHA256
from slopecoach_ml.sport_type import (
    MockSportEvidenceProvider,
    NotConfiguredVisualSportEvidenceProvider,
    SportEvidenceKind,
    SportEvidenceObservation,
    SportEvidenceScope,
)


def _upstream_report():
    return {
        "video": {"path": "ski_test_001.mp4"},
        "sampling": {"sample_fps": 5.0, "sampled_frames": 3},
        "models": {"detector": {"name": "person-only"}},
        "identity_input": {"trusted_target_frames": 2},
        "temporal_input": {"temporal_segment_count": 1},
        "biomechanics_result": {
            "contract_version": "temporal-biomechanics-v2",
            "temporal_segment_features": [
                {
                    "feature_id": "ankle_separation_body_scale",
                    "median": 0.25,
                    "range": 0.1,
                },
                {
                    "feature_id": "bilateral_knee_mean_angle_2d_deg",
                    "median": 90.0,
                    "range": 18.0,
                },
            ],
        },
        "performance": {
            "detector_total_seconds": 1.0,
            "tracking_identity_total_seconds": 0.1,
            "pose_total_seconds": 2.0,
            "temporal_total_seconds": 0.01,
            "turn_total_seconds": 0.01,
            "biomechanics_frame_total_seconds": 0.01,
            "biomechanics_aggregation_total_seconds": 0.01,
            "biomechanics_turn_total_seconds": 0.01,
        },
    }


def test_benchmark_reports_honest_provider_gt_and_gate(monkeypatch):
    monkeypatch.setattr(
        module, "benchmark_biomechanics_frames", lambda **kwargs: _upstream_report()
    )
    report = module.benchmark_sport_type_frames(
        input_path="ski_test_001.mp4",
        frames=(),
        detector=None,
        pose_provider=None,
        detector_model={},
        pose_model={},
    )
    assert report["benchmark_contract_version"] == "ski-bench-sport-type-v2"
    assert report["sport_type"]["effective_sport_type"] == "UNKNOWN"
    assert report["sport_type"]["resolution_status"] == "INSUFFICIENT_PRIMARY_EVIDENCE"
    assert report["sport_type"]["config"]["profile"] == "RESEARCH_DEFAULTS_A6"
    assert (
        report["provider_validation"]["EQUIPMENT_SPORT_PROVIDER_STATUS"]
        == "NOT_CONFIGURED"
    )
    assert (
        report["provider_validation"]["VISUAL_SPORT_PROVIDER_STATUS"]
        == "NOT_CONFIGURED"
    )
    assert report["ground_truth"]["sport_type_accuracy"] is None
    assert report["ground_truth"]["sport_type_precision"] is None
    assert report["ground_truth"]["sport_type_recall"] is None
    assert not report["downstream_gate"]["SPORT_SPECIFIC_ANALYSIS_ALLOWED"]
    assert (
        report["biomechanics_input"]["feature_registry_sha256"]
        == FEATURE_REGISTRY_SHA256
    )
    json.dumps(report, allow_nan=False, sort_keys=True)


def test_executed_equipment_provider_removes_only_equipment_limitation(monkeypatch):
    monkeypatch.setattr(
        module, "benchmark_biomechanics_frames", lambda **kwargs: _upstream_report()
    )
    observations = tuple(
        SportEvidenceObservation(
            f"equipment-{timestamp}",
            SportEvidenceKind.EQUIPMENT,
            "mock-equipment",
            0.9,
            0.0,
            1.0,
            SportEvidenceScope.FRAME,
            timestamp_us=timestamp,
        )
        for timestamp in (0, 200000)
    )
    report = module.benchmark_sport_type_frames(
        input_path="ski_test_001.mp4",
        frames=(),
        detector=None,
        pose_provider=None,
        detector_model={},
        pose_model={},
        evidence_providers=(
            MockSportEvidenceProvider(
                "mock-equipment", SportEvidenceKind.EQUIPMENT, observations
            ),
            NotConfiguredVisualSportEvidenceProvider(),
        ),
    )
    assert "NO_CONFIGURED_EQUIPMENT_CLASSIFIER" not in report["limitations"]
    assert "NO_CONFIGURED_VISUAL_SPORT_CLASSIFIER" in report["limitations"]
    summary = report["provider_validation"]["provider_kind_summaries"][0]
    assert summary["overall_status"] == "EXECUTED_WITH_EVIDENCE"
