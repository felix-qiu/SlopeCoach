from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from slopecoach_ml.benchmark import sport_type as module
from slopecoach_ml.biomechanics import FEATURE_REGISTRY_SHA256
from slopecoach_ml.sport_type import (
    FailedSportEvidenceProvider,
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
    assert report["benchmark_contract_version"] == "ski-bench-sport-type-v5"
    assert report["SPORT_TYPE_ROUTING_POLICY"] == ("EQUIPMENT_FIRST_VISUAL_FALLBACK_V1")
    assert not report["CALIBRATED_FUSION_CONTROLS_ROUTING"]
    assert report["sport_type"]["effective_sport_type"] == "UNKNOWN"
    assert report["sport_type"]["contract_version"] == "sport-type-v2"
    assert report["sport_type"]["routing_policy"] == (
        "EQUIPMENT_FIRST_VISUAL_FALLBACK_V1"
    )
    assert report["sport_type"]["resolution_status"] == "INSUFFICIENT_PRIMARY_EVIDENCE"
    assert report["sport_type"]["config"]["profile"] == "RESEARCH_DEFAULTS_A6_4"
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
    assert (
        report["validation"]["AUTO_SPORT_TYPE_PRODUCT_READINESS"]
        == "NOT_READY_PRIMARY_PROVIDER_REQUIRED"
    )


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


def test_benchmark_readiness_distinguishes_unknown_evidence_from_resolved(monkeypatch):
    monkeypatch.setattr(
        module, "benchmark_biomechanics_frames", lambda **kwargs: _upstream_report()
    )

    def run(ski, snowboard):
        observations = tuple(
            SportEvidenceObservation(
                f"EQUIPMENT:mock-equipment:{timestamp}:{index}",
                SportEvidenceKind.EQUIPMENT,
                "mock-equipment",
                ski,
                snowboard,
                1.0,
                SportEvidenceScope.FRAME,
                timestamp_us=timestamp,
            )
            for index, timestamp in enumerate((0, 200000))
        )
        return module.benchmark_sport_type_frames(
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

    unresolved = run(0.20, 0.30)
    assert (
        unresolved["validation"]["AUTO_SPORT_TYPE_PRODUCT_READINESS"]
        == "NOT_READY_PRIMARY_EVIDENCE_COVERAGE_AND_GT_REQUIRED"
    )
    resolved = run(0.90, 0.02)
    assert (
        resolved["validation"]["AUTO_SPORT_TYPE_PRODUCT_READINESS"]
        == "NOT_READY_GT_REQUIRED"
    )
    assert (
        resolved["diagnostic_auto_decisions"]["combined_auto_decision"]
        == resolved["sport_type"]["auto_decision"]
    )


def test_benchmark_locked_contexts_are_not_summed_across_providers(monkeypatch):
    monkeypatch.setattr(
        module, "benchmark_biomechanics_frames", lambda **kwargs: _upstream_report()
    )

    class Collector:
        sport_contexts = tuple(
            SimpleNamespace(timestamp_us=timestamp) for timestamp in (0, 200000, 400000)
        )

        def release_frame_contexts(self):
            pass

    report = module.benchmark_sport_type_frames(
        input_path="ski_test_001.mp4",
        frames=(),
        detector=None,
        pose_provider=None,
        detector_model={},
        pose_model={},
        collector=Collector(),
        evidence_providers=(
            MockSportEvidenceProvider("equipment-a", SportEvidenceKind.EQUIPMENT),
            MockSportEvidenceProvider("equipment-b", SportEvidenceKind.EQUIPMENT),
            NotConfiguredVisualSportEvidenceProvider(),
        ),
    )
    assert report["sport_frame_contexts"] == {
        "locked_context_count": 3,
        "distinct_timestamp_count": 3,
    }


def test_initialized_provider_failure_does_not_erase_other_primary_evidence(
    monkeypatch,
):
    monkeypatch.setattr(
        module, "benchmark_biomechanics_frames", lambda **kwargs: _upstream_report()
    )
    observations = tuple(
        SportEvidenceObservation(
            f"EQUIPMENT:equipment:{timestamp}:{index}",
            SportEvidenceKind.EQUIPMENT,
            "equipment",
            0.9,
            0.01,
            1.0,
            SportEvidenceScope.FRAME,
            timestamp_us=timestamp,
        )
        for index, timestamp in enumerate((0, 200000))
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
                "equipment", SportEvidenceKind.EQUIPMENT, observations
            ),
            FailedSportEvidenceProvider(
                "openai-clip-vit-b32-visual-sport",
                SportEvidenceKind.VISUAL_CLASSIFIER,
                "RuntimeError: MODEL_LOAD_FAILED",
            ),
        ),
    )
    assert report["provider_validation"]["VISUAL_SPORT_PROVIDER_STATUS"] == "FAILED"
    assert report["provider_validation"]["EQUIPMENT_SPORT_PROVIDER_STATUS"] == (
        "EXECUTED_WITH_EVIDENCE"
    )
    assert report["sport_type"]["auto_decision"]["sport_type"] == "SKI"


def test_benchmark_exposes_equipment_first_hierarchy_without_hiding_visual(monkeypatch):
    monkeypatch.setattr(
        module, "benchmark_biomechanics_frames", lambda **kwargs: _upstream_report()
    )

    def provider(name, kind, ski, snowboard):
        observations = tuple(
            SportEvidenceObservation(
                f"{kind.value}:{name}:{timestamp}:{index}",
                kind,
                name,
                ski,
                snowboard,
                1.0,
                SportEvidenceScope.FRAME,
                timestamp_us=timestamp,
            )
            for index, timestamp in enumerate(index * 200000 for index in range(6))
        )
        return MockSportEvidenceProvider(name, kind, observations)

    report = module.benchmark_sport_type_frames(
        input_path="ski_test_001.mp4",
        frames=(),
        detector=None,
        pose_provider=None,
        detector_model={},
        pose_model={},
        evidence_providers=(
            provider("equipment", SportEvidenceKind.EQUIPMENT, 0.90, 0.05),
            provider("visual", SportEvidenceKind.VISUAL_CLASSIFIER, 0.05, 0.90),
        ),
    )
    sport = report["sport_type"]
    assert sport["effective_sport_type"] == "SKI"
    assert sport["auto_decision"]["routing_basis"] == "EQUIPMENT_PRIMARY"
    assert sport["auto_decision"]["ski_support"] == pytest.approx(0.90)
    assert sport["auto_decision"]["snowboard_support"] == pytest.approx(0.05)
    assert (
        "EQUIPMENT_PRIMARY_VISUAL_DISAGREES" in sport["auto_decision"]["reason_codes"]
    )
    provider_results = {
        item["evidence_kind"]: item for item in sport["provider_results"]
    }
    assert (
        provider_results["VISUAL_CLASSIFIER"]["observations"][0]["snowboard_support"]
        == 0.90
    )
    diagnostics = report["diagnostic_auto_decisions"]
    assert diagnostics["equipment_only_auto_decision"]["sport_type"] == "SKI"
    assert diagnostics["visual_only_auto_decision"]["sport_type"] == "SNOWBOARD"
    assert (
        report["equipment_only_auto_decision"]
        == diagnostics["equipment_only_auto_decision"]
    )
    assert (
        report["visual_only_auto_decision"] == diagnostics["visual_only_auto_decision"]
    )
    assert diagnostics["hierarchical_auto_decision"] == report["raw_auto_decision"]
    assert diagnostics["combined_auto_decision"] == report["raw_auto_decision"]
    assert not report["CALIBRATED_FUSION_CONTROLS_ROUTING"]
    assert report["ground_truth"]["SPORT_TYPE_GT_STATUS"] == "NOT_AVAILABLE"


def test_calibrated_diagnostic_cannot_override_hierarchical_raw_routing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        module, "benchmark_biomechanics_frames", lambda **kwargs: _upstream_report()
    )
    monkeypatch.setattr(
        module,
        "apply_calibrated_fusion",
        lambda summaries, artifact: {
            "status": "AVAILABLE_SINGLE_PRIMARY_KIND",
            "calibrated_ski_probability": 0.01,
            "calibrated_snowboard_probability": 0.99,
            "routing_eligible": False,
            "CALIBRATED_FUSION_CONTROLS_ROUTING": False,
        },
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"synthetic-not-decoded")
    observations = tuple(
        SportEvidenceObservation(
            f"eq-{timestamp}",
            SportEvidenceKind.EQUIPMENT,
            "equipment",
            0.90,
            0.05,
            1.0,
            SportEvidenceScope.FRAME,
            timestamp_us=timestamp,
        )
        for timestamp in (0, 200000)
    )
    report = module.benchmark_sport_type_frames(
        input_path=video,
        frames=(),
        detector=None,
        pose_provider=None,
        detector_model={},
        pose_model={},
        evidence_providers=(
            MockSportEvidenceProvider(
                "equipment", SportEvidenceKind.EQUIPMENT, observations
            ),
            NotConfiguredVisualSportEvidenceProvider(),
        ),
        calibration_artifact={"diagnostic": True},
    )
    assert (
        report["calibrated_fusion_result"]["calibrated_snowboard_probability"] == 0.99
    )
    assert report["sport_type"]["effective_sport_type"] == "SKI"
    assert report["raw_auto_decision"]["routing_basis"] == "EQUIPMENT_PRIMARY"
    assert not report["CALIBRATED_FUSION_CONTROLS_ROUTING"]
