from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from slopecoach_ml.sport_type import SportEvidenceKind
from slopecoach_ml.sport_type.calibration.aggregation import (
    aggregate_subclips_by_source,
    summarize_observations,
)
from slopecoach_ml.sport_type.calibration.artifact import (
    artifact_fingerprint,
    compatible_channel,
)
from slopecoach_ml.sport_type.calibration.contracts import (
    AnnotationSource,
    GroundTruthSportType,
    IntendedTargetConfirmation,
    SportCalibrationFitConfig,
    SportTypeGroundTruth,
)
from slopecoach_ml.sport_type.calibration.fusion import (
    apply_calibrated_fusion,
    run_calibration_golden,
)
from slopecoach_ml.sport_type.calibration.platt import (
    CalibrationSample,
    deterministic_fold_assignment,
    evaluate_channel,
    stable_sigmoid,
)


def _observations(count=12, ski=0.2, snowboard=0.7):
    return [
        {
            "ski_support": ski,
            "snowboard_support": snowboard,
            "quality": 0.5 + index / 100,
            "timestamp_us": index * 200000,
        }
        for index in range(count)
    ]


def _summary(provider="equipment", kind=SportEvidenceKind.EQUIPMENT, source="source-1"):
    return summarize_observations(
        provider_name=provider,
        evidence_kind=kind,
        source_video_id=source,
        video_sha256="a" * 64,
        observations=_observations(),
        provenance={
            "model_id": "model",
            "checkpoint_sha256": "b" * 64,
            **(
                {"visual_prompt_sha256": "c" * 64}
                if kind is SportEvidenceKind.VISUAL_CLASSIFIER
                else {}
            ),
        },
    )


def test_frame_count_is_one_source_sample_and_subclips_aggregate():
    first = _summary()
    second = _summary()
    result = aggregate_subclips_by_source([first, second])
    assert len(result) == 1
    assert result[0].observation_count == 24
    assert result[0].clip_count_per_source == 2
    assert math.isclose(result[0].raw_direction, 0.5)


def test_gt_manual_contract_and_fit_eligibility():
    base = SportTypeGroundTruth("clip", "source", "a" * 64)
    assert not base.eligible_for_fitting
    labeled = replace(
        base,
        target_sport_type=GroundTruthSportType.SKI,
        intended_target_confirmation=IntendedTargetConfirmation.CONFIRMED,
    )
    assert labeled.eligible_for_fitting
    for forbidden in ("MODEL", "AUTO", "CLIP", "EQUIPMENT"):
        payload = labeled.to_dict() | {"annotation_source": forbidden}
        with pytest.raises(ValueError):
            SportTypeGroundTruth.from_dict(payload)
    with pytest.raises(ValueError):
        SportTypeGroundTruth.from_dict(
            labeled.to_dict() | {"target_sport_type": "UNKNOWN"}
        )
    assert labeled.annotation_source is AnnotationSource.USER_MANUAL


def _samples(useful=True):
    result = []
    for label, prefix in ((0, "ski"), (1, "snow")):
        for index in range(12):
            direction = ((index % 3) - 1) * 0.01
            if useful:
                direction += -0.8 if label == 0 else 0.8
            result.append(CalibrationSample(f"{prefix}-{index:02d}", direction, label))
    return result


def test_good_channel_deterministic_fit_and_grouped_folds():
    first = evaluate_channel(_samples(), "dataset", SportCalibrationFitConfig())
    second = evaluate_channel(_samples(), "dataset", SportCalibrationFitConfig())
    assert first == second
    assert first["status"] == "ACCEPTED_RESEARCH_CALIBRATION"
    assert first["final_fit"]["slope_a"] > 0
    assert first["brier_score"] < first["prior_only_brier_score"]
    duplicated = _samples() + [CalibrationSample("ski-00", -0.7, 0)]
    assignments = deterministic_fold_assignment(duplicated, "dataset", 5)
    assert len(assignments) == 24


def test_non_monotonic_and_useless_channels_rejected():
    reversed_samples = [
        CalibrationSample(
            item.source_video_id, -item.raw_direction, item.snowboard_label
        )
        for item in _samples()
    ]
    assert evaluate_channel(reversed_samples, "dataset", SportCalibrationFitConfig())[
        "status"
    ] == ("REJECTED_NON_MONOTONIC_CHANNEL")
    useless = [
        CalibrationSample(item.source_video_id, 0.0, item.snowboard_label)
        for item in _samples()
    ]
    assert evaluate_channel(useless, "dataset", SportCalibrationFitConfig())[
        "status"
    ] in {
        "REJECTED_NON_MONOTONIC_CHANNEL",
        "REJECTED_NO_BRIER_IMPROVEMENT",
        "REJECTED_NO_LOG_LOSS_IMPROVEMENT",
    }


def _artifact_channel(channel, kind, slope=1.0, intercept=0.0):
    return {
        "calibration_channel_id": channel,
        "provider_name": channel.split("::", 1)[1],
        "evidence_kind": kind,
        "slope_a": slope,
        "intercept_b": intercept,
        "training_snowboard_prior": 0.5,
        "status": "ACCEPTED_RESEARCH_CALIBRATION",
        "provenance": {
            "model_id": "model",
            "checkpoint_sha256": "b" * 64,
            **(
                {"visual_prompt_sha256": "c" * 64}
                if kind == "VISUAL_CLASSIFIER"
                else {}
            ),
        },
    }


def _artifact(channels):
    payload = {
        "contract_version": "sport-evidence-calibration-v1",
        "calibration_artifact_sha256": "",
        "fit_config": {"probability_epsilon": 1e-6},
        "channels": channels,
        "status": "RESEARCH_CALIBRATION_AVAILABLE",
        "fusion": {
            "fusion_prior_snowboard": 0.5,
            "calibrated_conflict_probability": 0.8,
        },
    }
    payload["calibration_artifact_sha256"] = artifact_fingerprint(payload)
    return payload


def test_calibrated_agreement_accumulates_same_kind_averages_and_conflict():
    equipment = _summary().to_dict()
    visual = _summary("visual", SportEvidenceKind.VISUAL_CLASSIFIER).to_dict()
    artifact = _artifact(
        [
            _artifact_channel(equipment["calibration_channel_id"], "EQUIPMENT", 2.0),
            _artifact_channel(
                visual["calibration_channel_id"], "VISUAL_CLASSIFIER", 3.0
            ),
        ]
    )
    agreed = apply_calibrated_fusion([equipment, visual], artifact)
    assert agreed["agreement_state"] == "AGREE_SNOWBOARD"
    assert math.isclose(agreed["fused_log_odds"], 2.5)
    assert not agreed["routing_eligible"]
    duplicate = equipment | {
        "calibration_channel_id": "EQUIPMENT::equipment-2",
        "provider_name": "equipment-2",
    }
    artifact2 = _artifact(
        artifact["channels"]
        + [_artifact_channel("EQUIPMENT::equipment-2", "EQUIPMENT", 2.0)]
    )
    same_kind = apply_calibrated_fusion([equipment, duplicate, visual], artifact2)
    assert math.isclose(same_kind["fused_log_odds"], agreed["fused_log_odds"])

    ski_equipment = equipment | {"raw_direction": -1.0}
    strong_visual = visual | {"raw_direction": 1.0}
    conflict_artifact = _artifact(
        [
            _artifact_channel(equipment["calibration_channel_id"], "EQUIPMENT", 2.0),
            _artifact_channel(
                visual["calibration_channel_id"], "VISUAL_CLASSIFIER", 2.0
            ),
        ]
    )
    conflict = apply_calibrated_fusion(
        [ski_equipment, strong_visual], conflict_artifact
    )
    assert conflict["status"] == "CONFLICTING_CALIBRATED_PRIMARY_EVIDENCE"
    assert conflict["agreement_state"] == "CONFLICT"


def test_provider_specific_calibration_and_compatibility():
    assert stable_sigmoid(1.0 * 0.3) != stable_sigmoid(3.0 * 0.3)
    visual = _summary("visual", SportEvidenceKind.VISUAL_CLASSIFIER).to_dict()
    channel = _artifact_channel(visual["calibration_channel_id"], "VISUAL_CLASSIFIER")
    assert compatible_channel(channel, visual)
    drifted = visual | {
        "provenance": visual["provenance"] | {"visual_prompt_sha256": "d" * 64}
    }
    assert not compatible_channel(channel, drifted)
    equipment = _summary().to_dict()
    equipment_channel = _artifact_channel(
        equipment["calibration_channel_id"], "EQUIPMENT"
    )
    drifted = equipment | {
        "provenance": equipment["provenance"] | {"checkpoint_sha256": "d" * 64}
    }
    assert not compatible_channel(equipment_channel, drifted)


def test_calibration_golden_and_strict_json():
    repo_root = Path(__file__).resolve().parents[1]
    result = run_calibration_golden(
        repo_root / "fixtures/golden_sport_calibration_001.json"
    )
    assert result["golden_passed"]
    json.dumps(result, allow_nan=False, sort_keys=True)
