"""Calibration fitting, canonical fingerprinting, and provenance compatibility."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict

from .contracts import (
    CALIBRATED_FUSION_VERSION,
    SPORT_EVIDENCE_CALIBRATION_CONTRACT_VERSION,
    SPORT_TYPE_CALIBRATION_DATASET_VERSION,
    CalibrationChannelStatus,
    GroundTruthSportType,
    SportCalibrationFitConfig,
    strict_json,
)
from .dataset import semantic_dataset_payload
from .platt import CalibrationSample, evaluate_channel


def fit_calibration_artifact(
    dataset: dict[str, object], config: SportCalibrationFitConfig | None = None
) -> dict[str, object]:
    if dataset.get("contract_version") != SPORT_TYPE_CALIBRATION_DATASET_VERSION:
        raise ValueError("incompatible calibration dataset")
    resolved = config or SportCalibrationFitConfig()
    annotations = dataset.get("annotations", {})
    by_channel = defaultdict(list)
    provenance: dict[str, dict[str, object]] = {}
    provenance_signatures: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    for summary in dataset.get("source_samples", []):
        annotation = annotations.get(summary["source_video_id"])
        if not annotation:
            continue
        if annotation["target_sport_type"] not in {"SKI", "SNOWBOARD"}:
            continue
        if (
            annotation["annotation_source"] != "USER_MANUAL"
            or annotation["intended_target_confirmation"] != "CONFIRMED"
            or annotation["video_sha256"] != summary["video_sha256"]
        ):
            continue
        if summary["raw_direction"] is None:
            continue
        channel = summary["calibration_channel_id"]
        relevant = _calibration_provenance(summary)
        provenance_signatures[channel].add(tuple(sorted(relevant.items())))
        by_channel[channel].append(
            CalibrationSample(
                summary["source_video_id"],
                summary["raw_direction"],
                int(annotation["target_sport_type"] == GroundTruthSportType.SNOWBOARD.value),
            )
        )
        provenance[channel] = summary.get("provenance", {})
    cv_started = time.perf_counter()
    evaluations = {
        channel: (
            {"status": CalibrationChannelStatus.CALIBRATION_CHANNEL_PROVENANCE_MISMATCH.value}
            if len(provenance_signatures[channel]) != 1
            or any(
                key == "_missing_required_provenance" and value
                for key, value in next(iter(provenance_signatures[channel]), ())
            )
            else evaluate_channel(samples, str(dataset["dataset_id"]), resolved)
        )
        for channel, samples in sorted(by_channel.items())
    }
    cv_seconds = time.perf_counter() - cv_started
    channels = []
    for channel in sorted(
        set(dataset.get("per_channel_usable_source_counts", {})) | set(by_channel)
    ):
        summary = next(
            item
            for item in dataset.get("source_samples", [])
            if item["calibration_channel_id"] == channel
        )
        evaluation = evaluations.get(channel) or {
            "status": CalibrationChannelStatus.REJECTED_INSUFFICIENT_DATA.value,
            "sample_count": 0,
            "ski_count": 0,
            "snowboard_count": 0,
            "effective_cv_folds": 0,
            "fold_assignment": {},
            "brier_score": None,
            "log_loss": None,
            "prior_only_brier_score": None,
            "prior_only_log_loss": None,
            "brier_skill_vs_prior": None,
            "log_loss_improvement_vs_prior": None,
            "classification_accuracy_at_0_5": None,
            "ece_5_bin": None,
        }
        defaults = {
            "sample_count": 0,
            "ski_count": 0,
            "snowboard_count": 0,
            "effective_cv_folds": 0,
            "fold_assignment": {},
            "brier_score": None,
            "log_loss": None,
            "prior_only_brier_score": None,
            "prior_only_log_loss": None,
            "brier_skill_vs_prior": None,
            "log_loss_improvement_vs_prior": None,
            "classification_accuracy_at_0_5": None,
            "ece_5_bin": None,
        }
        evaluation = {**defaults, **evaluation}
        accepted = evaluation["status"] == CalibrationChannelStatus.ACCEPTED_RESEARCH_CALIBRATION
        final_fit = evaluation.get("final_fit", {})
        total = evaluation["ski_count"] + evaluation["snowboard_count"]
        channels.append(
            {
                "calibration_channel_id": channel,
                "provider_name": summary["provider_name"],
                "evidence_kind": summary["evidence_kind"],
                "raw_feature": "snowboard_support_minus_ski_support",
                "model_type": "PLATT_SCALAR_LOGISTIC",
                "slope_a": final_fit.get("slope_a") if accepted else None,
                "intercept_b": final_fit.get("intercept_b") if accepted else None,
                "training_ski_count": evaluation["ski_count"],
                "training_snowboard_count": evaluation["snowboard_count"],
                "training_snowboard_prior": evaluation["snowboard_count"] / total
                if total
                else None,
                "oof_metrics": {
                    key: value for key, value in evaluation.items() if key != "final_fit"
                },
                "status": evaluation["status"],
                "provenance": provenance.get(channel, summary.get("provenance", {})),
                "research_only": True,
            }
        )
    accepted_count = sum(
        item["status"] == CalibrationChannelStatus.ACCEPTED_RESEARCH_CALIBRATION
        for item in channels
    )
    payload = {
        "contract_version": SPORT_EVIDENCE_CALIBRATION_CONTRACT_VERSION,
        "calibration_artifact_sha256": "",
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "dataset_sha256": hashlib.sha256(
                strict_json(semantic_dataset_payload(dataset)).encode()
            ).hexdigest(),
            "independent_labeled_source_count": dataset["labeled_source_count"],
            "ski_source_count": dataset["ski_labeled_source_count"],
            "snowboard_source_count": dataset["snowboard_labeled_source_count"],
        },
        "fit_config": resolved.to_dict(),
        "channels": channels,
        "fusion": {
            "version": CALIBRATED_FUSION_VERSION,
            "fusion_prior_snowboard": resolved.fusion_prior_snowboard,
            "same_kind_combination": "MEAN_LLR",
            "cross_kind_combination": "SUM_LLR",
            "calibrated_conflict_probability": resolved.calibrated_conflict_probability,
        },
        "status": "RESEARCH_CALIBRATION_AVAILABLE"
        if accepted_count
        else "INSUFFICIENT_LABELED_SPORT_TYPE_GT",
        "performance": {
            "cross_validation_seconds": cv_seconds,
            "final_fit_seconds": 0.0,
        },
        "limitations": [
            "RESEARCH_ONLY",
            "CALIBRATED_FUSION_DOES_NOT_CONTROL_ROUTING",
            "FUSION_PRIOR_IS_RESEARCH_NEUTRAL_NOT_PRODUCT_PREVALENCE",
        ],
    }
    payload["calibration_artifact_sha256"] = artifact_fingerprint(payload)
    return payload


def artifact_fingerprint(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("calibration_artifact_sha256", None)
    canonical.pop("performance", None)
    return hashlib.sha256(strict_json(canonical).encode()).hexdigest()


def _calibration_provenance(summary: dict[str, object]) -> dict[str, object]:
    provenance = summary.get("provenance", {})
    required = ["model_id", "checkpoint_sha256"]
    if summary.get("evidence_kind") == "VISUAL_CLASSIFIER":
        required.append("visual_prompt_sha256")
    result = {key: provenance.get(key) for key in required}
    if any(value is None for value in result.values()):
        return {**result, "_missing_required_provenance": True}
    return result


def validate_artifact_fingerprint(payload: dict[str, object]) -> bool:
    return payload.get("calibration_artifact_sha256") == artifact_fingerprint(payload)


def compatible_channel(channel: dict[str, object], summary: dict[str, object]) -> bool:
    if channel.get("calibration_channel_id") != summary.get("calibration_channel_id"):
        return False
    expected = channel.get("provenance", {})
    actual = summary.get("provenance", {})
    required = ["model_id", "checkpoint_sha256"]
    if channel.get("evidence_kind") == "VISUAL_CLASSIFIER":
        required.append("visual_prompt_sha256")
    return all(expected.get(key) and expected.get(key) == actual.get(key) for key in required)
