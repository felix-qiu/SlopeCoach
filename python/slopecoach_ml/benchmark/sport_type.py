"""A6 real-video SportType benchmark composed on the single-pass A5.1 pipeline."""

from __future__ import annotations

import time

from slopecoach_ml.biomechanics import (
    BIOMECHANICS_FEATURE_SCHEMA_VERSION,
    FEATURE_REGISTRY_SHA256,
)
from slopecoach_ml.sport_type import (
    NotConfiguredEquipmentSportEvidenceProvider,
    NotConfiguredVisualSportEvidenceProvider,
    SportEvidenceKind,
    SportType,
    SportTypeConfig,
    extract_uncalibrated_sport_cues,
    resolve_sport_type,
    sport_specific_analysis_allowed,
)

from .biomechanics_features import benchmark_biomechanics_frames

SPORT_TYPE_BENCHMARK_CONTRACT_VERSION = "ski-bench-sport-type-v1"


def benchmark_sport_type_frames(
    *,
    input_path,
    frames,
    detector,
    pose_provider,
    detector_model,
    pose_model,
    device="cpu",
    sample_fps=5.0,
    model_load=None,
    warmup_frames=0,
    collector=None,
    appearance_encoder=None,
    target_identity_gt_status="NOT_AVAILABLE",
    user_selection: SportType | None = None,
    evidence_providers=None,
    config: SportTypeConfig | None = None,
):
    started = time.perf_counter()
    upstream = benchmark_biomechanics_frames(
        input_path=input_path,
        frames=frames,
        detector=detector,
        pose_provider=pose_provider,
        detector_model=detector_model,
        pose_model=pose_model,
        device=device,
        sample_fps=sample_fps,
        model_load=model_load,
        warmup_frames=warmup_frames,
        collector=collector,
        appearance_encoder=appearance_encoder,
        target_identity_gt_status=target_identity_gt_status,
    )
    stage = time.perf_counter()
    cues = extract_uncalibrated_sport_cues(upstream["biomechanics_result"])
    cue_seconds = time.perf_counter() - stage
    providers = tuple(
        evidence_providers
        if evidence_providers is not None
        else (
            NotConfiguredEquipmentSportEvidenceProvider(),
            NotConfiguredVisualSportEvidenceProvider(),
        )
    )
    stage = time.perf_counter()
    provider_results = tuple(provider.infer(None) for provider in providers)
    provider_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    sport_result = resolve_sport_type(
        provider_results,
        user_selection=user_selection,
        config=config,
        cue_measurements=cues,
    )
    fusion_seconds = time.perf_counter() - stage
    auto = sport_result.auto_decision
    real_status = _real_status(sport_result)
    upstream_perf = upstream["performance"]
    report = {
        "benchmark_contract_version": SPORT_TYPE_BENCHMARK_CONTRACT_VERSION,
        "input_kind": "REAL_VIDEO",
        "video": upstream["video"],
        "sampling": upstream["sampling"],
        "models": upstream["models"],
        "identity_input": upstream["identity_input"],
        "temporal_input": upstream["temporal_input"],
        "biomechanics_input": {
            "contract_version": upstream["biomechanics_result"]["contract_version"],
            "feature_schema_version": BIOMECHANICS_FEATURE_SCHEMA_VERSION,
            "feature_registry_sha256": FEATURE_REGISTRY_SHA256,
        },
        "sport_type": sport_result.to_dict(),
        "provider_validation": {
            "EQUIPMENT_SPORT_PROVIDER_STATUS": _provider_status(
                provider_results, SportEvidenceKind.EQUIPMENT
            ),
            "VISUAL_SPORT_PROVIDER_STATUS": _provider_status(
                provider_results, SportEvidenceKind.VISUAL_CLASSIFIER
            ),
            "POSE_SPORT_EVIDENCE_STATUS": "UNCALIBRATED_MEASUREMENTS_ONLY",
            "TEMPORAL_SPORT_EVIDENCE_STATUS": "UNCALIBRATED_MEASUREMENTS_ONLY",
        },
        "downstream_gate": {
            "GENERIC_POSE_ALLOWED": True,
            "GENERIC_BIOMECHANICS_ALLOWED": True,
            "SPORT_SPECIFIC_ANALYSIS_ALLOWED": sport_specific_analysis_allowed(sport_result),
            "reason": None
            if sport_specific_analysis_allowed(sport_result)
            else "SPORT_TYPE_UNKNOWN",
        },
        "ground_truth": {
            "SPORT_TYPE_GT_STATUS": "NOT_AVAILABLE",
            "sport_type_accuracy": None,
            "sport_type_precision": None,
            "sport_type_recall": None,
            "TARGET_IDENTITY_GT_ANNOTATION_STATUS": "DEFERRED",
            "TURN_SEGMENTATION_GT_STATUS": "NOT_AVAILABLE",
            "BIOMECHANICS_GT_STATUS": "NOT_AVAILABLE",
        },
        "performance": {
            "detector_total_seconds": upstream_perf["detector_total_seconds"],
            "tracking_identity_total_seconds": upstream_perf["tracking_identity_total_seconds"],
            "pose_total_seconds": upstream_perf["pose_total_seconds"],
            "temporal_total_seconds": upstream_perf["temporal_total_seconds"],
            "turn_total_seconds": upstream_perf["turn_total_seconds"],
            "biomechanics_total_seconds": (
                upstream_perf["biomechanics_frame_total_seconds"]
                + upstream_perf["biomechanics_aggregation_total_seconds"]
                + upstream_perf["biomechanics_turn_total_seconds"]
            ),
            "sport_cue_extraction_total_seconds": cue_seconds,
            "sport_evidence_provider_total_seconds": provider_seconds,
            "sport_fusion_total_seconds": fusion_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "validation": {
            "REAL_SPORT_TYPE_STATUS": real_status,
            "A6_ENGINEERING_VALIDATION": "PASS",
            "A6_AUTO_CLASSIFICATION_VALIDATION": (
                "ENGINEERING_EVIDENCE_ONLY"
                if auto.primary_evidence_kinds
                else "NOT_VALIDATED_NO_PRIMARY_PROVIDER"
            ),
            "A6_PRODUCT_VALIDATION": "BLOCKED_BY_PRIMARY_SPORT_EVIDENCE_AND_GT",
            "A7_ENGINEERING_READINESS": "READY_WITH_LIMITATIONS",
            "AUTO_SPORT_TYPE_PRODUCT_READINESS": (
                "NOT_READY_GT_REQUIRED"
                if auto.primary_evidence_kinds
                else "NOT_READY_PRIMARY_PROVIDER_REQUIRED"
            ),
        },
        "limitations": list(sport_result.limitations)
        + [
            "NO_CONFIGURED_EQUIPMENT_CLASSIFIER",
            "NO_CONFIGURED_VISUAL_SPORT_CLASSIFIER",
            "NO_SPORT_TYPE_GROUND_TRUTH",
            "NO_DIAGNOSIS_OR_SCORE",
        ],
        "_upstream_biomechanics_report": upstream,
    }
    return report


def _provider_status(results, kind):
    item = next((result for result in results if result.evidence_kind is kind), None)
    return item.status.value if item else "NOT_CONFIGURED"


def _real_status(result):
    if result.effective_source.value == "USER":
        return "USER_RESOLVED"
    return {
        "RESOLVED_AUTO": "AUTO_RESOLVED",
        "AMBIGUOUS": "AUTO_AMBIGUOUS",
        "CONFLICTING_PRIMARY_EVIDENCE": "AUTO_CONFLICTING_EVIDENCE",
        "INSUFFICIENT_PRIMARY_EVIDENCE": "AUTO_INSUFFICIENT_PRIMARY_EVIDENCE",
        "INSUFFICIENT_TOTAL_EVIDENCE": "AUTO_INSUFFICIENT_TOTAL_EVIDENCE",
    }[result.resolution_status.value]
