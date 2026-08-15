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
    execute_sport_evidence_providers,
    extract_uncalibrated_sport_cues,
    resolve_sport_type,
    sport_specific_analysis_allowed,
    summarize_provider_kind,
)

from .biomechanics_features import benchmark_biomechanics_frames
from .sport_type_collector import SportTypeBenchmarkCollector

SPORT_TYPE_BENCHMARK_CONTRACT_VERSION = "ski-bench-sport-type-v2"


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
    sink = collector or SportTypeBenchmarkCollector()
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
        collector=sink,
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
    provider_results = execute_sport_evidence_providers(
        providers, tuple(getattr(sink, "sport_contexts", ()))
    )
    provider_seconds = time.perf_counter() - stage
    if hasattr(sink, "release_frame_contexts"):
        sink.release_frame_contexts()
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
    equipment_summary = summarize_provider_kind(provider_results, SportEvidenceKind.EQUIPMENT)
    visual_summary = summarize_provider_kind(provider_results, SportEvidenceKind.VISUAL_CLASSIFIER)
    equipment_providers = [
        provider for provider in providers if provider.kind is SportEvidenceKind.EQUIPMENT
    ]
    equipment_debug = [
        item
        for provider in equipment_providers
        for item in getattr(provider, "last_debug_frames", ())
    ]
    equipment_execution = _equipment_execution_summary(equipment_providers, equipment_summary)
    equipment_performance = _equipment_performance(equipment_providers)
    upstream_perf = upstream["performance"]
    report = {
        "benchmark_contract_version": SPORT_TYPE_BENCHMARK_CONTRACT_VERSION,
        "input_kind": "REAL_VIDEO",
        "video": upstream["video"],
        "sampling": upstream["sampling"],
        "models": upstream["models"],
        "equipment_models": [
            provider.provenance()
            for provider in equipment_providers
            if hasattr(provider, "provenance")
        ],
        "identity_input": upstream["identity_input"],
        "temporal_input": upstream["temporal_input"],
        "biomechanics_input": {
            "contract_version": upstream["biomechanics_result"]["contract_version"],
            "feature_schema_version": BIOMECHANICS_FEATURE_SCHEMA_VERSION,
            "feature_registry_sha256": FEATURE_REGISTRY_SHA256,
        },
        "sport_type": sport_result.to_dict(),
        "provider_validation": {
            "EQUIPMENT_SPORT_PROVIDER_STATUS": equipment_summary["overall_status"],
            "VISUAL_SPORT_PROVIDER_STATUS": visual_summary["overall_status"],
            "provider_kind_summaries": [equipment_summary, visual_summary],
            "POSE_SPORT_EVIDENCE_STATUS": "UNCALIBRATED_MEASUREMENTS_ONLY",
            "TEMPORAL_SPORT_EVIDENCE_STATUS": "UNCALIBRATED_MEASUREMENTS_ONLY",
        },
        "equipment_evidence": equipment_execution,
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
            **equipment_performance,
            "total_seconds": time.perf_counter() - started,
        },
        "validation": {
            "REAL_SPORT_TYPE_STATUS": real_status,
            "A6_1_ENGINEERING_VALIDATION": (
                "PASS"
                if equipment_summary["overall_status"]
                in {"EXECUTED_NO_EVIDENCE", "EXECUTED_WITH_EVIDENCE"}
                else "PASS_WITH_LIMITATIONS"
            ),
            "A6_1_AUTO_CLASSIFICATION_VALIDATION": (
                "ENGINEERING_EVIDENCE_ONLY"
                if auto.primary_evidence_kinds
                else "NOT_VALIDATED_NO_PRIMARY_EVIDENCE_ON_REAL_CLIP"
                if equipment_summary["executed_provider_count"]
                else "NOT_VALIDATED_PROVIDER_UNAVAILABLE"
            ),
            "A6_PRODUCT_VALIDATION": "BLOCKED_BY_SPORT_TYPE_GT",
            "A7_ENGINEERING_READINESS": "READY_WITH_LIMITATIONS",
            "AUTO_SPORT_TYPE_PRODUCT_READINESS": (
                "NOT_READY_GT_REQUIRED"
                if auto.primary_evidence_kinds
                else "NOT_READY_PRIMARY_PROVIDER_REQUIRED"
            ),
        },
        "limitations": list(sport_result.limitations)
        + (
            ["NO_CONFIGURED_EQUIPMENT_CLASSIFIER"]
            if equipment_summary["configured_provider_count"] == 0
            else []
        )
        + (
            ["NO_CONFIGURED_VISUAL_SPORT_CLASSIFIER"]
            if visual_summary["configured_provider_count"] == 0
            else []
        )
        + ["NO_SPORT_TYPE_GROUND_TRUTH", "NO_DIAGNOSIS_OR_SCORE"],
        "_upstream_biomechanics_report": upstream,
        "_equipment_debug_frames": equipment_debug,
    }
    return report


def _equipment_execution_summary(providers, kind_summary):
    concrete = [provider for provider in providers if hasattr(provider, "last_summary")]
    totals = {
        key: sum(getattr(provider, "last_summary", {}).get(key, 0) or 0 for provider in concrete)
        for key in (
            "eligible_locked_context_count",
            "selected_equipment_context_count",
            "contexts_below_target_size_threshold",
            "equipment_inference_context_count",
            "frames_with_associated_skis",
            "frames_with_associated_snowboard",
            "frames_with_both",
            "equipment_observation_count",
        )
    }
    means = [
        provider.last_summary.get("mean_detector_support")
        for provider in concrete
        if provider.last_summary.get("mean_detector_support") is not None
    ]
    medians = [
        provider.last_summary.get("median_detector_support")
        for provider in concrete
        if provider.last_summary.get("median_detector_support") is not None
    ]
    totals["mean_detector_support"] = sum(means) / len(means) if means else None
    totals["median_detector_support"] = sorted(medians)[len(medians) // 2] if medians else None
    overall = kind_summary["overall_status"]
    totals["REAL_PRIMARY_EQUIPMENT_EVIDENCE_STATUS"] = (
        "EXECUTED_WITH_PRIMARY_EVIDENCE"
        if overall == "EXECUTED_WITH_EVIDENCE"
        else "EXECUTED_NO_ELIGIBLE_LOCKED_CONTEXTS"
        if overall == "EXECUTED_NO_EVIDENCE" and totals["eligible_locked_context_count"] == 0
        else "EXECUTED_NO_ASSOCIATED_EQUIPMENT"
        if overall == "EXECUTED_NO_EVIDENCE"
        else "FAILED"
        if overall == "FAILED"
        else "NOT_CONFIGURED"
    )
    return totals


def _equipment_performance(providers):
    values = [getattr(provider, "last_performance", {}) for provider in providers]
    return {
        "equipment_model_load_seconds": sum(
            item.get("equipment_model_load_seconds") or 0.0 for item in values
        )
        if any(item.get("equipment_model_load_seconds") is not None for item in values)
        else None,
        "equipment_inference_total_seconds": sum(
            item.get("equipment_inference_total_seconds") or 0.0 for item in values
        ),
        "equipment_mean_inference_seconds": (
            sum(
                item.get("equipment_mean_inference_seconds") or 0.0
                for item in values
                if item.get("equipment_mean_inference_seconds") is not None
            )
            / sum(item.get("equipment_mean_inference_seconds") is not None for item in values)
            if any(item.get("equipment_mean_inference_seconds") is not None for item in values)
            else None
        ),
        "equipment_p95_inference_seconds": next(
            (
                item["equipment_p95_inference_seconds"]
                for item in values
                if item.get("equipment_p95_inference_seconds") is not None
            ),
            None,
        ),
        "equipment_association_total_seconds": sum(
            item.get("equipment_association_total_seconds") or 0.0 for item in values
        ),
    }


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
