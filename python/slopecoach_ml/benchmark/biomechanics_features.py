"""A5 benchmark composed directly on the A4.1 temporal-turn pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from slopecoach_ml.biomechanics import (
    BIOMECHANICS_FEATURE_SCHEMA_VERSION,
    FEATURE_REGISTRY_SHA256,
    FEATURE_REGISTRY_V1,
    FIXED_ML_FEATURE_VECTOR_STATUS,
    BiomechanicsFeatureConfig,
    TemporalBiomechanicsResult,
    aggregate_frame_facts,
    compute_frame_biomechanics,
    compute_turn_biomechanics,
    derivative_aggregates,
    feature_coverage,
)
from slopecoach_ml.temporal import (
    TemporalPoseConfig,
    segment_body_scales,
    stabilize_target_pose_stream,
)
from slopecoach_ml.turns import (
    ReferencePeakDetector,
    TurnSegmentationConfig,
    build_turn_signal,
    detect_zero_crossings,
    segment_turns,
    valid_signal_runs,
)

from .temporal_turns import TemporalTurnCollector, benchmark_temporal_turns_frames


def benchmark_biomechanics_frames(
    *,
    input_path: str | Path,
    frames,
    detector,
    pose_provider,
    detector_model: dict[str, Any],
    pose_model: dict[str, Any],
    device: str = "cpu",
    sample_fps: float = 5.0,
    model_load: dict[str, float] | None = None,
    warmup_frames: int = 0,
    collector: TemporalTurnCollector | None = None,
    appearance_encoder: Any | None = None,
    target_identity_gt_status: str = "NOT_AVAILABLE",
) -> dict[str, Any]:
    started = time.perf_counter()
    sink = collector or TemporalTurnCollector()
    upstream = benchmark_temporal_turns_frames(
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
    temporal_config, turn_config = TemporalPoseConfig(), TurnSegmentationConfig()
    temporal = stabilize_target_pose_stream(sink.samples, temporal_config)
    signal = build_turn_signal(
        temporal.samples, minimum_confidence=turn_config.minimum_signal_confidence
    )
    peaks = ReferencePeakDetector().detect(signal, turn_config)
    crossings = detect_zero_crossings(
        signal,
        turn_config.zero_crossing_tolerance,
        minimum_signal_confidence=turn_config.minimum_signal_confidence,
    )
    turns = segment_turns(signal, peaks, crossings, turn_config)
    runs = valid_signal_runs(signal, turn_config)
    biomechanics_config = BiomechanicsFeatureConfig()
    stage = time.perf_counter()
    scales = segment_body_scales(temporal.samples)
    frame_facts = tuple(
        fact
        for sample in temporal.samples
        if sample.temporal_segment_id is not None
        for fact in compute_frame_biomechanics(
            sample, scales.get(sample.temporal_segment_id), biomechanics_config
        )
    )
    frame_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    aggregates = aggregate_frame_facts(frame_facts, biomechanics_config) + derivative_aggregates(
        frame_facts, biomechanics_config
    )
    coverage = feature_coverage(frame_facts)
    aggregation_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    run_timestamps = {
        run.signal_run_id: {sample.timestamp_us for _, sample in run.indexed_samples}
        for run in runs
    }
    turn_features = compute_turn_biomechanics(
        turns, frame_facts, run_timestamps, biomechanics_config
    )
    turn_seconds = time.perf_counter() - stage
    result = TemporalBiomechanicsResult(
        contract_version="temporal-biomechanics-v2",
        feature_schema_version=BIOMECHANICS_FEATURE_SCHEMA_VERSION,
        feature_registry_sha256=FEATURE_REGISTRY_SHA256,
        config=biomechanics_config,
        frame_facts=frame_facts,
        temporal_segment_features=aggregates,
        turn_features=turn_features,
        feature_coverage=coverage,
    )
    serialized_result = json.loads(result.to_json())
    trusted = len({fact.timestamp_us for fact in result.frame_facts if fact.temporal_segment_id})
    real_status = (
        "NOT_ANALYZABLE_NO_TRUSTED_TARGET_POSE"
        if trusted == 0
        else "EXECUTED_WITH_TURN_BIOMECHANICS"
        if result.turn_features
        else "EXECUTED_FRAME_AND_SEGMENT_FEATURES_NO_TURNS"
    )
    upstream_perf = upstream["performance"]
    report = {
        "benchmark_contract_version": "ski-bench-biomechanics-v2",
        "input_kind": "REAL_VIDEO",
        "runtime": upstream["runtime"],
        "models": upstream["models"],
        "config": {
            "profile": "RESEARCH_DEFAULTS_A5_1",
            "biomechanics": serialized_result["config"],
            "FIXED_ML_FEATURE_VECTOR_STATUS": FIXED_ML_FEATURE_VECTOR_STATUS,
        },
        "feature_schema_version": BIOMECHANICS_FEATURE_SCHEMA_VERSION,
        "feature_registry_sha256": FEATURE_REGISTRY_SHA256,
        "feature_registry": [
            {
                "feature_id": item.feature_id,
                "family": item.family.value,
                "scope": item.scope.value,
                "unit": item.unit,
                "description": item.description,
                "required_joints": [joint.value for joint in item.required_joints],
                "limitations": list(item.limitations),
            }
            for item in FEATURE_REGISTRY_V1
        ],
        "video": upstream["video"],
        "sampling": upstream["sampling"],
        "identity_input": upstream["identity_input"],
        "temporal_input": {
            "temporal_segment_count": temporal.temporal_segment_count,
        },
        "turn_input": {
            "turn_status": upstream["turn_segmentation"]["REAL_TURN_SEGMENTATION_STATUS"],
            "qualified_turn_count": len(result.turn_features),
            "TURN_SEGMENTATION_GT_STATUS": "NOT_AVAILABLE",
        },
        "frame_biomechanics": {
            "trusted_frame_count": trusted,
            "feature_coverage": result.feature_coverage,
        },
        "temporal_segment_features": [item.to_dict() for item in result.temporal_segment_features],
        "turn_biomechanics": [item.to_dict() for item in result.turn_features],
        "biomechanics_result": serialized_result,
        "performance": {
            "detector_total_seconds": upstream_perf["detector_total_seconds"],
            "tracking_identity_total_seconds": upstream_perf["tracking_identity_total_seconds"],
            "pose_total_seconds": upstream_perf["pose_total_seconds"],
            "temporal_total_seconds": upstream_perf["interpolation_total_seconds"]
            + upstream_perf["stabilization_total_seconds"],
            "turn_total_seconds": upstream_perf["turn_signal_total_seconds"]
            + upstream_perf["turn_segmentation_total_seconds"],
            "biomechanics_frame_total_seconds": frame_seconds,
            "biomechanics_aggregation_total_seconds": aggregation_seconds,
            "biomechanics_turn_total_seconds": turn_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "ground_truth": {
            "TARGET_IDENTITY_GT_ANNOTATION_STATUS": "DEFERRED",
            "TARGET_IDENTITY_ACCURACY_STATUS": "UNKNOWN",
            "TURN_SEGMENTATION_GT_STATUS": "NOT_AVAILABLE",
            "BIOMECHANICS_GT_STATUS": "NOT_AVAILABLE",
            "feature_accuracy": None,
            "biomechanics_mae": None,
        },
        "validation": {
            "REAL_BIOMECHANICS_STATUS": real_status,
            "A5_1_ENGINEERING_VALIDATION": "PASS" if trusted else "PASS_WITH_LIMITATIONS",
            "A5_PRODUCT_VALIDATION": "BLOCKED_BY_GT",
        },
        "limitations": list(result.limitations),
        "_upstream_debug_report": upstream,
    }
    return report
