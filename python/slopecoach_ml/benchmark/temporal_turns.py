"""A4 identity-safe temporal pose and provisional turn engineering benchmark."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import PoseFrame
from slopecoach_ml.temporal import (
    TargetPoseSample,
    TemporalPoseConfig,
    stabilize_target_pose_stream,
)
from slopecoach_ml.turns import (
    ReferencePeakDetector,
    TurnSegmentationConfig,
    build_turn_signal,
    detect_zero_crossings,
    segment_turns,
    segmentation_summary,
)

from .target_identity import benchmark_target_identity_frames


class TemporalTurnCollector:
    def __init__(self, *, keep_images: bool = False) -> None:
        self.samples: list[TargetPoseSample] = []
        self.images: dict[int, bytes] = {}
        self.keep_images = keep_images

    def observe(self, frame, observation: dict[str, Any], pose_frame: PoseFrame | None) -> None:
        active_track_id = observation["active_track_id"]
        active = next(
            (track for track in observation["tracks"] if track["track_id"] == active_track_id),
            None,
        )
        target_pose = None
        if pose_frame is not None and active is not None and active["detection_id"] is not None:
            target_pose = next(
                (
                    person
                    for person in pose_frame.persons
                    if person.detection_id == active["detection_id"]
                ),
                None,
            )
        self.samples.append(
            TargetPoseSample(
                observation["timestamp_us"],
                observation["frame_index"],
                observation["target_id"],
                active_track_id,
                TargetIdentityState(observation["identity_state"]),
                observation["identity_confidence"],
                frame.geometry,
                target_pose,
                tuple(observation["limitations"]),
            )
        )
        if self.keep_images:
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError("DEBUG_DEPENDENCY_MISSING: opencv-python") from error
            ok, encoded = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise RuntimeError("TEMPORAL_DEBUG_FRAME_ENCODE_FAILED")
            self.images[frame.frame_index] = encoded.tobytes()


def benchmark_temporal_turns_frames(
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
    temporal_config: TemporalPoseConfig | None = None,
    turn_config: TurnSegmentationConfig | None = None,
    collector: TemporalTurnCollector | None = None,
    appearance_encoder: Any | None = None,
    target_identity_gt_status: str = "NOT_AVAILABLE",
) -> dict[str, Any]:
    temporal_settings = temporal_config or TemporalPoseConfig()
    turn_settings = turn_config or TurnSegmentationConfig()
    temporal_settings.validate()
    turn_settings.validate()
    sink = collector or TemporalTurnCollector()
    started = time.perf_counter()
    identity_report = benchmark_target_identity_frames(
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
        frame_observer=sink.observe,
        appearance_encoder=appearance_encoder,
        ground_truth_status=target_identity_gt_status,
    )
    temporal = stabilize_target_pose_stream(sink.samples, temporal_settings)
    stage = time.perf_counter()
    signal = build_turn_signal(
        temporal.samples, minimum_confidence=turn_settings.minimum_signal_confidence
    )
    turn_signal_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    peaks = ReferencePeakDetector().detect(signal, turn_settings)
    crossings = detect_zero_crossings(signal, turn_settings.zero_crossing_tolerance)
    segments = segment_turns(signal, peaks, crossings, turn_settings)
    turn_segmentation_seconds = time.perf_counter() - stage
    temporal_counts = Counter()
    for sample in temporal.samples:
        temporal_counts["observed"] += sample.observed_joint_count
        temporal_counts["interpolated"] += sample.interpolated_joint_count
        temporal_counts["missing"] += sample.missing_joint_count
    positive = sum(peak.value > 0 for peak in peaks)
    negative = sum(peak.value < 0 for peak in peaks)
    valid_signal = sum(sample.value is not None for sample in signal)
    locked = sum(sample.identity_state is TargetIdentityState.LOCKED for sample in sink.samples)
    signal_status = (
        "PASS"
        if valid_signal and segments
        else "PASS_WITH_LIMITATIONS"
        if valid_signal
        else "NOT_ANALYZABLE"
    )
    report = {
        "benchmark_contract_version": "ski-bench-temporal-turns-v1",
        "input_kind": "REAL_VIDEO",
        "runtime": identity_report["runtime"],
        "models": identity_report["models"],
        "config": {
            "profile": "RESEARCH_DEFAULTS_A4",
            "temporal_pose": asdict(temporal_settings),
            "turn_segmentation": asdict(turn_settings),
            "peak_detector": "ReferencePeakDetector",
            "SCIPY_STATUS": "NOT_CONFIGURED_OPTIONAL",
        },
        "video": identity_report["video"],
        "sampling": identity_report["sampling"],
        "identity_input": {
            "target_identity_gt_annotation_status": "DEFERRED",
            "target_identity_gt_status": target_identity_gt_status,
            "target_identity_accuracy_status": "UNKNOWN",
            "identity_locked_frame_count": locked,
            "identity_unsafe_frame_count": len(sink.samples) - locked,
        },
        "temporal_pose": {
            "temporal_segment_count": temporal.temporal_segment_count,
            "observed_joint_count": temporal_counts["observed"],
            "interpolated_joint_count": temporal_counts["interpolated"],
            "missing_joint_count": temporal_counts["missing"],
            "filter_reset_count": temporal.filter_reset_count,
            "short_gap_interpolation_count": temporal.short_gap_interpolation_count,
            "long_gap_unfilled_count": temporal.long_gap_unfilled_count,
        },
        "stability": temporal.stability,
        "turn_signal": {
            "valid_sample_count": valid_signal,
            "missing_sample_count": len(signal) - valid_signal,
            "positive_peak_count": positive,
            "negative_peak_count": negative,
            "zero_crossing_count": len(crossings),
        },
        "turn_segmentation": {
            **segmentation_summary(segments),
            "REAL_TURN_SEGMENTATION_STATUS": (
                "EXECUTED_PROVISIONAL_CANDIDATES"
                if segments
                else "NOT_ANALYZABLE_INSUFFICIENT_CONTINUOUS_TARGET_POSE"
            ),
            "TURN_SEGMENTATION_GT_STATUS": "NOT_AVAILABLE",
            "turn_precision": None,
            "turn_recall": None,
            "turn_f1": None,
            "TURN_SEGMENTATION_ENGINEERING_STATUS": signal_status,
        },
        "performance": {
            "detector_total_seconds": identity_report["performance"]["detector_total_seconds"],
            "tracking_identity_total_seconds": identity_report["performance"][
                "tracking_total_seconds"
            ]
            + identity_report["performance"]["identity_total_seconds"],
            "pose_total_seconds": identity_report["performance"]["pose_total_seconds"],
            "interpolation_total_seconds": temporal.interpolation_seconds,
            "stabilization_total_seconds": temporal.stabilization_seconds,
            "turn_signal_total_seconds": turn_signal_seconds,
            "turn_segmentation_total_seconds": turn_segmentation_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "validation": {
            "TEMPORAL_POSE_VALIDATION": "PASS_WITH_LIMITATIONS"
            if temporal.temporal_segment_count
            else "NOT_ANALYZABLE",
            "A4_ENGINEERING_VALIDATION": "PASS_WITH_LIMITATIONS",
            "A4_PRODUCT_VALIDATION": "BLOCKED_BY_IDENTITY_GT",
        },
        "limitations": [
            "IMAGE_SPACE_2D_PROXY_ONLY",
            "TARGET_IDENTITY_ACCURACY_UNKNOWN",
            "TURN_SEGMENTATION_GT_NOT_AVAILABLE",
            "NO_DIAGNOSIS_OR_PHYSICAL_EDGE_ANGLE",
            "PYTHON_RESEARCH_REFERENCE_ONLY",
        ],
        "temporal_trace": [sample.to_dict() for sample in temporal.samples],
        "turn_signal_samples": [sample.to_dict() for sample in signal],
        "turn_events": [peak.to_dict() for peak in peaks],
        "zero_crossings": [crossing.to_dict() for crossing in crossings],
        "turn_segments": [segment.to_dict() for segment in segments],
    }
    return report
