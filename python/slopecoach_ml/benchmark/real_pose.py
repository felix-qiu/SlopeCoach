from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from statistics import mean
from typing import Any

from slopecoach_ml.biomechanics import knee_angle_2d
from slopecoach_ml.detection import DetectorProvider
from slopecoach_ml.pose import Joint
from slopecoach_ml.pose.mmpose_provider import MMPoseRTMWPoseProvider
from slopecoach_ml.reference import ReferenceAnalysisConfig
from slopecoach_ml.video import SampledFrame, inspect_video

REQUIRED = (Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def benchmark_real_pose_frames(
    *,
    input_path: str | Path,
    input_kind: str,
    frames: Iterable[SampledFrame],
    detector: DetectorProvider,
    pose_provider: MMPoseRTMWPoseProvider,
    detector_model: dict[str, Any],
    pose_model: dict[str, Any],
    config: ReferenceAnalysisConfig | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if input_kind not in {"REAL_VIDEO", "SYNTHETIC_PIPELINE_SMOKE"}:
        raise ValueError("real-pose input kind must be REAL_VIDEO or SYNTHETIC_PIPELINE_SMOKE")
    settings = config or ReferenceAnalysisConfig()
    started = clock()
    detector_seconds = pose_seconds = biomechanics_seconds = 0.0
    person_counts: list[int] = []
    frame_latencies: list[float] = []
    successful = required_visible = knee_available = out_of_frame = total_joints = 0
    pose_confidences: list[float] = []
    required_confidences: list[float] = []
    sampled = 0
    for sampled_frame in frames:
        frame_started = clock()
        sampled += 1
        stage = clock()
        detections = detector.detect(sampled_frame.image, sampled_frame.geometry)
        detector_seconds += clock() - stage
        person_counts.append(len(detections))
        stage = clock()
        pose_frame = pose_provider.estimate_detections(
            sampled_frame.image,
            detections,
            sampled_frame.geometry,
            timestamp_us=sampled_frame.timestamp_us,
            frame_index=sampled_frame.frame_index,
        )
        pose_seconds += clock() - stage
        successful += int(bool(pose_frame.persons) or not detections)
        for person in pose_frame.persons:
            total_joints += len(person.keypoints)
            out_of_frame += sum(
                not point.is_inside_frame(pose_frame.geometry)
                for point in person.keypoints.values()
            )
            pose_confidences.append(mean(point.confidence for point in person.keypoints.values()))
            required = [person.joint(joint) for joint in REQUIRED]
            if all(required):
                required_confidences.append(mean(point.confidence for point in required if point))
                required_visible += int(
                    all(point.is_inside_frame(pose_frame.geometry) for point in required if point)
                )
        stage = clock()
        if len(pose_frame.persons) == 1:
            angle = knee_angle_2d(
                pose_frame.persons[0],
                pose_frame.geometry,
                side="left",
                minimum_confidence=settings.min_joint_confidence,
                square_pixel_tolerance=settings.square_pixel_tolerance,
            )
            knee_available += int(angle is not None)
        biomechanics_seconds += clock() - stage
        frame_latencies.append(clock() - frame_started)
    finished = clock()
    ordered = sorted(frame_latencies)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1] if len(ordered) >= 20 else None
    detected_frames = sum(count > 0 for count in person_counts)
    single_frames = sum(count == 1 for count in person_counts)
    multi_frames = sum(count > 1 for count in person_counts)
    return {
        "benchmark_contract_version": "ski-bench-real-pose-v1",
        "input_kind": input_kind,
        "input_path": str(input_path),
        "video_metadata": inspect_video(input_path).to_dict(),
        "detector": {
            **detector_model,
            "detected_frame_ratio": _ratio(detected_frames, sampled),
            "mean_person_count": mean(person_counts) if person_counts else None,
            "single_person_frame_ratio": _ratio(single_frames, sampled),
            "multi_person_frame_ratio": _ratio(multi_frames, sampled),
        },
        "pose": {
            **pose_model,
            "attempted_frames": sampled,
            "successful_frames": successful,
            "coverage": _ratio(successful, sampled),
            "mean_person_confidence": mean(pose_confidences) if pose_confidences else None,
            "mean_required_joint_confidence": mean(required_confidences)
            if required_confidences
            else None,
            "required_joint_coverage": _ratio(required_visible, sum(person_counts)),
        },
        "biomechanics": {"left_knee_angle_2d_coverage": _ratio(knee_available, sampled)},
        "coordinate": {
            "canonical_space": "SourcePixel2D",
            "orientation": "CanonicalUpright",
            "mirrored": False,
            "out_of_frame_joint_ratio": _ratio(out_of_frame, total_joints),
        },
        "performance": {
            "total_processing_seconds": finished - started,
            "decode_seconds": None,
            "detector_seconds": detector_seconds,
            "pose_seconds": pose_seconds,
            "canonical_adapter_seconds": 0.0,
            "biomechanics_seconds": biomechanics_seconds,
            "mean_frame_latency_seconds": mean(frame_latencies) if frame_latencies else None,
            "p95_frame_latency_seconds": p95,
            "sampled_frame_count": sampled,
        },
        "warnings": [],
        "limitations": (["MULTIPLE_PERSONS_TARGET_IDENTITY_UNRESOLVED"] if multi_frames else []),
        "REAL_GT_STATUS": "NOT_AVAILABLE",
        "ground_truth_metrics": {
            "diagnosis_precision": None,
            "diagnosis_recall": None,
            "diagnosis_f1": None,
        },
    }
