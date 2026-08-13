from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

from slopecoach_ml.biomechanics import knee_angle_2d
from slopecoach_ml.detection import DetectorProvider
from slopecoach_ml.pose import Joint, PoseFrame
from slopecoach_ml.pose.mmpose_provider import MMPoseRTMWPoseProvider
from slopecoach_ml.pose.overlay import render_debug_overlay
from slopecoach_ml.quality import VideoQualityGate
from slopecoach_ml.reference import ReferenceAnalysisConfig
from slopecoach_ml.video import SampledFrame, inspect_video

LEFT_REQUIRED = (Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE)
RIGHT_REQUIRED = (Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE)
MINIMUM_P95_SAMPLES = 20


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], percentile: float, *, minimum: int = 1) -> float | None:
    if len(values) < minimum:
        return None
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def _required_quality(person: Any, joints: tuple[Joint, ...], geometry: Any, minimum: float):
    points = [person.joint(joint) for joint in joints]
    confidence_ok = [point is not None and point.confidence >= minimum for point in points]
    visible = [point is not None and point.is_inside_frame(geometry) for point in points]
    return points, confidence_ok, visible


def _frame_status(
    pose_frame: PoseFrame,
    *,
    left_confidence_ok: list[bool],
    left_visible: list[bool],
    square_pixel_tolerance: float,
) -> str:
    if not pose_frame.persons:
        return "NO_PERSON_DETECTED"
    if len(pose_frame.persons) > 1:
        return "MULTIPLE_PERSONS_TARGET_UNRESOLVED"
    if not math.isclose(
        pose_frame.geometry.pixel_aspect_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=square_pixel_tolerance,
    ):
        return "NON_SQUARE_PIXEL_ASPECT_RATIO_UNSUPPORTED"
    if not all(left_confidence_ok):
        return "REQUIRED_JOINTS_LOW_CONFIDENCE"
    if not all(left_visible):
        return "REQUIRED_JOINTS_OUT_OF_FRAME"
    return "POSE_FRAME_OK"


def _failure_reasons(
    pose_frame: PoseFrame,
    *,
    minimum_confidence: float,
    square_pixel_tolerance: float,
) -> list[str]:
    if not pose_frame.persons:
        return ["NO_PERSON"]
    if len(pose_frame.persons) > 1:
        return ["MULTIPLE_PERSONS"]
    if not math.isclose(
        pose_frame.geometry.pixel_aspect_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=square_pixel_tolerance,
    ):
        return ["NON_SQUARE_PIXEL_ASPECT_RATIO"]
    person = pose_frame.persons[0]
    reasons = []
    for joint in LEFT_REQUIRED:
        point = person.joint(joint)
        label = joint.value.upper()
        if point is None or point.confidence < minimum_confidence:
            reasons.append(f"LOW_CONFIDENCE_{label}")
        if point is not None and not point.is_inside_frame(pose_frame.geometry):
            reasons.append(f"{label}_OUT_OF_FRAME")
    return reasons


def _temporal_observation(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    previous_person = previous["pose_frame"].persons[0]
    current_person = current["pose_frame"].persons[0]
    previous_scale = math.hypot(previous_person.bbox.width_px, previous_person.bbox.height_px)
    current_scale = math.hypot(current_person.bbox.width_px, current_person.bbox.height_px)
    scale = mean((previous_scale, current_scale))
    joint_displacements = []
    for joint in Joint:
        first, second = previous_person.joint(joint), current_person.joint(joint)
        if first is not None and second is not None and scale > 0:
            joint_displacements.append(
                math.hypot(second.x_px - first.x_px, second.y_px - first.y_px) / scale
            )
    first_center = (
        previous_person.bbox.x_px + previous_person.bbox.width_px / 2,
        previous_person.bbox.y_px + previous_person.bbox.height_px / 2,
    )
    second_center = (
        current_person.bbox.x_px + current_person.bbox.width_px / 2,
        current_person.bbox.y_px + current_person.bbox.height_px / 2,
    )
    result = {
        "normalized_bbox_center_displacement": math.hypot(
            second_center[0] - first_center[0], second_center[1] - first_center[1]
        )
        / scale,
        "normalized_joint_displacement": mean(joint_displacements),
        "confidence_delta": abs(
            current["mean_joint_confidence"] - previous["mean_joint_confidence"]
        ),
    }
    if previous["left_knee_angle"] is not None and current["left_knee_angle"] is not None:
        result["left_knee_angle_delta_degrees"] = abs(
            current["left_knee_angle"] - previous["left_knee_angle"]
        )
    return result


def select_debug_frame_indices(
    observations: list[dict[str, Any]], *, max_frames: int = 10
) -> list[int]:
    if max_frames <= 0 or not observations:
        return []
    selected: list[int] = []

    def add(index: int | None) -> None:
        if index is not None and index not in selected and len(selected) < max_frames:
            selected.append(index)

    valid = [item for item in observations if item["status"] == "POSE_FRAME_OK"]
    if valid:
        add(valid[0]["frame_index"])
        add(valid[len(valid) // 2]["frame_index"])
        add(valid[-1]["frame_index"])
    with_confidence = [item for item in observations if item["mean_joint_confidence"] is not None]
    if with_confidence:
        add(min(with_confidence, key=lambda item: item["mean_joint_confidence"])["frame_index"])
    with_instability = [
        item for item in observations if item["normalized_joint_displacement"] is not None
    ]
    if with_instability:
        add(
            max(with_instability, key=lambda item: item["normalized_joint_displacement"])[
                "frame_index"
            ]
        )
    for status in ("NO_PERSON_DETECTED", "MULTIPLE_PERSONS_TARGET_UNRESOLVED"):
        item = next((item for item in observations if item["status"] == status), None)
        add(item["frame_index"] if item else None)
    return selected


class RealPoseDebugCollector:
    """Bounded research debug writer; compressed frames are not benchmark inputs or tracks."""

    def __init__(self) -> None:
        self._records: dict[int, tuple[bytes, PoseFrame | None, dict[str, Any]]] = {}

    def observe(
        self, sampled_frame: SampledFrame, pose_frame: PoseFrame | None, observation: dict[str, Any]
    ) -> None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OPENMMLAB_DEPENDENCY_MISSING: opencv-python") from error
        ok, encoded = cv2.imencode(".jpg", sampled_frame.image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("DEBUG_FRAME_ENCODE_FAILED")
        self._records[sampled_frame.frame_index] = (encoded.tobytes(), pose_frame, observation)

    def write(
        self,
        output_dir: str | Path,
        observations: list[dict[str, Any]],
        *,
        provider_name: str,
        max_frames: int = 10,
    ) -> dict[str, Any]:
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise RuntimeError("OPENMMLAB_DEPENDENCY_MISSING: opencv-python/numpy") from error
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        selected = select_debug_frame_indices(observations, max_frames=max_frames)
        overlay_paths: list[str] = []
        contact_images = []
        for frame_index in selected:
            encoded, pose_frame, observation = self._records[frame_index]
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("DEBUG_FRAME_DECODE_FAILED")
            confidence = observation["mean_joint_confidence"]
            label = (
                f"frame={frame_index} t={observation['timestamp_us'] / 1_000_000:.3f}s "
                f"status={observation['status']} conf="
                f"{confidence if confidence is not None else 'null'}"
            )
            overlay_path = destination / f"frame_{frame_index:06d}.jpg"
            if pose_frame is not None:
                render_debug_overlay(
                    image,
                    pose_frame,
                    overlay_path,
                    provider_name=provider_name,
                    annotation=label,
                )
                rendered = cv2.imread(str(overlay_path))
            else:
                rendered = image.copy()
                cv2.putText(
                    rendered,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                if not cv2.imwrite(str(overlay_path), rendered):
                    raise RuntimeError("DEBUG_OVERLAY_WRITE_FAILED")
            overlay_paths.append(str(overlay_path))
            contact_images.append(rendered)
        contact_sheet_path = None
        if contact_images:
            tile_width = 360
            tiles = []
            for image in contact_images:
                height, width = image.shape[:2]
                tile_height = max(1, round(height * tile_width / width))
                tiles.append(cv2.resize(image, (tile_width, tile_height)))
            row_height = max(tile.shape[0] for tile in tiles)
            padded = []
            for tile in tiles:
                bottom = row_height - tile.shape[0]
                padded.append(cv2.copyMakeBorder(tile, 0, bottom, 0, 0, cv2.BORDER_CONSTANT))
            rows = []
            for offset in range(0, len(padded), 3):
                row = padded[offset : offset + 3]
                while len(row) < 3:
                    row.append(np.zeros_like(padded[0]))
                rows.append(cv2.hconcat(row))
            contact_sheet_path = destination / "contact_sheet.jpg"
            if not cv2.imwrite(str(contact_sheet_path), cv2.vconcat(rows)):
                raise RuntimeError("DEBUG_CONTACT_SHEET_WRITE_FAILED")
        return {
            "selected_frame_indices": selected,
            "overlay_paths": overlay_paths,
            "contact_sheet": str(contact_sheet_path) if contact_sheet_path else None,
        }


def benchmark_real_pose_frames(
    *,
    input_path: str | Path,
    input_kind: str,
    frames: Iterable[SampledFrame],
    detector: DetectorProvider,
    pose_provider: MMPoseRTMWPoseProvider,
    detector_model: dict[str, Any],
    pose_model: dict[str, Any],
    device: str = "cpu",
    sample_fps: float | None = None,
    model_load: dict[str, float] | None = None,
    warmup_frames: int = 0,
    frame_observer: Callable[[SampledFrame, PoseFrame | None, dict[str, Any]], None] | None = None,
    config: ReferenceAnalysisConfig | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if input_kind not in {"REAL_VIDEO", "SYNTHETIC_PIPELINE_SMOKE"}:
        raise ValueError("real-pose input kind must be REAL_VIDEO or SYNTHETIC_PIPELINE_SMOKE")
    settings = config or ReferenceAnalysisConfig()
    settings.validate()
    metadata = inspect_video(input_path)
    quality = VideoQualityGate().evaluate(metadata)
    started = clock()
    decode_seconds = detector_seconds = pose_seconds = adapter_seconds = biomechanics_seconds = 0.0
    debug_capture_seconds = 0.0
    detector_latencies: list[float] = []
    pose_latencies: list[float] = []
    frame_latencies: list[float] = []
    person_counts: list[int] = []
    detection_confidences: list[float] = []
    joint_confidences: list[float] = []
    left_required_confidences: list[float] = []
    right_required_confidences: list[float] = []
    observed_raw_joint_counts: set[int] = set()
    left_coverage = right_coverage = total_required = 0
    out_of_frame = total_joints = invalid_coordinates = nonfinite_coordinates = 0
    attempted_persons = successful_persons = 0
    failure_counts: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    observations: list[dict[str, Any]] = []
    temporal: list[dict[str, float]] = []
    knee_angles: list[float] = []
    previous_single: dict[str, Any] | None = None
    iterator = iter(frames)
    while True:
        decode_started = clock()
        try:
            sampled_frame = next(iterator)
        except StopIteration:
            decode_seconds += clock() - decode_started
            break
        decode_seconds += clock() - decode_started
        pipeline_started = clock()
        stage = clock()
        detections = detector.detect(sampled_frame.image, sampled_frame.geometry)
        detector_latency = clock() - stage
        detector_seconds += detector_latency
        detector_latencies.append(detector_latency)
        person_counts.append(len(detections))
        detection_confidences.extend(detection.confidence for detection in detections)
        attempted_persons += len(detections)
        stage = clock()
        try:
            pose_frame = pose_provider.estimate_detections(
                sampled_frame.image,
                detections,
                sampled_frame.geometry,
                timestamp_us=sampled_frame.timestamp_us,
                frame_index=sampled_frame.frame_index,
            )
        except (RuntimeError, ValueError) as error:
            pose_latency = clock() - stage
            pose_seconds += pose_latency
            pose_latencies.append(pose_latency)
            status = "POSE_PROVIDER_FAILED"
            statuses[status] += 1
            failure_counts["POSE_INFERENCE_ERROR"] += 1
            observation = {
                "timestamp_us": sampled_frame.timestamp_us,
                "frame_index": sampled_frame.frame_index,
                "detection_count": len(detections),
                "status": status,
                "mean_joint_confidence": None,
                "normalized_joint_displacement": None,
                "left_knee_angle_2d_degrees": None,
                "detector_latency_seconds": detector_latency,
                "pose_latency_seconds": pose_latency,
                "error": f"{type(error).__name__}: {error}",
            }
            observations.append(observation)
            previous_single = None
            frame_latencies.append(clock() - pipeline_started)
            if frame_observer:
                observer_started = clock()
                frame_observer(sampled_frame, None, observation)
                debug_capture_seconds += clock() - observer_started
            continue
        pose_latency = clock() - stage
        pose_seconds += pose_provider.last_backend_seconds
        adapter_seconds += pose_provider.last_adapter_seconds
        pose_latencies.append(pose_provider.last_backend_seconds)
        successful_persons += len(pose_frame.persons)
        if pose_provider.last_raw_joint_count is not None:
            observed_raw_joint_counts.add(pose_provider.last_raw_joint_count)
        frame_joint_confidences = [
            point.confidence for person in pose_frame.persons for point in person.keypoints.values()
        ]
        joint_confidences.extend(frame_joint_confidences)
        frame_out = 0
        for person in pose_frame.persons:
            total_joints += len(person.keypoints)
            for point in person.keypoints.values():
                if not math.isfinite(point.x_px) or not math.isfinite(point.y_px):
                    nonfinite_coordinates += 1
                if not point.is_inside_frame(pose_frame.geometry):
                    frame_out += 1
                    out_of_frame += 1
            left_points, left_ok, left_visible = _required_quality(
                person, LEFT_REQUIRED, pose_frame.geometry, settings.min_joint_confidence
            )
            right_points, right_ok, right_visible = _required_quality(
                person, RIGHT_REQUIRED, pose_frame.geometry, settings.min_joint_confidence
            )
            left_required_confidences.extend(
                point.confidence for point in left_points if point is not None
            )
            right_required_confidences.extend(
                point.confidence for point in right_points if point is not None
            )
            left_coverage += sum(a and b for a, b in zip(left_ok, left_visible, strict=True))
            right_coverage += sum(a and b for a, b in zip(right_ok, right_visible, strict=True))
            total_required += 3
        left_ok = left_visible = []
        if len(pose_frame.persons) == 1:
            _, left_ok, left_visible = _required_quality(
                pose_frame.persons[0],
                LEFT_REQUIRED,
                pose_frame.geometry,
                settings.min_joint_confidence,
            )
        status = _frame_status(
            pose_frame,
            left_confidence_ok=left_ok,
            left_visible=left_visible,
            square_pixel_tolerance=settings.square_pixel_tolerance,
        )
        statuses[status] += 1
        for reason in _failure_reasons(
            pose_frame,
            minimum_confidence=settings.min_joint_confidence,
            square_pixel_tolerance=settings.square_pixel_tolerance,
        ):
            failure_counts[reason] += 1
        stage = clock()
        angle = None
        if len(pose_frame.persons) == 1:
            angle = knee_angle_2d(
                pose_frame.persons[0],
                pose_frame.geometry,
                side="left",
                minimum_confidence=settings.min_joint_confidence,
                square_pixel_tolerance=settings.square_pixel_tolerance,
            )
            if angle is not None:
                knee_angles.append(angle)
        biomechanics_seconds += clock() - stage
        mean_joint_confidence = mean(frame_joint_confidences) if frame_joint_confidences else None
        internal = {
            "pose_frame": pose_frame,
            "mean_joint_confidence": mean_joint_confidence,
            "left_knee_angle": angle,
        }
        transition = None
        if len(pose_frame.persons) == 1 and previous_single is not None:
            transition = _temporal_observation(previous_single, internal)
            temporal.append(transition)
        previous_single = (
            internal
            if len(pose_frame.persons) == 1
            and status != "NON_SQUARE_PIXEL_ASPECT_RATIO_UNSUPPORTED"
            else None
        )
        observation = {
            "timestamp_us": sampled_frame.timestamp_us,
            "frame_index": sampled_frame.frame_index,
            "detection_count": len(detections),
            "detector_latency_seconds": detector_latency,
            "pose_latency_seconds": pose_provider.last_backend_seconds,
            "person_confidences": [person.person_confidence for person in pose_frame.persons],
            "bboxes": [person.bbox.to_dict() for person in pose_frame.persons],
            "raw_joint_count": pose_provider.last_raw_joint_count,
            "canonical_joint_counts": [len(person.keypoints) for person in pose_frame.persons],
            "mean_joint_confidence": mean_joint_confidence,
            "required_left_joint_mean_confidence": mean(
                point.confidence
                for point in (pose_frame.persons[0].joint(joint) for joint in LEFT_REQUIRED)
                if point is not None
            )
            if len(pose_frame.persons) == 1
            else None,
            "required_right_joint_mean_confidence": mean(
                point.confidence
                for point in (pose_frame.persons[0].joint(joint) for joint in RIGHT_REQUIRED)
                if point is not None
            )
            if len(pose_frame.persons) == 1
            else None,
            "out_of_frame_joint_count": frame_out,
            "target_analysis_eligible": status == "POSE_FRAME_OK",
            "status": status,
            "left_knee_angle_2d_degrees": angle,
            "normalized_joint_displacement": transition["normalized_joint_displacement"]
            if transition
            else None,
        }
        observations.append(observation)
        frame_latencies.append(clock() - pipeline_started)
        if frame_observer:
            observer_started = clock()
            frame_observer(sampled_frame, pose_frame, observation)
            debug_capture_seconds += clock() - observer_started
    finished = clock()
    sampled = len(observations)
    detected_frames = sum(count > 0 for count in person_counts)
    single_frames = sum(count == 1 for count in person_counts)
    multi_frames = sum(count > 1 for count in person_counts)
    no_person_frames = sum(count == 0 for count in person_counts)
    joint_displacements = [item["normalized_joint_displacement"] for item in temporal]
    bbox_displacements = [item["normalized_bbox_center_displacement"] for item in temporal]
    confidence_deltas = [item["confidence_delta"] for item in temporal]
    angle_deltas = [
        item["left_knee_angle_delta_degrees"]
        for item in temporal
        if "left_knee_angle_delta_degrees" in item
    ]
    return {
        "benchmark_contract_version": "ski-bench-real-pose-v2",
        "input_kind": input_kind,
        "video": {
            **metadata.to_dict(),
            "quality_gate": quality.to_dict(),
            "sampled_frames": sampled,
        },
        "runtime": {
            "device": device,
            "model_load": model_load or {"detector_seconds": None, "pose_seconds": None},
            "warmup_frames": warmup_frames,
            "warmup_included_in_per_frame_timing": False,
        },
        "models": {"detector": detector_model, "pose": pose_model},
        "sampling": {
            "sample_fps": sample_fps,
            "sampled_frame_count": sampled,
            "timestamps_us": [item["timestamp_us"] for item in observations],
            "source_frame_indices": [item["frame_index"] for item in observations],
        },
        "detector": {
            "attempted_frames": sampled,
            "frames_with_detection": detected_frames,
            "detection_coverage": _ratio(detected_frames, sampled),
            "mean_person_count": mean(person_counts) if person_counts else None,
            "single_person_frame_ratio": _ratio(single_frames, sampled),
            "multi_person_frame_ratio": _ratio(multi_frames, sampled),
            "no_person_frame_ratio": _ratio(no_person_frames, sampled),
            "mean_detection_confidence": mean(detection_confidences)
            if detection_confidences
            else None,
        },
        "pose": {
            "attempted_persons": attempted_persons,
            "successful_persons": successful_persons,
            "pose_success_ratio": _ratio(successful_persons, attempted_persons),
            "raw_joint_count_expected": 133,
            "raw_joint_counts_observed": sorted(observed_raw_joint_counts),
            "canonical_joint_count": 17,
            "mean_joint_confidence": mean(joint_confidences) if joint_confidences else None,
            "median_joint_confidence": median(joint_confidences) if joint_confidences else None,
            "minimum_joint_confidence": min(joint_confidences) if joint_confidences else None,
            "required_left_joint_mean_confidence": mean(left_required_confidences)
            if left_required_confidences
            else None,
            "required_right_joint_mean_confidence": mean(right_required_confidences)
            if right_required_confidences
            else None,
            "required_left_joint_coverage": _ratio(left_coverage, total_required),
            "required_right_joint_coverage": _ratio(right_coverage, total_required),
            "out_of_frame_joint_ratio": _ratio(out_of_frame, total_joints),
        },
        "analysis": {
            "frame_status_counts": dict(sorted(statuses.items())),
            "single_person_analyzable_frame_ratio": _ratio(statuses["POSE_FRAME_OK"], sampled),
            "left_knee_angle_coverage": _ratio(len(knee_angles), sampled),
            "multiple_person_unresolved_ratio": _ratio(
                statuses["MULTIPLE_PERSONS_TARGET_UNRESOLVED"], sampled
            ),
            "left_knee_angle_2d_degrees": {
                "count": len(knee_angles),
                "min": min(knee_angles) if knee_angles else None,
                "median": median(knee_angles) if knee_angles else None,
                "max": max(knee_angles) if knee_angles else None,
                "p05": _percentile(knee_angles, 0.05, minimum=20),
                "p95": _percentile(knee_angles, 0.95, minimum=20),
                "time_series": [
                    {
                        "timestamp_us": item["timestamp_us"],
                        "left_knee_angle_2d_degrees": item["left_knee_angle_2d_degrees"],
                    }
                    for item in observations
                    if item["left_knee_angle_2d_degrees"] is not None
                ],
            },
        },
        "temporal_observation": {
            "consecutive_single_person_pairs": len(temporal),
            "normalization": "mean consecutive bbox diagonal",
            "median_normalized_joint_displacement": median(joint_displacements)
            if joint_displacements
            else None,
            "p95_normalized_joint_displacement": _percentile(joint_displacements, 0.95, minimum=20),
            "median_normalized_bbox_center_displacement": median(bbox_displacements)
            if bbox_displacements
            else None,
            "p95_normalized_bbox_center_displacement": _percentile(
                bbox_displacements, 0.95, minimum=20
            ),
            "median_confidence_delta": median(confidence_deltas) if confidence_deltas else None,
            "p95_confidence_delta": _percentile(confidence_deltas, 0.95, minimum=20),
            "median_left_knee_angle_delta_degrees": median(angle_deltas) if angle_deltas else None,
            "p95_left_knee_angle_delta_degrees": _percentile(angle_deltas, 0.95, minimum=20),
        },
        "failure_reasons": {
            reason: failure_counts.get(reason, 0)
            for reason in (
                "NO_PERSON",
                "MULTIPLE_PERSONS",
                "LOW_CONFIDENCE_LEFT_HIP",
                "LOW_CONFIDENCE_LEFT_KNEE",
                "LOW_CONFIDENCE_LEFT_ANKLE",
                "LEFT_HIP_OUT_OF_FRAME",
                "LEFT_KNEE_OUT_OF_FRAME",
                "LEFT_ANKLE_OUT_OF_FRAME",
                "NON_SQUARE_PIXEL_ASPECT_RATIO",
                "POSE_INFERENCE_ERROR",
            )
        },
        "coordinates": {
            "canonical_space": "SourcePixel2D",
            "orientation": "CanonicalUpright",
            "mirrored": False,
            "invalid_coordinate_count": invalid_coordinates,
            "nonfinite_coordinate_count": nonfinite_coordinates,
            "out_of_frame_coordinate_count": out_of_frame,
        },
        "performance": {
            "decode_seconds": decode_seconds,
            "detector_total_seconds": detector_seconds,
            "pose_total_seconds": pose_seconds,
            "canonical_adapter_total_seconds": adapter_seconds,
            "biomechanics_total_seconds": biomechanics_seconds,
            "overlay_seconds": 0.0,
            "debug_capture_seconds": debug_capture_seconds,
            "total_processing_seconds": finished - started - debug_capture_seconds,
            "mean_detector_latency_seconds": mean(detector_latencies)
            if detector_latencies
            else None,
            "p95_detector_latency_seconds": _percentile(
                detector_latencies, 0.95, minimum=MINIMUM_P95_SAMPLES
            ),
            "mean_pose_latency_seconds": mean(pose_latencies) if pose_latencies else None,
            "p95_pose_latency_seconds": _percentile(
                pose_latencies, 0.95, minimum=MINIMUM_P95_SAMPLES
            ),
            "mean_pipeline_latency_seconds": mean(frame_latencies) if frame_latencies else None,
            "p95_pipeline_latency_seconds": _percentile(
                frame_latencies, 0.95, minimum=MINIMUM_P95_SAMPLES
            ),
            "sampled_pipeline_throughput_fps": _ratio(
                sampled, finished - started - debug_capture_seconds
            ),
        },
        "frame_observations": observations,
        "debug_selection_frame_indices": select_debug_frame_indices(observations),
        "warnings": (["MULTIPLE_PERSONS_TARGET_IDENTITY_UNRESOLVED"] if multi_frames else []),
        "limitations": [
            "IMAGE_2D_ONLY_NOT_PHYSICAL_3D",
            "NO_TEMPORAL_SMOOTHING_APPLIED",
            "FRAME_INDEPENDENT_OBSERVATIONS_NOT_TRACKS",
        ],
        "REAL_GT_STATUS": "NOT_AVAILABLE",
        "ground_truth_metrics": {
            "diagnosis_precision": None,
            "diagnosis_recall": None,
            "diagnosis_f1": None,
        },
    }
