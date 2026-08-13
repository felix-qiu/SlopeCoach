"""Identity-safe temporal segmentation, interpolation, stabilization, and metrics."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from statistics import median

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import Joint

from .contracts import (
    StabilizedPoseSample,
    TargetPoseSample,
    TemporalJoint2D,
    TemporalPoseConfig,
    TemporalPoseQuality,
    TemporalPoseRun,
    TemporalProvenance,
)
from .interpolation import interpolate_segment
from .one_euro import OneEuroFilter1D


def _geometry_key(sample: TargetPoseSample) -> tuple[object, ...]:
    geometry = sample.geometry
    return (
        geometry.width_px,
        geometry.height_px,
        geometry.pixel_aspect_ratio,
        geometry.coordinate_space,
        geometry.orientation,
        geometry.mirrored,
    )


def _trusted(sample: TargetPoseSample, config: TemporalPoseConfig) -> bool:
    return (
        sample.identity_state is TargetIdentityState.LOCKED
        and sample.identity_confidence >= config.minimum_identity_confidence
        and sample.active_track_id is not None
        and not sample.explicit_discontinuity
    )


def stabilize_target_pose_stream(
    samples: list[TargetPoseSample], config: TemporalPoseConfig | None = None
) -> TemporalPoseRun:
    settings = config or TemporalPoseConfig()
    settings.validate()
    if not samples:
        return TemporalPoseRun([], 0, 0, 0, 0)
    previous_timestamp = None
    segment_ids: list[int | None] = []
    segment_id = 0
    prior_trusted: TargetPoseSample | None = None
    for sample in samples:
        sample.validate()
        if previous_timestamp is not None and sample.timestamp_us <= previous_timestamp:
            raise ValueError("target pose stream timestamps must strictly increase")
        previous_timestamp = sample.timestamp_us
        if not _trusted(sample, settings):
            segment_ids.append(None)
            prior_trusted = None
            continue
        new_segment = prior_trusted is None
        if prior_trusted is not None:
            new_segment = (
                sample.timestamp_us - prior_trusted.timestamp_us > settings.hard_reset_gap_us
                or sample.active_track_id != prior_trusted.active_track_id
                or _geometry_key(sample) != _geometry_key(prior_trusted)
                or sample.target_id != prior_trusted.target_id
            )
        if new_segment:
            segment_id += 1
        segment_ids.append(segment_id)
        prior_trusted = sample

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, current_id in enumerate(segment_ids):
        if current_id is not None:
            grouped[current_id].append(index)
    results: list[StabilizedPoseSample | None] = [None] * len(samples)
    interpolation_seconds = stabilization_seconds = 0.0
    total_interpolated = total_long_unfilled = 0
    joint_filter_resets = 0
    for current_id, indices in grouped.items():
        segment = [samples[index] for index in indices]
        started = time.perf_counter()
        supports, filled, long_unfilled = interpolate_segment(segment, settings)
        interpolation_seconds += time.perf_counter() - started
        total_interpolated += filled
        total_long_unfilled += long_unfilled
        filters = {
            joint: (
                OneEuroFilter1D(
                    min_cutoff_hz=settings.one_euro_min_cutoff_hz,
                    beta=settings.one_euro_beta,
                    derivative_cutoff_hz=settings.one_euro_derivative_cutoff_hz,
                ),
                OneEuroFilter1D(
                    min_cutoff_hz=settings.one_euro_min_cutoff_hz,
                    beta=settings.one_euro_beta,
                    derivative_cutoff_hz=settings.one_euro_derivative_cutoff_hz,
                ),
            )
            for joint in Joint
        }
        last_joint_timestamp: dict[Joint, int] = {}
        started = time.perf_counter()
        for local_index, (source, support) in enumerate(zip(segment, supports, strict=True)):
            joints = {}
            for joint in Joint:
                point = support[joint]
                sx = sy = None
                if point.x_px is not None:
                    previous_joint_timestamp = last_joint_timestamp.get(joint)
                    if (
                        previous_joint_timestamp is not None
                        and source.timestamp_us - previous_joint_timestamp
                        > settings.hard_reset_gap_us
                    ):
                        filters[joint][0].reset()
                        filters[joint][1].reset()
                        joint_filter_resets += 1
                    sx = filters[joint][0].filter(point.x_px, source.timestamp_us)
                    sy = filters[joint][1].filter(point.y_px, source.timestamp_us)
                    last_joint_timestamp[joint] = source.timestamp_us
                raw = point.raw
                joints[joint] = TemporalJoint2D(
                    joint,
                    raw.x_px if raw else None,
                    raw.y_px if raw else None,
                    point.x_px,
                    point.y_px,
                    sx,
                    sy,
                    point.confidence,
                    point.provenance,
                    sx is not None,
                )
            observed = sum(p.provenance is TemporalProvenance.OBSERVED for p in joints.values())
            interpolated = sum(
                p.provenance is TemporalProvenance.INTERPOLATED for p in joints.values()
            )
            missing = len(Joint) - observed - interpolated
            quality = (
                TemporalPoseQuality.GOOD
                if missing == 0
                else TemporalPoseQuality.PARTIAL
                if observed + interpolated >= 8
                else TemporalPoseQuality.INSUFFICIENT
            )
            results[indices[local_index]] = StabilizedPoseSample(
                source.timestamp_us,
                source.frame_index,
                current_id,
                source.geometry,
                source.target_id,
                source.active_track_id,
                source.identity_state,
                joints,
                observed,
                interpolated,
                missing,
                quality,
                (),
                ("IMAGE_SPACE_2D_ONLY", "PYTHON_RESEARCH_REFERENCE_ONLY"),
            )
        stabilization_seconds += time.perf_counter() - started

    for index, source in enumerate(samples):
        if results[index] is not None:
            continue
        joints = {
            joint: TemporalJoint2D(
                joint,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                TemporalProvenance.MISSING,
                False,
            )
            for joint in Joint
        }
        results[index] = StabilizedPoseSample(
            source.timestamp_us,
            source.frame_index,
            None,
            source.geometry,
            source.target_id,
            source.active_track_id,
            source.identity_state,
            joints,
            0,
            0,
            len(Joint),
            TemporalPoseQuality.INSUFFICIENT,
            ("IDENTITY_UNSAFE_TEMPORAL_BOUNDARY",),
            ("IMAGE_SPACE_2D_ONLY", "PYTHON_RESEARCH_REFERENCE_ONLY"),
        )
    completed = [item for item in results if item is not None]
    return TemporalPoseRun(
        completed,
        len(grouped) + joint_filter_resets,
        len(grouped),
        total_interpolated,
        total_long_unfilled,
        interpolation_seconds,
        stabilization_seconds,
        temporal_stability_metrics(completed),
    )


def temporal_stability_metrics(
    samples: list[StabilizedPoseSample],
) -> dict[str, object]:
    segment_ids = {
        sample.temporal_segment_id for sample in samples if sample.temporal_segment_id is not None
    }
    segment_scales = segment_body_scales(samples)
    raw_steps, stable_steps, raw_second, stable_second = [], [], [], []
    histories: dict[tuple[int, Joint], list[tuple[float, float, float, float]]] = defaultdict(list)
    for sample in samples:
        if sample.temporal_segment_id is None:
            continue
        body_scale = segment_scales.get(sample.temporal_segment_id)
        if body_scale is None:
            continue
        for joint, point in sample.joints.items():
            if None in (
                point.raw_x_px,
                point.raw_y_px,
                point.stabilized_x_px,
                point.stabilized_y_px,
            ):
                histories.pop((sample.temporal_segment_id, joint), None)
                continue
            values = (
                point.raw_x_px / body_scale,
                point.raw_y_px / body_scale,
                point.stabilized_x_px / body_scale,
                point.stabilized_y_px / body_scale,
            )
            history = histories[(sample.temporal_segment_id, joint)]
            history.append(values)
            if len(history) >= 2:
                previous = history[-2]
                raw_steps.append(math.hypot(values[0] - previous[0], values[1] - previous[1]))
                stable_steps.append(math.hypot(values[2] - previous[2], values[3] - previous[3]))
            if len(history) >= 3:
                a, b = history[-3], history[-2]
                raw_second.append(
                    math.hypot(values[0] - 2 * b[0] + a[0], values[1] - 2 * b[1] + a[1])
                )
                stable_second.append(
                    math.hypot(values[2] - 2 * b[2] + a[2], values[3] - 2 * b[3] + a[3])
                )
    return {
        "normalization_strategy": ("SEGMENT_MEDIAN_RAW_SHOULDER_CENTER_TO_ANKLE_CENTER"),
        "body_scale_valid_segment_count": len(segment_scales),
        "body_scale_unavailable_segment_count": len(segment_ids) - len(segment_scales),
        "raw_joint_step_median": median(raw_steps) if raw_steps else None,
        "stabilized_joint_step_median": median(stable_steps) if stable_steps else None,
        "raw_second_difference_median": median(raw_second) if raw_second else None,
        "stabilized_second_difference_median": median(stable_second) if stable_second else None,
        "joint_step_comparison_count": len(raw_steps),
        "second_difference_comparison_count": len(raw_second),
    }


def symmetric_frame_body_scale(sample: StabilizedPoseSample) -> float | None:
    points = [
        sample.joint(Joint.LEFT_SHOULDER),
        sample.joint(Joint.RIGHT_SHOULDER),
        sample.joint(Joint.LEFT_ANKLE),
        sample.joint(Joint.RIGHT_ANKLE),
    ]
    if any(
        point is None or point.provenance is not TemporalProvenance.OBSERVED for point in points
    ):
        return None
    coordinates = [(point.raw_x_px, point.raw_y_px) for point in points]
    if any(
        x is None or y is None or not math.isfinite(x) or not math.isfinite(y)
        for x, y in coordinates
    ):
        return None
    left_shoulder, right_shoulder, left_ankle, right_ankle = coordinates
    shoulder_center = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    )
    ankle_center = (
        (left_ankle[0] + right_ankle[0]) / 2,
        (left_ankle[1] + right_ankle[1]) / 2,
    )
    scale = math.hypot(ankle_center[0] - shoulder_center[0], ankle_center[1] - shoulder_center[1])
    return scale if math.isfinite(scale) and scale > 1e-9 else None


def segment_body_scales(samples: list[StabilizedPoseSample]) -> dict[int, float]:
    """Return one constant median raw symmetric scale per temporal segment."""
    scale_candidates: dict[int, list[float]] = defaultdict(list)
    for sample in samples:
        if sample.temporal_segment_id is None:
            continue
        scale = symmetric_frame_body_scale(sample)
        if scale is not None:
            scale_candidates[sample.temporal_segment_id].append(scale)
    return {
        segment_id: median(candidates)
        for segment_id, candidates in scale_candidates.items()
        if candidates
    }
