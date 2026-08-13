"""Frame-level facts from trusted stabilized canonical pose samples."""

from __future__ import annotations

import math

from slopecoach_ml.pose import Joint
from slopecoach_ml.temporal import StabilizedPoseSample, TemporalProvenance
from slopecoach_ml.turns import signed_lateral_body_proxy

from .contracts import (
    BiomechanicsFact,
    BiomechanicsFactStatus,
    BiomechanicsFeatureConfig,
)
from .geometry import (
    angle_three_points_2d,
    distance_2d,
    midpoint_2d,
    normalized_screen_x_offset,
    signed_screen_angle_from_vertical,
    undirected_axis_difference_deg,
)
from .registry import FRAME_FEATURE_REGISTRY_V1


def _joint_evidence(sample, required, config):
    if sample.temporal_segment_id is None:
        return None, BiomechanicsFactStatus.INSUFFICIENT_EVIDENCE, 0, 0, None
    points, observed, interpolated, confidences = {}, 0, 0, []
    for joint in required:
        point = sample.joint(joint)
        if point is None or point.stabilized_x_px is None or point.stabilized_y_px is None:
            return None, BiomechanicsFactStatus.REQUIRED_JOINT_MISSING, observed, interpolated, None
        if not math.isfinite(point.stabilized_x_px) or not math.isfinite(point.stabilized_y_px):
            return None, BiomechanicsFactStatus.REQUIRED_JOINT_MISSING, observed, interpolated, None
        if (
            point.support_confidence is None
            or point.support_confidence < config.minimum_joint_support_confidence
        ):
            return (
                None,
                BiomechanicsFactStatus.LOW_CONFIDENCE,
                observed,
                interpolated,
                point.support_confidence,
            )
        if not (
            0 <= point.stabilized_x_px < sample.geometry.width_px
            and 0 <= point.stabilized_y_px < sample.geometry.height_px
        ):
            return (
                None,
                BiomechanicsFactStatus.REQUIRED_JOINT_OUT_OF_FRAME,
                observed,
                interpolated,
                point.support_confidence,
            )
        points[joint] = (point.stabilized_x_px, point.stabilized_y_px)
        confidences.append(point.support_confidence)
        observed += point.provenance is TemporalProvenance.OBSERVED
        interpolated += point.provenance is TemporalProvenance.INTERPOLATED
    return points, BiomechanicsFactStatus.AVAILABLE, observed, interpolated, min(confidences)


def compute_frame_biomechanics(
    sample: StabilizedPoseSample,
    segment_body_scale: float | None,
    config: BiomechanicsFeatureConfig | None = None,
) -> tuple[BiomechanicsFact, ...]:
    settings = config or BiomechanicsFeatureConfig()
    settings.validate()
    facts = []
    computed: dict[str, float | None] = {}
    sample.geometry.validate()
    for definition in FRAME_FEATURE_REGISTRY_V1:
        points, status, observed, interpolated, confidence = _joint_evidence(
            sample, definition.required_joints, settings
        )
        value = None
        if (
            status is BiomechanicsFactStatus.AVAILABLE
            and abs(sample.geometry.pixel_aspect_ratio - 1.0) > settings.square_pixel_tolerance
        ):
            status = BiomechanicsFactStatus.UNSUPPORTED_PIXEL_ASPECT_RATIO
        if status is BiomechanicsFactStatus.AVAILABLE:
            fid = definition.feature_id
            if fid == "left_knee_angle_2d_deg":
                value = angle_three_points_2d(
                    points[Joint.LEFT_HIP], points[Joint.LEFT_KNEE], points[Joint.LEFT_ANKLE]
                )
            elif fid == "right_knee_angle_2d_deg":
                value = angle_three_points_2d(
                    points[Joint.RIGHT_HIP], points[Joint.RIGHT_KNEE], points[Joint.RIGHT_ANKLE]
                )
            elif fid in (
                "bilateral_knee_mean_angle_2d_deg",
                "bilateral_knee_abs_difference_2d_deg",
            ):
                left = angle_three_points_2d(
                    points[Joint.LEFT_HIP], points[Joint.LEFT_KNEE], points[Joint.LEFT_ANKLE]
                )
                right = angle_three_points_2d(
                    points[Joint.RIGHT_HIP], points[Joint.RIGHT_KNEE], points[Joint.RIGHT_ANKLE]
                )
                value = (
                    (left + right) / 2
                    if fid.endswith("mean_angle_2d_deg") and left is not None and right is not None
                    else abs(left - right)
                    if left is not None and right is not None
                    else None
                )
            elif fid.endswith("_separation_body_scale"):
                pair = {
                    "ankle_separation_body_scale": (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE),
                    "knee_separation_body_scale": (Joint.LEFT_KNEE, Joint.RIGHT_KNEE),
                    "hip_separation_body_scale": (Joint.LEFT_HIP, Joint.RIGHT_HIP),
                    "shoulder_separation_body_scale": (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
                }[fid]
                value = (
                    distance_2d(points[pair[0]], points[pair[1]]) / segment_body_scale
                    if segment_body_scale
                    else None
                )
            elif fid == "ankle_to_shoulder_separation_ratio_2d":
                denominator = distance_2d(points[Joint.LEFT_SHOULDER], points[Joint.RIGHT_SHOULDER])
                value = (
                    distance_2d(points[Joint.LEFT_ANKLE], points[Joint.RIGHT_ANKLE]) / denominator
                    if denominator > 1e-12
                    else None
                )
            elif fid == "shoulder_to_ankle_screen_lateral_offset_body_scale":
                value = (
                    normalized_screen_x_offset(
                        midpoint_2d(points[Joint.LEFT_SHOULDER], points[Joint.RIGHT_SHOULDER]),
                        midpoint_2d(points[Joint.LEFT_ANKLE], points[Joint.RIGHT_ANKLE]),
                        segment_body_scale,
                    )
                    if segment_body_scale
                    else None
                )
            elif fid == "torso_screen_inclination_deg":
                value = signed_screen_angle_from_vertical(
                    midpoint_2d(points[Joint.LEFT_SHOULDER], points[Joint.RIGHT_SHOULDER]),
                    midpoint_2d(points[Joint.LEFT_HIP], points[Joint.RIGHT_HIP]),
                )
            elif fid == "hip_to_ankle_screen_lateral_offset_body_scale":
                value = (
                    normalized_screen_x_offset(
                        midpoint_2d(points[Joint.LEFT_HIP], points[Joint.RIGHT_HIP]),
                        midpoint_2d(points[Joint.LEFT_ANKLE], points[Joint.RIGHT_ANKLE]),
                        segment_body_scale,
                    )
                    if segment_body_scale
                    else None
                )
            elif fid == "signed_lateral_body_proxy":
                value = signed_lateral_body_proxy(
                    sample, minimum_confidence=settings.minimum_joint_support_confidence
                ).value
            elif fid == "shoulder_hip_axis_difference_2d_deg":
                value = undirected_axis_difference_deg(
                    points[Joint.LEFT_SHOULDER],
                    points[Joint.RIGHT_SHOULDER],
                    points[Joint.LEFT_HIP],
                    points[Joint.RIGHT_HIP],
                )
            if value is None:
                status = (
                    BiomechanicsFactStatus.DEGENERATE_GEOMETRY
                    if segment_body_scale is not None or "body_scale" not in fid
                    else BiomechanicsFactStatus.INSUFFICIENT_EVIDENCE
                )
        computed[definition.feature_id] = value
        facts.append(
            BiomechanicsFact(
                definition.feature_id,
                definition.family,
                definition.scope,
                definition.unit,
                value,
                status,
                sample.timestamp_us,
                sample.temporal_segment_id,
                support_confidence=confidence,
                required_joints=definition.required_joints,
                observed_joint_count=observed,
                interpolated_joint_count=interpolated,
                limitations=definition.limitations,
            )
        )
    return tuple(facts)
