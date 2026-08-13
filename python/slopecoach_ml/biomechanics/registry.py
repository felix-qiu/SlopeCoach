"""Deterministically ordered A5 feature registry; no ML vector is frozen."""

from __future__ import annotations

from dataclasses import dataclass

from slopecoach_ml.pose import Joint

from .contracts import BiomechanicsFactScope, BiomechanicsFeatureFamily

LEFT_KNEE = (Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE)
RIGHT_KNEE = (Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE)
SHOULDERS = (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)
HIPS = (Joint.LEFT_HIP, Joint.RIGHT_HIP)
KNEES = (Joint.LEFT_KNEE, Joint.RIGHT_KNEE)
ANKLES = (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)
TURN_JOINTS = SHOULDERS + HIPS + KNEES + ANKLES


@dataclass(frozen=True)
class FeatureDefinition:
    feature_id: str
    family: BiomechanicsFeatureFamily
    scope: BiomechanicsFactScope
    unit: str
    description: str
    required_joints: tuple[Joint, ...]
    limitations: tuple[str, ...]


def _frame(feature_id, family, unit, description, joints, *limitations):
    return FeatureDefinition(
        feature_id,
        family,
        BiomechanicsFactScope.FRAME,
        unit,
        description,
        joints,
        tuple(limitations),
    )


FRAME_FEATURE_REGISTRY_V1 = (
    _frame(
        "left_knee_angle_2d_deg",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        "deg",
        "Left hip-knee-ankle image angle.",
        LEFT_KNEE,
        "IMAGE_2D_ONLY_NOT_PHYSICAL_3D",
    ),
    _frame(
        "right_knee_angle_2d_deg",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        "deg",
        "Right hip-knee-ankle image angle.",
        RIGHT_KNEE,
        "IMAGE_2D_ONLY_NOT_PHYSICAL_3D",
    ),
    _frame(
        "bilateral_knee_mean_angle_2d_deg",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        "deg",
        "Mean of available bilateral knee image angles.",
        LEFT_KNEE + RIGHT_KNEE,
        "IMAGE_2D_ONLY_NOT_PHYSICAL_3D",
    ),
    _frame(
        "bilateral_knee_abs_difference_2d_deg",
        BiomechanicsFeatureFamily.SYMMETRY_PROXY,
        "deg",
        "Absolute bilateral knee image-angle difference.",
        LEFT_KNEE + RIGHT_KNEE,
        "VIEWPOINT_DEPENDENT",
    ),
    _frame(
        "ankle_separation_body_scale",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        "ratio",
        "Ankle distance divided by segment body scale.",
        ANKLES,
        "CAMERA_VIEW_DEPENDENT",
    ),
    _frame(
        "knee_separation_body_scale",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        "ratio",
        "Knee distance divided by segment body scale.",
        KNEES,
        "CAMERA_VIEW_DEPENDENT",
    ),
    _frame(
        "hip_separation_body_scale",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        "ratio",
        "Hip distance divided by segment body scale.",
        HIPS,
        "CAMERA_VIEW_DEPENDENT",
    ),
    _frame(
        "shoulder_separation_body_scale",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        "ratio",
        "Shoulder distance divided by segment body scale.",
        SHOULDERS,
        "CAMERA_VIEW_DEPENDENT",
    ),
    _frame(
        "ankle_to_shoulder_separation_ratio_2d",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        "ratio",
        "Ankle separation divided by shoulder separation.",
        ANKLES + SHOULDERS,
        "CAMERA_VIEW_DEPENDENT",
    ),
    _frame(
        "shoulder_to_ankle_screen_lateral_offset_body_scale",
        BiomechanicsFeatureFamily.BALANCE_PROXY,
        "ratio",
        "Screen-x shoulder-center minus ankle-center, normalized.",
        SHOULDERS + ANKLES,
        "CAMERA_VIEW_DEPENDENT",
        "NOT_PHYSICAL_CENTER_OF_MASS",
    ),
    _frame(
        "torso_screen_inclination_deg",
        BiomechanicsFeatureFamily.BALANCE_PROXY,
        "deg",
        "Shoulder-center to hip-center signed angle from image vertical.",
        SHOULDERS + HIPS,
        "IMAGE_SCREEN_ORIENTATION_ONLY",
        "CAMERA_VIEW_DEPENDENT",
    ),
    _frame(
        "hip_to_ankle_screen_lateral_offset_body_scale",
        BiomechanicsFeatureFamily.EDGE_CONTROL_PROXY,
        "ratio",
        "Screen-x hip-center minus ankle-center, normalized.",
        HIPS + ANKLES,
        "IMAGE_SPACE_2D_PROXY_ONLY",
        "NO_PHYSICAL_EDGE_ANGLE",
    ),
    _frame(
        "signed_lateral_body_proxy",
        BiomechanicsFeatureFamily.EDGE_CONTROL_PROXY,
        "ratio",
        "Existing A4 signed lower-body image-space proxy.",
        TURN_JOINTS,
        "IMAGE_SPACE_2D_PROXY_ONLY",
        "NO_PHYSICAL_EDGE_ANGLE",
    ),
    _frame(
        "shoulder_hip_axis_difference_2d_deg",
        BiomechanicsFeatureFamily.SYMMETRY_PROXY,
        "deg",
        "Smallest undirected shoulder-versus-hip screen-axis difference.",
        SHOULDERS + HIPS,
        "VIEWPOINT_DEPENDENT",
    ),
)


def _nonframe(feature_id, family, scope, unit, description, *limitations):
    return FeatureDefinition(
        feature_id,
        family,
        scope,
        unit,
        description,
        (),
        tuple(limitations),
    )


TEMPORAL_FEATURE_REGISTRY_V1 = (
    _nonframe(
        "left_knee_angle_abs_velocity_median_deg_per_s",
        BiomechanicsFeatureFamily.TIMING_PROXY,
        BiomechanicsFactScope.TEMPORAL_SEGMENT,
        "deg/s",
        "Median absolute timestamp-derived left-knee angle velocity.",
        "IMAGE_2D_ONLY_NOT_PHYSICAL_3D",
    ),
    _nonframe(
        "right_knee_angle_abs_velocity_median_deg_per_s",
        BiomechanicsFeatureFamily.TIMING_PROXY,
        BiomechanicsFactScope.TEMPORAL_SEGMENT,
        "deg/s",
        "Median absolute timestamp-derived right-knee angle velocity.",
        "IMAGE_2D_ONLY_NOT_PHYSICAL_3D",
    ),
    _nonframe(
        "bilateral_knee_mean_angle_abs_velocity_median_deg_per_s",
        BiomechanicsFeatureFamily.TIMING_PROXY,
        BiomechanicsFactScope.TEMPORAL_SEGMENT,
        "deg/s",
        "Median absolute timestamp-derived bilateral-mean knee velocity.",
        "IMAGE_2D_ONLY_NOT_PHYSICAL_3D",
    ),
    _nonframe(
        "signed_lateral_body_proxy_abs_velocity_median_per_s",
        BiomechanicsFeatureFamily.TIMING_PROXY,
        BiomechanicsFactScope.TEMPORAL_SEGMENT,
        "1/s",
        "Median absolute timestamp-derived lateral-proxy velocity.",
        "IMAGE_SPACE_2D_PROXY_ONLY",
    ),
)

TURN_FEATURE_REGISTRY_V1 = (
    _nonframe(
        "turn_duration_us",
        BiomechanicsFeatureFamily.TIMING_PROXY,
        BiomechanicsFactScope.TURN,
        "us",
        "A4.1 turn boundary duration.",
    ),
    _nonframe(
        "turn_peak_lateral_proxy",
        BiomechanicsFeatureFamily.EDGE_CONTROL_PROXY,
        BiomechanicsFactScope.TURN,
        "ratio",
        "A4.1 image-space peak proxy.",
        "NO_PHYSICAL_EDGE_ANGLE",
    ),
    _nonframe(
        "bilateral_knee_mean_angle_at_apex_deg",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        BiomechanicsFactScope.TURN,
        "deg",
        "Bilateral mean knee image angle nearest turn apex.",
        "IMAGE_2D_ONLY_NOT_PHYSICAL_3D",
    ),
    _nonframe(
        "bilateral_knee_abs_difference_at_apex_deg",
        BiomechanicsFeatureFamily.SYMMETRY_PROXY,
        BiomechanicsFactScope.TURN,
        "deg",
        "Bilateral knee image-angle difference nearest apex.",
        "VIEWPOINT_DEPENDENT",
    ),
    _nonframe(
        "ankle_separation_at_apex_body_scale",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        BiomechanicsFactScope.TURN,
        "ratio",
        "Normalized ankle separation nearest apex.",
        "CAMERA_VIEW_DEPENDENT",
    ),
    _nonframe(
        "bilateral_knee_mean_angle_at_start_deg",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        BiomechanicsFactScope.TURN,
        "deg",
        "Bilateral mean knee image angle nearest start.",
        "IMAGE_2D_ONLY_NOT_PHYSICAL_3D",
    ),
    _nonframe(
        "bilateral_knee_mean_angle_at_end_deg",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        BiomechanicsFactScope.TURN,
        "deg",
        "Bilateral mean knee image angle nearest end.",
        "IMAGE_2D_ONLY_NOT_PHYSICAL_3D",
    ),
    _nonframe(
        "knee_angle_change_start_to_apex_deg",
        BiomechanicsFeatureFamily.TIMING_PROXY,
        BiomechanicsFactScope.TURN,
        "deg",
        "Signed apex minus start bilateral mean angle.",
    ),
    _nonframe(
        "knee_angle_change_apex_to_end_deg",
        BiomechanicsFeatureFamily.TIMING_PROXY,
        BiomechanicsFactScope.TURN,
        "deg",
        "Signed end minus apex bilateral mean angle.",
    ),
    _nonframe(
        "minimum_mean_knee_angle_timestamp_us",
        BiomechanicsFeatureFamily.TIMING_PROXY,
        BiomechanicsFactScope.TURN,
        "us",
        "Timestamp of minimum bilateral mean image angle in complete turn.",
    ),
    _nonframe(
        "minimum_mean_knee_angle_offset_from_apex_us",
        BiomechanicsFeatureFamily.TIMING_PROXY,
        BiomechanicsFactScope.TURN,
        "us",
        "Minimum-angle timestamp minus apex timestamp.",
    ),
    _nonframe(
        "minimum_mean_knee_angle_phase_offset",
        BiomechanicsFeatureFamily.TIMING_PROXY,
        BiomechanicsFactScope.TURN,
        "ratio",
        "Minimum-angle apex offset divided by complete turn duration.",
    ),
)

FEATURE_REGISTRY_V1 = (
    FRAME_FEATURE_REGISTRY_V1 + TEMPORAL_FEATURE_REGISTRY_V1 + TURN_FEATURE_REGISTRY_V1
)

FEATURE_BY_ID = {definition.feature_id: definition for definition in FEATURE_REGISTRY_V1}
