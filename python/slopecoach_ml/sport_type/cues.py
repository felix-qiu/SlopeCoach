"""Extraction of uncalibrated, viewpoint-dependent cues from existing A5 aggregates."""

from __future__ import annotations

from statistics import median

from .contracts import SportCueMeasurement, SportCueStatus

_CUES = (
    ("ankle_separation_body_scale_median", "ankle_separation_body_scale", "body_scale", "median"),
    (
        "shoulder_hip_axis_difference_2d_deg_median",
        "shoulder_hip_axis_difference_2d_deg",
        "deg",
        "median",
    ),
    (
        "signed_lateral_body_proxy_abs_velocity_median_per_s",
        "signed_lateral_body_proxy_abs_velocity_median_per_s",
        "1/s",
        "median",
    ),
    (
        "bilateral_knee_mean_angle_2d_deg_range",
        "bilateral_knee_mean_angle_2d_deg",
        "deg",
        "range",
    ),
)


def extract_uncalibrated_sport_cues(biomechanics_result) -> tuple[SportCueMeasurement, ...]:
    aggregates = (
        biomechanics_result["temporal_segment_features"]
        if isinstance(biomechanics_result, dict)
        else biomechanics_result.temporal_segment_features
    )
    output = []
    for cue_id, feature_id, unit, statistic in _CUES:
        values = [
            (item.get(statistic) if isinstance(item, dict) else getattr(item, statistic))
            for item in aggregates
            if (item.get("feature_id") if isinstance(item, dict) else item.feature_id) == feature_id
            and (item.get(statistic) if isinstance(item, dict) else getattr(item, statistic))
            is not None
        ]
        value = median(values) if values else None
        output.append(
            SportCueMeasurement(
                cue_id=cue_id,
                value=value,
                unit=unit,
                status=(
                    SportCueStatus.AVAILABLE if value is not None else SportCueStatus.NOT_AVAILABLE
                ),
                source_feature_ids=(feature_id,),
                limitations=("UNCALIBRATED", "VIEWPOINT_DEPENDENT", "NOT_CLASSIFICATION_EVIDENCE"),
                contributes_to_auto_fusion=False,
            )
        )
    return tuple(output)
