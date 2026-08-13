"""Temporal aggregation and timestamp-based derivatives for A5 facts."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

from .contracts import (
    BiomechanicsFact,
    BiomechanicsFactStatus,
    BiomechanicsFeatureConfig,
    FeatureAggregate,
)
from .registry import FRAME_FEATURE_REGISTRY_V1

DERIVATIVE_FEATURES = (
    ("left_knee_angle_2d_deg", "left_knee_angle_abs_velocity_median_deg_per_s", "deg/s"),
    ("right_knee_angle_2d_deg", "right_knee_angle_abs_velocity_median_deg_per_s", "deg/s"),
    (
        "bilateral_knee_mean_angle_2d_deg",
        "bilateral_knee_mean_angle_abs_velocity_median_deg_per_s",
        "deg/s",
    ),
    ("signed_lateral_body_proxy", "signed_lateral_body_proxy_abs_velocity_median_per_s", "1/s"),
)


def aggregate_frame_facts(
    facts: tuple[BiomechanicsFact, ...], config: BiomechanicsFeatureConfig
) -> tuple[FeatureAggregate, ...]:
    config.validate()
    grouped = defaultdict(list)
    for fact in facts:
        if fact.temporal_segment_id is not None:
            grouped[(fact.temporal_segment_id, fact.feature_id)].append(fact)
    results = []
    for segment_id in sorted({key[0] for key in grouped}):
        for definition in FRAME_FEATURE_REGISTRY_V1:
            items = grouped.get((segment_id, definition.feature_id), [])
            available = [item for item in items if item.status is BiomechanicsFactStatus.AVAILABLE]
            values = [float(item.value) for item in available]
            enough = len(values) >= config.minimum_aggregate_samples
            observed = sum(item.interpolated_joint_count == 0 for item in available)
            interpolated = len(available) - observed
            confidences = [
                item.support_confidence for item in available if item.support_confidence is not None
            ]
            results.append(
                FeatureAggregate(
                    definition.feature_id,
                    segment_id,
                    definition.unit,
                    len(items),
                    len(values),
                    len(values) / len(items) if items else 0.0,
                    median(values) if enough else None,
                    min(values) if enough else None,
                    max(values) if enough else None,
                    max(values) - min(values) if enough else None,
                    median(confidences) if confidences else None,
                    observed,
                    interpolated,
                    observed / len(items) if items else 0.0,
                    interpolated / len(items) if items else 0.0,
                    BiomechanicsFactStatus.AVAILABLE
                    if enough
                    else BiomechanicsFactStatus.INSUFFICIENT_SAMPLES,
                )
            )
    return tuple(results)


def derivative_aggregates(
    facts: tuple[BiomechanicsFact, ...], config: BiomechanicsFeatureConfig
) -> tuple[FeatureAggregate, ...]:
    results = []
    segments = sorted(
        {fact.temporal_segment_id for fact in facts if fact.temporal_segment_id is not None}
    )
    previous_by_feature = {}
    by_key = defaultdict(dict)
    for fact in facts:
        key = (fact.temporal_segment_id, fact.feature_id)
        if fact.temporal_segment_id is not None and fact.timestamp_us is not None:
            previous_timestamp = previous_by_feature.get(key)
            if previous_timestamp is not None and fact.timestamp_us <= previous_timestamp:
                raise ValueError("biomechanics derivative timestamps must strictly increase")
            previous_by_feature[key] = fact.timestamp_us
        by_key[(fact.temporal_segment_id, fact.timestamp_us)][fact.feature_id] = fact
    for segment_id in segments:
        timestamps = sorted(timestamp for sid, timestamp in by_key if sid == segment_id)
        for source_id, output_id, unit in DERIVATIVE_FEATURES:
            velocities, previous = [], None
            for timestamp in timestamps:
                fact = by_key[(segment_id, timestamp)].get(source_id)
                if fact is None or fact.status is not BiomechanicsFactStatus.AVAILABLE:
                    previous = None
                    continue
                if previous is not None:
                    dt = timestamp - previous.timestamp_us
                    if dt < config.minimum_derivative_dt_us:
                        raise ValueError(
                            "biomechanics derivative timestamps must strictly increase"
                        )
                    velocities.append(
                        abs(float(fact.value) - float(previous.value)) / (dt / 1_000_000)
                    )
                previous = fact
            enough = bool(velocities)
            results.append(
                FeatureAggregate(
                    output_id,
                    segment_id,
                    unit,
                    max(0, len(timestamps) - 1),
                    len(velocities),
                    len(velocities) / max(1, len(timestamps) - 1),
                    median(velocities) if enough else None,
                    min(velocities) if enough else None,
                    max(velocities) if enough else None,
                    max(velocities) - min(velocities) if enough else None,
                    None,
                    0,
                    0,
                    0.0,
                    0.0,
                    BiomechanicsFactStatus.AVAILABLE
                    if enough
                    else BiomechanicsFactStatus.INSUFFICIENT_SAMPLES,
                )
            )
    return tuple(results)


def feature_coverage(facts: tuple[BiomechanicsFact, ...]) -> dict[str, dict[str, object]]:
    trusted = len({fact.timestamp_us for fact in facts if fact.temporal_segment_id is not None})
    result = {}
    for definition in FRAME_FEATURE_REGISTRY_V1:
        items = [
            fact
            for fact in facts
            if fact.feature_id == definition.feature_id and fact.temporal_segment_id is not None
        ]
        available = sum(fact.status is BiomechanicsFactStatus.AVAILABLE for fact in items)
        result[definition.feature_id] = {
            "total_trusted_frames": trusted,
            "available_frame_count": available,
            "coverage_ratio": available / trusted if trusted else 0.0,
            "status_reason_counts": dict(
                sorted(
                    Counter(
                        fact.status.value
                        for fact in items
                        if fact.status is not BiomechanicsFactStatus.AVAILABLE
                    ).items()
                )
            ),
        }
    return result
