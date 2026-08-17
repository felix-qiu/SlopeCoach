"""Centralized exact-turn-window biomechanics evidence extraction."""

from __future__ import annotations

import math


def collect_turn_feature_evidence(
    *, turn: dict[str, object], frame_facts: list[dict[str, object]], feature_id: str
) -> dict[str, object]:
    start = turn.get("start_timestamp_us")
    end = turn.get("end_timestamp_us")
    segment = turn.get("temporal_segment_id")
    run = turn.get("signal_run_id")
    eligible_timestamps = sorted(
        {
            fact["timestamp_us"]
            for fact in frame_facts
            if fact.get("timestamp_us") is not None
            and start <= fact["timestamp_us"] <= end
            and fact.get("temporal_segment_id") == segment
            and fact.get("signal_run_id") in {None, run}
        }
    )
    selected = [
        fact
        for fact in frame_facts
        if fact.get("feature_id") == feature_id
        and fact.get("status") == "AVAILABLE"
        and fact.get("timestamp_us") in eligible_timestamps
        and fact.get("temporal_segment_id") == segment
        and fact.get("signal_run_id") in {None, run}
        and _finite(fact.get("value"))
    ]
    selected.sort(key=lambda item: item["timestamp_us"])
    timestamps = [item["timestamp_us"] for item in selected]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError(f"duplicate frame feature fact for {feature_id}")
    observed = sum((item.get("interpolated_joint_count") or 0) == 0 for item in selected)
    interpolated = len(selected) - observed
    confidences = [
        item["support_confidence"]
        for item in selected
        if item.get("support_confidence") is not None
    ]
    return {
        "facts": selected,
        "eligible_turn_pose_timestamps": eligible_timestamps,
        "sample_count": len(selected),
        "eligible_sample_count": len(eligible_timestamps),
        "coverage": len(selected) / len(eligible_timestamps) if eligible_timestamps else 0.0,
        "evidence_timestamps_us": timestamps,
        "minimum_support_confidence": min(confidences) if confidences else None,
        "observed_sample_count": observed,
        "interpolated_sample_count": interpolated,
    }


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
