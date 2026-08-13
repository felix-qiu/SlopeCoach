"""Synthetic known-answer temporal and turn Golden runners."""

from __future__ import annotations

import json
from pathlib import Path

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import BoundingBox2D, FrameGeometry, Joint, Keypoint2D, PersonPose2D
from slopecoach_ml.turns import (
    ReferencePeakDetector,
    TurnSegmentationConfig,
    TurnSignalSample,
    detect_zero_crossings,
    segment_turns,
)

from .contracts import TargetPoseSample, TemporalPoseConfig, TemporalProvenance
from .stabilizer import stabilize_target_pose_stream


def run_temporal_golden(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    geometry = FrameGeometry.from_dict(data["frame_geometry"])
    samples = []
    truth = {}
    for item in data["samples"]:
        raw = item.get("joint")
        pose = PersonPose2D(
            BoundingBox2D(40, 20, 120, 260),
            0.95,
            {Joint.LEFT_ANKLE: Keypoint2D(raw["x_px"], raw["y_px"], raw["confidence"])}
            if raw
            else {},
            1,
        )
        samples.append(
            TargetPoseSample(
                item["timestamp_us"],
                item["frame_index"],
                "golden-target",
                item.get("active_track_id"),
                TargetIdentityState(item["identity_state"]),
                item["identity_confidence"],
                geometry,
                pose,
            )
        )
        if item.get("truth_x_px") is not None:
            truth[item["timestamp_us"]] = item["truth_x_px"]
    config = TemporalPoseConfig(**data["config"])
    run = stabilize_target_pose_stream(samples, config)
    raw_errors, stable_errors = [], []
    for sample in run.samples:
        point = sample.joint(Joint.LEFT_ANKLE)
        expected = truth.get(sample.timestamp_us)
        if expected is None or point.raw_x_px is None or point.stabilized_x_px is None:
            continue
        raw_errors.append(abs(point.raw_x_px - expected))
        stable_errors.append(abs(point.stabilized_x_px - expected))
    expected = data["expected"]
    short = sum(
        point.provenance is TemporalProvenance.INTERPOLATED
        for sample in run.samples
        if (point := sample.joint(Joint.LEFT_ANKLE)) is not None
    )
    long_missing = sum(
        sample.joint(Joint.LEFT_ANKLE).provenance is TemporalProvenance.MISSING
        for sample in run.samples
        if sample.timestamp_us in expected["long_missing_timestamps_us"]
    )
    passed = (
        run.temporal_segment_count == expected["temporal_segment_count"]
        and short == expected["short_interpolated_count"]
        and long_missing == len(expected["long_missing_timestamps_us"])
        and sum(stable_errors) / len(stable_errors) < sum(raw_errors) / len(raw_errors)
    )
    return {
        "golden_passed": passed,
        "fixture_contract_version": data["contract_version"],
        "temporal_segment_count": run.temporal_segment_count,
        "short_interpolated_count": short,
        "long_missing_count": long_missing,
        "raw_mean_absolute_error_px": sum(raw_errors) / len(raw_errors),
        "stabilized_mean_absolute_error_px": sum(stable_errors) / len(stable_errors),
        "run": run.to_dict(),
    }


def run_turn_golden(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = [TurnSignalSample(**item) for item in data["samples"]]
    config = TurnSegmentationConfig(**data["config"])
    peaks = ReferencePeakDetector().detect(samples, config)
    crossings = detect_zero_crossings(
        samples,
        config.zero_crossing_tolerance,
        minimum_signal_confidence=config.minimum_signal_confidence,
    )
    segments = segment_turns(samples, peaks, crossings, config)
    expected = data["expected"]
    tolerance = expected["apex_timestamp_tolerance_us"]
    passed = (
        len(peaks) == len(expected["apex_timestamps_us"])
        and all(
            abs(actual.timestamp_us - wanted) <= tolerance
            for actual, wanted in zip(peaks, expected["apex_timestamps_us"], strict=True)
        )
        and len(crossings) == expected["zero_crossing_count"]
        and [[segment.start_timestamp_us, segment.end_timestamp_us] for segment in segments]
        == expected["segment_boundaries_us"]
        and all(
            segment.temporal_segment_id == peaks[index].temporal_segment_id
            for index, segment in enumerate(segments)
        )
    )
    return {
        "golden_passed": passed,
        "fixture_contract_version": data["contract_version"],
        "accepted_apex_count": len(peaks),
        "apex_timestamps_us": [peak.timestamp_us for peak in peaks],
        "zero_crossing_count": len(crossings),
        "segments": [segment.to_dict() for segment in segments],
    }
