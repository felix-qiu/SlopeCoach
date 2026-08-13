from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import FrameGeometry, Joint
from slopecoach_ml.temporal import (
    StabilizedPoseSample,
    TemporalJoint2D,
    TemporalPoseQuality,
    TemporalProvenance,
    segment_body_scales,
    symmetric_frame_body_scale,
    temporal_stability_metrics,
)
from slopecoach_ml.turns import (
    ReferencePeakDetector,
    TurnSegmentationConfig,
    TurnSignalSample,
    classify_real_turn_status,
    detect_zero_crossings,
    no_qualified_candidate_reason,
    segment_turns,
    signal_sufficiency_diagnostics,
    valid_signal_runs,
)


def signal(values, *, segments=None, timestamps=None, confidences=None):
    timestamps = timestamps or [index * 100_000 for index in range(len(values))]
    segments = segments or [1 if value is not None else None for value in values]
    confidences = confidences or [
        0.9 if value is not None else None for value in values
    ]
    return [
        TurnSignalSample(
            timestamp,
            segment,
            value,
            confidence,
            "OBSERVED" if value is not None else "MISSING",
        )
        for timestamp, segment, value, confidence in zip(
            timestamps, segments, values, confidences, strict=True
        )
    ]


def config(**updates):
    values = {
        "minimum_peak_prominence": 0.1,
        "minimum_peak_amplitude": 0.1,
        "minimum_peak_separation_us": 0,
        "minimum_turn_duration_us": 100_000,
        "maximum_turn_duration_us": 1_000_000,
        "minimum_valid_samples_per_turn": 3,
        **updates,
    }
    return TurnSegmentationConfig(**values)


@pytest.mark.parametrize(
    "values,expected",
    [([0, 0.8, 0, 0, 0.9, 0], [0.8, 0.9]), ([0, -0.8, 0, 0, -0.9, 0], [-0.8, -0.9])],
)
def test_same_sign_extrema_in_different_temporal_segments_are_retained(
    values, expected
):
    samples = signal(values, segments=[1, 1, 1, 2, 2, 2])
    peaks = ReferencePeakDetector().detect(samples, config())
    assert [peak.value for peak in peaks] == expected
    assert [peak.temporal_segment_id for peak in peaks] == [1, 2]
    assert [peak.signal_run_id for peak in peaks] == [1, 2]


def test_missing_gap_splits_same_temporal_segment_and_preserves_both_peaks():
    samples = signal(
        [0, 0.8, 0, None, 0, 0.9, 0],
        segments=[1, 1, 1, 1, 1, 1, 1],
    )
    peaks = ReferencePeakDetector().detect(samples, config())
    assert [peak.value for peak in peaks] == [0.8, 0.9]
    assert [peak.signal_run_id for peak in peaks] == [1, 2]


def test_minimum_separation_and_same_sign_replacement_are_run_local():
    settings = config(minimum_peak_separation_us=250_000)
    within = ReferencePeakDetector().detect(signal([0, 0.8, 0, -0.9, 0]), settings)
    assert [peak.value for peak in within] == [-0.9]
    separate = signal(
        [0, 0.8, 0, None, 0, -0.9, 0],
        timestamps=[0, 100_000, 150_000, 160_000, 170_000, 180_000, 190_000],
        segments=[1, 1, 1, 1, 1, 1, 1],
    )
    assert [
        peak.value for peak in ReferencePeakDetector().detect(separate, settings)
    ] == [
        0.8,
        -0.9,
    ]
    replacement = ReferencePeakDetector().detect(signal([0, 0.8, 0, 0.9, 0]), config())
    assert [peak.value for peak in replacement] == [0.9]


def test_plateau_choice_and_concatenated_results_are_deterministic_chronological():
    samples = signal(
        [0, 0.8, 0.8, 0, None, 0, -0.9, 0],
        timestamps=[
            500_000,
            600_000,
            700_000,
            800_000,
            900_000,
            10_000,
            20_000,
            30_000,
        ],
        segments=[2, 2, 2, 2, None, 1, 1, 1],
    )
    peaks = ReferencePeakDetector().detect(samples, config())
    assert [peak.timestamp_us for peak in peaks] == [20_000, 600_000]
    assert peaks[1].sample_index == 1


@pytest.mark.parametrize(
    "values,expected_count,expected_direction",
    [
        ([-1, 0, 1], 1, "NEGATIVE_TO_POSITIVE"),
        ([1, 0, -1], 1, "POSITIVE_TO_NEGATIVE"),
        ([1, 0, 0, 1], 0, None),
        ([-1, 0, 0, -1], 0, None),
        ([-1, 0, 0, 1], 1, "NEGATIVE_TO_POSITIVE"),
        ([1, 0, 0, -1], 1, "POSITIVE_TO_NEGATIVE"),
        ([0, 0, 1], 0, None),
        ([1, 0, 0], 0, None),
    ],
)
def test_zero_plateau_semantics(values, expected_count, expected_direction):
    crossings = detect_zero_crossings(signal(values))
    assert len(crossings) == expected_count
    if crossings:
        assert crossings[0].direction == expected_direction


def test_zero_plateau_midpoint_and_direct_irregular_timestamp_policy():
    plateau = detect_zero_crossings(
        signal([-1, 0, 0, 1], timestamps=[0, 100_000, 240_000, 500_000])
    )
    assert plateau[0].timestamp_us == 170_000
    direct = detect_zero_crossings(signal([-1, 3], timestamps=[70_000, 470_000]))
    assert direct[0].timestamp_us == 170_000
    near_zero = detect_zero_crossings(signal([-1, 1e-10, -1e-10, 1]), tolerance=1e-9)
    assert len(near_zero) == 1


@pytest.mark.parametrize(
    "samples",
    [
        signal([-1, None, 1], segments=[1, 1, 1]),
        signal([-1, 1], segments=[1, 2]),
        signal([-1, 1], confidences=[0.9, 0.1]),
    ],
)
def test_zero_crossings_never_cross_invalid_run_boundaries(samples):
    assert detect_zero_crossings(samples, minimum_signal_confidence=0.3) == []


def test_segmentation_boundaries_never_cross_missing_gap_with_same_temporal_segment():
    samples = signal(
        [-0.5, 0, 0.8, 0, -0.5, None, -0.5, 0, 0.9, 0, -0.5],
        segments=[1] * 11,
    )
    settings = config()
    peaks = ReferencePeakDetector().detect(samples, settings)
    crossings = detect_zero_crossings(samples, minimum_signal_confidence=0.3)
    segments = segment_turns(samples, peaks, crossings, settings)
    assert [(item.start_timestamp_us, item.end_timestamp_us) for item in segments] == [
        (100_000, 300_000),
        (700_000, 900_000),
    ]
    assert [item.signal_run_id for item in segments] == [1, 2]


def analyze(samples, settings=None):
    settings = settings or config()
    peaks = ReferencePeakDetector().detect(samples, settings)
    crossings = detect_zero_crossings(
        samples,
        settings.zero_crossing_tolerance,
        minimum_signal_confidence=settings.minimum_signal_confidence,
    )
    segments = segment_turns(samples, peaks, crossings, settings)
    diagnostics = signal_sufficiency_diagnostics(samples, peaks, crossings, settings)
    return classify_real_turn_status(diagnostics, segments), diagnostics, segments


def test_all_five_real_turn_statuses_and_no_peak_reason():
    no_signal = signal([None, None], segments=[None, None])
    assert analyze(no_signal)[0].value == "NOT_ANALYZABLE_NO_VALID_TURN_SIGNAL"
    isolated = signal([0, 0, None, 0, 0], segments=[1, 1, 1, 1, 1])
    assert (
        analyze(isolated)[0].value
        == "NOT_ANALYZABLE_INSUFFICIENT_CONTINUOUS_TARGET_POSE"
    )
    flat = signal([0.01] * 10)
    flat_status, flat_diagnostics, _ = analyze(flat)
    assert flat_status.value == "EXECUTED_NO_QUALIFIED_TURN_CANDIDATES"
    assert (
        no_qualified_candidate_reason(flat_status, flat_diagnostics, config())
        == "SIGNAL_VARIATION_BELOW_AMPLITUDE_THRESHOLD"
    )
    turning = signal([-0.5, 0, 0.8, 0, -0.5])
    rejected = analyze(turning, config(minimum_turn_duration_us=300_000))[0]
    assert rejected.value == "EXECUTED_CANDIDATES_REJECTED"
    accepted = analyze(turning)[0]
    assert accepted.value == "EXECUTED_PROVISIONAL_CANDIDATES"


def temporal_sample(timestamp, *, scale=1.0, translate=(0.0, 0.0), missing_scale=False):
    base = {
        Joint.LEFT_SHOULDER: (-20, 0),
        Joint.RIGHT_SHOULDER: (10, 0),
        Joint.LEFT_ANKLE: (-15, 100),
        Joint.RIGHT_ANKLE: (25, 100),
        Joint.NOSE: (timestamp / 100_000, 20),
    }
    joints = {}
    for joint, (x, y) in base.items():
        provenance = (
            TemporalProvenance.MISSING
            if missing_scale and joint is Joint.RIGHT_ANKLE
            else TemporalProvenance.OBSERVED
        )
        raw_x = (
            None
            if provenance is TemporalProvenance.MISSING
            else x * scale + translate[0]
        )
        raw_y = (
            None
            if provenance is TemporalProvenance.MISSING
            else y * scale + translate[1]
        )
        joints[joint] = TemporalJoint2D(
            joint,
            raw_x,
            raw_y,
            raw_x,
            raw_y,
            raw_x,
            raw_y,
            0.9 if raw_x is not None else None,
            provenance,
            raw_x is not None,
        )
    return StabilizedPoseSample(
        timestamp,
        timestamp // 100_000,
        1,
        FrameGeometry(640, 480),
        "target-1",
        1,
        TargetIdentityState.LOCKED,
        joints,
        len(joints),
        0,
        0,
        TemporalPoseQuality.GOOD,
    )


def test_symmetric_scale_uses_bilateral_centers_and_is_translation_invariant():
    sample = temporal_sample(0)
    assert symmetric_frame_body_scale(sample) == pytest.approx(math.hypot(10, 100))
    translated = temporal_sample(0, translate=(900, -400))
    assert symmetric_frame_body_scale(translated) == pytest.approx(math.hypot(10, 100))


def test_segment_median_scale_metrics_are_translation_and_uniform_scale_invariant():
    original = [temporal_sample(timestamp) for timestamp in (0, 100_000, 200_000)]
    transformed = [
        temporal_sample(timestamp, scale=3, translate=(800, -250))
        for timestamp in (0, 100_000, 200_000)
    ]
    first = temporal_stability_metrics(original)
    second = temporal_stability_metrics(transformed)
    for key in (
        "raw_joint_step_median",
        "stabilized_joint_step_median",
        "raw_second_difference_median",
        "stabilized_second_difference_median",
    ):
        assert first[key] == pytest.approx(second[key])
    assert first["raw_joint_step_median"] == first["stabilized_joint_step_median"]
    assert first["normalization_strategy"] == (
        "SEGMENT_MEDIAN_RAW_SHOULDER_CENTER_TO_ANKLE_CENTER"
    )


def test_per_frame_scales_collapse_to_one_segment_median():
    samples = [
        temporal_sample(0, scale=1),
        temporal_sample(100_000, scale=2),
        temporal_sample(200_000, scale=4),
    ]
    assert segment_body_scales(samples)[1] == pytest.approx(2 * math.hypot(10, 100))


def test_raw_and_stabilized_steps_use_the_same_segment_denominator():
    first = temporal_sample(0)
    second = temporal_sample(0)
    shifted_joints = {
        joint: replace(
            point,
            raw_x_px=point.raw_x_px + 10,
            support_x_px=point.support_x_px + 10,
            stabilized_x_px=point.stabilized_x_px + 20,
        )
        for joint, point in second.joints.items()
    }
    second = replace(second, timestamp_us=100_000, frame_index=1, joints=shifted_joints)
    metrics = temporal_stability_metrics([first, second])
    denominator = math.hypot(10, 100)
    assert metrics["raw_joint_step_median"] == pytest.approx(10 / denominator)
    assert metrics["stabilized_joint_step_median"] == pytest.approx(20 / denominator)


def test_unavailable_body_scale_segment_is_excluded_without_nonfinite_values():
    metrics = temporal_stability_metrics(
        [temporal_sample(timestamp, missing_scale=True) for timestamp in (0, 100_000)]
    )
    assert metrics["body_scale_valid_segment_count"] == 0
    assert metrics["body_scale_unavailable_segment_count"] == 1
    assert metrics["raw_joint_step_median"] is None
    assert metrics["joint_step_comparison_count"] == 0
    for value in metrics.values():
        assert not isinstance(value, float) or math.isfinite(value)
    json.dumps(metrics, allow_nan=False)


def test_valid_signal_run_duration_uses_actual_timestamps_and_rejects_nonmonotonic():
    samples = signal([0, 1], timestamps=[50_000, 875_000])
    runs = valid_signal_runs(samples, config())
    assert runs[0].duration_us == 825_000
    assert valid_signal_runs(signal([1]), config())[0].duration_us == 0
    with pytest.raises(ValueError, match="strictly increase"):
        valid_signal_runs(signal([0, 1], timestamps=[100_000, 100_000]), config())
