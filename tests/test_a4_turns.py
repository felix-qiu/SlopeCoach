from __future__ import annotations

import json

import pytest

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import FrameGeometry, Joint
from slopecoach_ml.temporal import (
    StabilizedPoseSample,
    TemporalJoint2D,
    TemporalPoseQuality,
    TemporalProvenance,
    run_turn_golden,
)
from slopecoach_ml.turns import (
    ReferencePeakDetector,
    SciPyFindPeaksDetector,
    TurnSegmentationConfig,
    TurnSignalSample,
    detect_zero_crossings,
    segment_turns,
    signed_lateral_body_proxy,
)


def temporal_pose(
    offset=0.0, *, scale=1.0, translate=(0.0, 0.0), missing=None, confidence=0.9
):
    base = {
        Joint.LEFT_SHOULDER: (-10, 0),
        Joint.RIGHT_SHOULDER: (10, 0),
        Joint.LEFT_HIP: (-8, 40),
        Joint.RIGHT_HIP: (8, 40),
        Joint.LEFT_KNEE: (-7 + offset, 80),
        Joint.RIGHT_KNEE: (7 + offset, 80),
        Joint.LEFT_ANKLE: (-6 + offset, 120),
        Joint.RIGHT_ANKLE: (6 + offset, 120),
    }
    joints = {}
    for joint in Joint:
        coords = base.get(joint, (0, 0))
        x, y = coords[0] * scale + translate[0], coords[1] * scale + translate[1]
        usable = joint is not missing
        joints[joint] = TemporalJoint2D(
            joint,
            x if usable else None,
            y if usable else None,
            x if usable else None,
            y if usable else None,
            x if usable else None,
            y if usable else None,
            confidence if usable else None,
            TemporalProvenance.OBSERVED if usable else TemporalProvenance.MISSING,
            usable,
        )
    return StabilizedPoseSample(
        100_000,
        3,
        1,
        FrameGeometry(640, 480),
        "target-1",
        1,
        TargetIdentityState.LOCKED,
        joints,
        17,
        0,
        0,
        TemporalPoseQuality.GOOD,
    )


def signal(values, timestamps=None, segment=1):
    timestamps = timestamps or [index * 100_000 for index in range(len(values))]
    return [
        TurnSignalSample(
            t,
            segment if value is not None else None,
            value,
            0.9 if value is not None else None,
            "OBSERVED" if value is not None else "MISSING",
        )
        for t, value in zip(timestamps, values, strict=True)
    ]


def test_turn_proxy_sign_symmetry_scale_translation_and_oof() -> None:
    assert signed_lateral_body_proxy(temporal_pose()).value == pytest.approx(
        0, abs=1e-12
    )
    positive = signed_lateral_body_proxy(temporal_pose(20)).value
    negative = signed_lateral_body_proxy(temporal_pose(-20)).value
    assert positive == pytest.approx(-negative)
    assert positive != 0
    assert signed_lateral_body_proxy(temporal_pose(20, scale=3)).value == pytest.approx(
        positive
    )
    assert signed_lateral_body_proxy(
        temporal_pose(20, translate=(-1000, 2000))
    ).value == pytest.approx(positive)


def test_turn_proxy_missing_low_confidence_and_finite_contract() -> None:
    assert (
        signed_lateral_body_proxy(temporal_pose(missing=Joint.LEFT_ANKLE)).value is None
    )
    low = temporal_pose(confidence=0.1)
    assert signed_lateral_body_proxy(low).value is None


def test_reference_peaks_positive_negative_prominence_separation_and_ties() -> None:
    config = TurnSegmentationConfig(
        minimum_peak_prominence=0.2,
        minimum_peak_amplitude=0.2,
        minimum_peak_separation_us=250_000,
    )
    samples = signal(
        [0, 0.8, 0, -0.7, -0.4, 0.05, 0],
        [0, 100_000, 200_000, 400_000, 500_000, 700_000, 800_000],
    )
    peaks = ReferencePeakDetector().detect(samples, config)
    assert [round(item.value, 2) for item in peaks] == [0.8, -0.7]
    assert ReferencePeakDetector().detect(signal([0, 0.05, 0]), config) == []
    tied = ReferencePeakDetector().detect(signal([0, 0.8, 0.8, 0]), config)
    assert len(tied) == 1 and tied[0].sample_index == 1


def test_peak_detection_irregular_timestamp_missing_and_segment_gap() -> None:
    config = TurnSegmentationConfig(
        minimum_peak_prominence=0.1, minimum_peak_amplitude=0.1
    )
    samples = signal(
        [0, 0.8, 0, None, 0, -0.9, 0],
        [0, 70_000, 250_000, 300_000, 800_000, 990_000, 1_300_000],
    )
    peaks = ReferencePeakDetector().detect(samples, config)
    assert [item.timestamp_us for item in peaks] == [70_000, 990_000]


def test_zero_crossings_irregular_exact_zero_no_duplicates_or_gap() -> None:
    samples = signal(
        [-1, 1, 0, -1, None, 1], [0, 300_000, 500_000, 700_000, 800_000, 900_000]
    )
    crossings = detect_zero_crossings(samples)
    assert [item.timestamp_us for item in crossings] == [150_000, 500_000]
    assert crossings[0].direction == "NEGATIVE_TO_POSITIVE"
    assert crossings[1].direction == "POSITIVE_TO_NEGATIVE"


def test_turn_segments_boundaries_status_and_no_cross_segment() -> None:
    config = TurnSegmentationConfig(
        minimum_peak_prominence=0.1,
        minimum_peak_amplitude=0.1,
        minimum_turn_duration_us=200_000,
        maximum_turn_duration_us=800_000,
        minimum_valid_samples_per_turn=3,
    )
    samples = signal(
        [-0.5, 0, 0.8, 0, -0.7, 0], [0, 100_000, 300_000, 500_000, 700_000, 900_000]
    )
    peaks = ReferencePeakDetector().detect(samples, config)
    crossings = detect_zero_crossings(samples)
    segments = segment_turns(samples, peaks, crossings, config)
    assert segments[0].start_timestamp_us == 100_000
    assert segments[0].end_timestamp_us == 500_000
    assert segments[0].temporal_segment_id == 1
    json.dumps([item.to_dict() for item in segments], allow_nan=False)


@pytest.mark.parametrize(
    "config_update,expected",
    [
        ({"minimum_turn_duration_us": 500_000}, "REJECTED_SHORT"),
        ({"maximum_turn_duration_us": 300_000}, "REJECTED_LONG"),
        ({"minimum_valid_samples_per_turn": 4}, "REJECTED_LOW_COVERAGE"),
    ],
)
def test_turn_segment_rejection_statuses(config_update, expected) -> None:
    settings = {
        "minimum_peak_prominence": 0.1,
        "minimum_peak_amplitude": 0.1,
        "minimum_turn_duration_us": 100_000,
        "maximum_turn_duration_us": 1_000_000,
        "minimum_valid_samples_per_turn": 3,
        **config_update,
    }
    config = TurnSegmentationConfig(**settings)
    samples = signal([-0.5, 0, 0.8, 0, -0.5], [0, 100_000, 300_000, 500_000, 700_000])
    segments = segment_turns(
        samples,
        ReferencePeakDetector().detect(samples, config),
        detect_zero_crossings(samples),
        config,
    )
    assert segments[0].status.value == expected


def test_partial_turn_and_separate_temporal_segments_never_bridge() -> None:
    config = TurnSegmentationConfig(
        minimum_peak_prominence=0.1,
        minimum_peak_amplitude=0.1,
        minimum_valid_samples_per_turn=2,
    )
    first = signal([0.2, 0.8, 0.2], [0, 100_000, 200_000], segment=1)
    second = signal([-0.2, -0.8, -0.2], [300_000, 400_000, 500_000], segment=2)
    samples = first + second
    peaks = ReferencePeakDetector().detect(samples, config)
    segments = segment_turns(samples, peaks, detect_zero_crossings(samples), config)
    assert all(segment.status.value == "PARTIAL" for segment in segments)
    assert [segment.temporal_segment_id for segment in segments] == [1, 2]


def test_scipy_adapter_explicitly_fails_if_not_configured(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__
    monkeypatch.setattr(
        builtins,
        "__import__",
        lambda name, *args, **kwargs: (_ for _ in ()).throw(ImportError())
        if name.startswith("scipy")
        else real_import(name, *args, **kwargs),
    )
    with pytest.raises(RuntimeError, match="SCIPY_PEAK_DETECTOR_NOT_CONFIGURED"):
        SciPyFindPeaksDetector().detect([], TurnSegmentationConfig())


def test_turn_golden_known_events() -> None:
    result = run_turn_golden("fixtures/golden_turn_signal_001.json")
    assert result["golden_passed"]
    assert result["accepted_apex_count"] == 3
    assert result["apex_timestamps_us"] == [330_000, 760_000, 2_400_000]
    assert result["zero_crossing_count"] == 6
