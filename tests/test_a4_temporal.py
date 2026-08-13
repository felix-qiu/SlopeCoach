from __future__ import annotations

import json
import math

import pytest

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import (
    BoundingBox2D,
    FrameGeometry,
    Joint,
    Keypoint2D,
    PersonPose2D,
)
from slopecoach_ml.temporal import (
    OneEuroFilter1D,
    TargetPoseSample,
    TemporalPoseConfig,
    TemporalProvenance,
    run_temporal_golden,
    stabilize_target_pose_stream,
)


GEOMETRY = FrameGeometry(640, 480)


def pose(x=100.0, confidence=0.9, *, all_joints=False):
    joints = (
        {
            joint: Keypoint2D(x + index, 100 + index * 5, confidence)
            for index, joint in enumerate(Joint)
        }
        if all_joints
        else {Joint.LEFT_ANKLE: Keypoint2D(x, 300, confidence)}
    )
    return PersonPose2D(BoundingBox2D(50, 20, 100, 300), 0.9, joints, 1)


def sample(
    timestamp,
    state=TargetIdentityState.LOCKED,
    raw=None,
    *,
    track=1,
    confidence=0.9,
    geometry=GEOMETRY,
    target="target-1",
    discontinuity=False,
):
    return TargetPoseSample(
        timestamp,
        timestamp // 10_000,
        target,
        track,
        state,
        confidence,
        geometry,
        raw,
        explicit_discontinuity=discontinuity,
    )


def test_one_euro_constant_irregular_and_reset() -> None:
    filt = OneEuroFilter1D()
    assert [filt.filter(5, t) for t in (0, 70_000, 250_000)] == [5, 5, 5]
    filt.reset()
    assert filt.filter(20, 1_000_000) == 20


def test_one_euro_step_is_deterministic_timestamp_aware_and_strict() -> None:
    a, b = OneEuroFilter1D(), OneEuroFilter1D()
    times = [0, 100_000, 350_000]
    values = [0, 10, 10]
    assert [
        a.filter(v, t) for v, t in zip(values, times, strict=True)
    ] == pytest.approx([b.filter(v, t) for v, t in zip(values, times, strict=True)])
    with pytest.raises(ValueError, match="strictly increase"):
        a.filter(10, 350_000)
    with pytest.raises(ValueError):
        OneEuroFilter1D().filter(float("nan"), 0)


@pytest.mark.parametrize(
    "unsafe",
    [
        TargetIdentityState.UNINITIALIZED,
        TargetIdentityState.SUSPECT,
        TargetIdentityState.LOST,
        TargetIdentityState.RECOVERING,
        TargetIdentityState.AMBIGUOUS,
    ],
)
def test_identity_uncertainty_is_hard_boundary(unsafe) -> None:
    run = stabilize_target_pose_stream(
        [
            sample(0, raw=pose()),
            sample(100_000, unsafe, pose()),
            sample(200_000, raw=pose(110)),
        ]
    )
    assert [item.temporal_segment_id for item in run.samples] == [1, None, 2]
    assert (
        run.samples[1].joint(Joint.LEFT_ANKLE).provenance is TemporalProvenance.MISSING
    )


def test_continuous_locked_segment_and_short_timestamp_interpolation() -> None:
    run = stabilize_target_pose_stream(
        [
            sample(0, raw=pose(0)),
            sample(80_000, raw=None),
            sample(200_000, raw=pose(20)),
        ]
    )
    assert [item.temporal_segment_id for item in run.samples] == [1, 1, 1]
    middle = run.samples[1].joint(Joint.LEFT_ANKLE)
    assert middle.provenance is TemporalProvenance.INTERPOLATED
    assert middle.support_x_px == pytest.approx(8)
    assert middle.support_confidence == pytest.approx(0.9)


def test_multiple_short_missing_samples_are_timestamp_weighted() -> None:
    run = stabilize_target_pose_stream(
        [
            sample(0, raw=pose(0)),
            sample(50_000),
            sample(150_000),
            sample(250_000, raw=pose(25)),
        ]
    )
    assert run.samples[1].joint(Joint.LEFT_ANKLE).support_x_px == pytest.approx(5)
    assert run.samples[2].joint(Joint.LEFT_ANKLE).support_x_px == pytest.approx(15)


def test_long_gap_low_confidence_and_geometry_boundary_block_interpolation() -> None:
    config = TemporalPoseConfig(
        maximum_interpolation_gap_us=200_000, hard_reset_gap_us=500_000
    )
    run = stabilize_target_pose_stream(
        [sample(0, raw=pose(0)), sample(250_000), sample(500_000, raw=pose(20))], config
    )
    assert (
        run.samples[1].joint(Joint.LEFT_ANKLE).provenance is TemporalProvenance.MISSING
    )
    low = stabilize_target_pose_stream(
        [sample(0, raw=pose(0, 0.1)), sample(100_000), sample(200_000, raw=pose(20))]
    )
    assert (
        low.samples[1].joint(Joint.LEFT_ANKLE).provenance is TemporalProvenance.MISSING
    )
    changed = stabilize_target_pose_stream(
        [
            sample(0, raw=pose()),
            sample(100_000, raw=pose(), geometry=FrameGeometry(800, 600)),
        ]
    )
    assert changed.temporal_segment_count == 2


def test_track_target_timestamp_gap_and_discontinuity_reset_semantics() -> None:
    run = stabilize_target_pose_stream(
        [
            sample(0, raw=pose()),
            sample(100_000, raw=pose(), track=2),
            sample(700_000, raw=pose(), track=2),
            sample(800_000, raw=pose(), track=2, discontinuity=True),
            sample(900_000, raw=pose(), track=2),
        ]
    )
    assert [item.temporal_segment_id for item in run.samples] == [1, 2, 3, None, 4]
    assert all(item.target_id == "target-1" for item in run.samples)


def test_stream_rejects_duplicate_decreasing_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="strictly increase"):
        stabilize_target_pose_stream([sample(0, raw=pose()), sample(0, raw=pose())])
    with pytest.raises(ValueError):
        stabilize_target_pose_stream([sample(0, raw=pose(float("inf")))])


def test_out_of_frame_finite_coordinate_is_preserved() -> None:
    run = stabilize_target_pose_stream([sample(0, raw=pose(-20))])
    point = run.samples[0].joint(Joint.LEFT_ANKLE)
    assert point.raw_x_px == -20
    assert math.isfinite(point.stabilized_x_px)


def test_joint_and_axis_filter_state_are_independent() -> None:
    run = stabilize_target_pose_stream(
        [
            sample(0, raw=pose(0, all_joints=True)),
            sample(100_000, raw=pose(10, all_joints=True)),
        ]
    )
    left = run.samples[-1].joint(Joint.LEFT_ANKLE)
    right = run.samples[-1].joint(Joint.RIGHT_ANKLE)
    assert left.stabilized_x_px != right.stabilized_x_px
    assert left.stabilized_x_px != left.stabilized_y_px


def test_temporal_golden_and_allow_nan_serialization() -> None:
    result = run_temporal_golden("fixtures/golden_temporal_pose_001.json")
    assert result["golden_passed"]
    assert result["temporal_segment_count"] == 2
    assert result["short_interpolated_count"] == 1
    assert result["long_missing_count"] == 2
    assert (
        result["stabilized_mean_absolute_error_px"]
        < result["raw_mean_absolute_error_px"]
    )
    json.dumps(result, allow_nan=False)
