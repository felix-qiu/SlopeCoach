from __future__ import annotations

import json
from dataclasses import replace

import pytest

from slopecoach_ml.biomechanics import (
    FEATURE_REGISTRY_V1,
    FRAME_FEATURE_REGISTRY_V1,
    BiomechanicsFact,
    BiomechanicsFactScope,
    BiomechanicsFactStatus,
    BiomechanicsFeatureConfig,
    BiomechanicsFeatureFamily,
    aggregate_frame_facts,
    analyze_temporal_biomechanics,
    compute_frame_biomechanics,
    compute_turn_biomechanics,
    derivative_aggregates,
    run_biomechanics_golden,
)
from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import FrameGeometry, Joint
from slopecoach_ml.temporal import (
    StabilizedPoseSample,
    TemporalJoint2D,
    TemporalPoseQuality,
    TemporalProvenance,
)
from slopecoach_ml.turns import TurnPhaseSign, TurnSegment, TurnSegmentStatus


def sample(
    timestamp=0,
    *,
    segment=1,
    scale=1.0,
    translate=(200, 100),
    confidence=0.9,
    missing=None,
    interpolated=None,
    geometry=None,
):
    base = {
        Joint.LEFT_SHOULDER: (40, 0),
        Joint.RIGHT_SHOULDER: (60, 0),
        Joint.LEFT_HIP: (0, 50),
        Joint.RIGHT_HIP: (20, 50),
        Joint.LEFT_KNEE: (0, 100),
        Joint.RIGHT_KNEE: (20, 100),
        Joint.LEFT_ANKLE: (50, 100),
        Joint.RIGHT_ANKLE: (70, 100),
    }
    joints = {}
    for joint in Joint:
        x, y = base.get(joint, (10, 10))
        x, y = x * scale + translate[0], y * scale + translate[1]
        unavailable = joint is missing
        provenance = (
            TemporalProvenance.INTERPOLATED
            if joint is interpolated
            else TemporalProvenance.OBSERVED
        )
        joints[joint] = TemporalJoint2D(
            joint,
            None if unavailable else x,
            None if unavailable else y,
            None if unavailable else x,
            None if unavailable else y,
            None if unavailable else x,
            None if unavailable else y,
            None if unavailable else confidence,
            TemporalProvenance.MISSING if unavailable else provenance,
            not unavailable,
        )
    return StabilizedPoseSample(
        timestamp,
        timestamp // 100_000,
        segment,
        geometry or FrameGeometry(800, 600),
        "target-1",
        1,
        TargetIdentityState.LOCKED,
        joints,
        17,
        0,
        0,
        TemporalPoseQuality.GOOD,
    )


def facts(current, body_scale=100.0):
    return {
        fact.feature_id: fact
        for fact in compute_frame_biomechanics(current, body_scale)
    }


def test_registry_and_known_frame_geometry_are_deterministic():
    assert len(FRAME_FEATURE_REGISTRY_V1) == 14
    assert len(FEATURE_REGISTRY_V1) == 30
    assert len({item.feature_id for item in FEATURE_REGISTRY_V1}) == 30
    result = facts(sample())
    assert result["left_knee_angle_2d_deg"].value == pytest.approx(90)
    assert result["right_knee_angle_2d_deg"].value == pytest.approx(90)
    assert result["bilateral_knee_mean_angle_2d_deg"].value == pytest.approx(90)
    assert result["bilateral_knee_abs_difference_2d_deg"].value == 0
    assert result["ankle_separation_body_scale"].value == pytest.approx(0.2)
    assert result["shoulder_to_ankle_screen_lateral_offset_body_scale"].value < 0
    assert result["torso_screen_inclination_deg"].value < 0
    assert result["shoulder_hip_axis_difference_2d_deg"].value == 0


def test_translation_and_uniform_scale_invariance():
    original = facts(sample())
    translated = facts(sample(translate=(500, 300)))
    scaled = facts(sample(scale=3), 300)
    for feature in (
        "left_knee_angle_2d_deg",
        "ankle_separation_body_scale",
        "ankle_to_shoulder_separation_ratio_2d",
        "shoulder_to_ankle_screen_lateral_offset_body_scale",
    ):
        assert translated[feature].value == pytest.approx(original[feature].value)
        assert scaled[feature].value == pytest.approx(original[feature].value)


@pytest.mark.parametrize(
    "current,expected",
    [
        (sample(confidence=0.1), "LOW_CONFIDENCE"),
        (sample(missing=Joint.LEFT_KNEE), "REQUIRED_JOINT_MISSING"),
        (sample(translate=(-500, 0)), "REQUIRED_JOINT_OUT_OF_FRAME"),
        (
            sample(geometry=FrameGeometry(800, 600, pixel_aspect_ratio=1.2)),
            "UNSUPPORTED_PIXEL_ASPECT_RATIO",
        ),
    ],
)
def test_invalid_evidence_is_null(current, expected):
    fact = facts(current)["left_knee_angle_2d_deg"]
    assert fact.value is None and fact.status.value == expected


def test_degenerate_and_provenance_semantics_no_nonfinite():
    current = sample(interpolated=Joint.LEFT_ANKLE)
    result = facts(current)
    left = result["left_knee_angle_2d_deg"]
    assert left.observed_joint_count == 2
    assert left.interpolated_joint_count == 1
    assert left.support_confidence == pytest.approx(0.9)
    degenerate = sample()
    knee = degenerate.joints[Joint.LEFT_KNEE]
    ankle = degenerate.joints[Joint.LEFT_ANKLE]
    degenerate.joints[Joint.LEFT_ANKLE] = replace(
        ankle,
        stabilized_x_px=knee.stabilized_x_px,
        stabilized_y_px=knee.stabilized_y_px,
    )
    fact = facts(degenerate)["left_knee_angle_2d_deg"]
    assert (
        fact.status is BiomechanicsFactStatus.DEGENERATE_GEOMETRY and fact.value is None
    )
    json.dumps([item.to_dict() for item in result.values()], allow_nan=False)


def simple_fact(
    timestamp,
    value,
    *,
    segment=1,
    status=BiomechanicsFactStatus.AVAILABLE,
    interpolated=0,
):
    return BiomechanicsFact(
        "left_knee_angle_2d_deg",
        BiomechanicsFeatureFamily.STANCE_PROXY,
        BiomechanicsFactScope.FRAME,
        "deg",
        value if status is BiomechanicsFactStatus.AVAILABLE else None,
        status,
        timestamp,
        segment,
        support_confidence=0.8,
        observed_joint_count=3 - interpolated,
        interpolated_joint_count=interpolated,
    )


def test_temporal_aggregation_nulls_missing_and_keeps_segments_separate():
    items = (
        simple_fact(0, 90),
        simple_fact(
            100_000, None, status=BiomechanicsFactStatus.REQUIRED_JOINT_MISSING
        ),
        simple_fact(200_000, 70, interpolated=1),
        simple_fact(300_000, 80),
        simple_fact(0, 100, segment=2),
    )
    result = aggregate_frame_facts(items, BiomechanicsFeatureConfig())
    first = next(
        item
        for item in result
        if item.temporal_segment_id == 1 and item.feature_id == "left_knee_angle_2d_deg"
    )
    assert (first.median, first.minimum, first.maximum, first.range) == (80, 70, 90, 20)
    assert first.support_ratio == pytest.approx(0.75)
    assert (
        first.observed_only_sample_count == 2
        and first.interpolated_support_sample_count == 1
    )
    second = next(
        item
        for item in result
        if item.temporal_segment_id == 2 and item.feature_id == "left_knee_angle_2d_deg"
    )
    assert (
        second.status is BiomechanicsFactStatus.INSUFFICIENT_SAMPLES
        and second.median is None
    )


def test_derivative_uses_timestamps_and_missing_breaks_chain():
    items = (simple_fact(0, 100), simple_fact(500_000, 110))
    result = derivative_aggregates(items, BiomechanicsFeatureConfig())
    left = next(item for item in result if item.feature_id.startswith("left_knee"))
    assert left.median == pytest.approx(20)
    broken = derivative_aggregates(
        (
            simple_fact(0, 100),
            simple_fact(
                250_000, None, status=BiomechanicsFactStatus.REQUIRED_JOINT_MISSING
            ),
            simple_fact(500_000, 110),
        ),
        BiomechanicsFeatureConfig(),
    )
    assert (
        next(item for item in broken if item.feature_id.startswith("left_knee")).median
        is None
    )
    with pytest.raises(ValueError, match="strictly increase"):
        derivative_aggregates(
            (simple_fact(100, 1), simple_fact(100, 2)), BiomechanicsFeatureConfig()
        )


def turn(status=TurnSegmentStatus.VALID, *, start=0, end=200_000, run=1, apex=100_000):
    return TurnSegment(
        "turn-1",
        1,
        run,
        start,
        apex,
        end,
        TurnPhaseSign.POSITIVE_PHASE,
        0.8,
        0.8,
        end - start if start is not None and end is not None else None,
        3,
        0,
        0,
        1,
        status,
    )


def test_turn_facts_are_gated_run_local_and_boundary_aware():
    frame_facts = tuple(
        fact
        for current in (sample(0), sample(100_000), sample(200_000))
        for fact in compute_frame_biomechanics(current, 100)
    )
    result = compute_turn_biomechanics(
        [turn()], frame_facts, {1: {0, 100_000, 200_000}}, BiomechanicsFeatureConfig()
    )
    assert len(result) == 1
    assert {fact.feature_id: fact.value for fact in result[0].facts}[
        "bilateral_knee_mean_angle_at_apex_deg"
    ] == pytest.approx(90)
    assert (
        compute_turn_biomechanics(
            [turn(TurnSegmentStatus.REJECTED_SHORT)],
            frame_facts,
            {1: {0, 100_000, 200_000}},
            BiomechanicsFeatureConfig(),
        )
        == ()
    )
    assert (
        compute_turn_biomechanics(
            [turn(run=2)],
            frame_facts,
            {1: {0, 100_000, 200_000}},
            BiomechanicsFeatureConfig(),
        )[0]
        .facts[2]
        .value
        is None
    )
    partial = compute_turn_biomechanics(
        [turn(TurnSegmentStatus.PARTIAL, start=None)],
        frame_facts,
        {1: {0, 100_000, 200_000}},
        BiomechanicsFeatureConfig(),
    )[0]
    start_fact = next(
        fact for fact in partial.facts if fact.feature_id.endswith("at_start_deg")
    )
    assert start_fact.status is BiomechanicsFactStatus.TURN_BOUNDARY_UNAVAILABLE


def test_apex_tolerance_and_golden_serialization():
    frame_facts = compute_frame_biomechanics(sample(0), 100)
    result = compute_turn_biomechanics(
        [turn(apex=500_000)], frame_facts, {1: {0}}, BiomechanicsFeatureConfig()
    )
    apex = next(
        fact
        for fact in result[0].facts
        if fact.feature_id == "bilateral_knee_mean_angle_at_apex_deg"
    )
    assert apex.value is None
    golden = run_biomechanics_golden("fixtures/golden_temporal_biomechanics_001.json")
    assert golden["golden_passed"]
    json.dumps(golden, allow_nan=False, sort_keys=True)


def test_pipeline_empty_unsafe_input_is_honest():
    result = analyze_temporal_biomechanics([sample(segment=None)])
    assert result.frame_facts == ()
    assert not any(
        item["available_frame_count"] for item in result.feature_coverage.values()
    )
