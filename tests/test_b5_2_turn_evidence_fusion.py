from __future__ import annotations

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import FrameGeometry, Joint
from slopecoach_ml.temporal import (
    StabilizedPoseSample,
    TemporalJoint2D,
    TemporalPoseQuality,
    TemporalProvenance,
)
from slopecoach_ml.turns import (
    TurnEvidenceFusionConfig,
    TurnSegmentationConfig,
    build_turn_debug_trace,
    detect_turns_with_evidence_fusion,
)


def _pose_sample(
    index: int,
    *,
    lateral: float,
    center_x: float,
    shoulder_tilt: float = 0.0,
    hip_tilt: float = 0.0,
    knee_bias: float = 0.0,
    step_us: int = 200_000,
) -> StabilizedPoseSample:
    geometry = FrameGeometry(640, 480)

    def point(x: float, y: float) -> tuple[float, float]:
        return (x + center_x, y)

    coordinates = {
        Joint.LEFT_SHOULDER: point(-10, shoulder_tilt),
        Joint.RIGHT_SHOULDER: point(10, -shoulder_tilt),
        Joint.LEFT_HIP: point(-8, 40 + hip_tilt),
        Joint.RIGHT_HIP: point(8, 40 - hip_tilt),
        Joint.LEFT_KNEE: point(-7 + lateral - knee_bias, 80),
        Joint.RIGHT_KNEE: point(7 + lateral + knee_bias, 80),
        Joint.LEFT_ANKLE: point(-6 + lateral - knee_bias, 120),
        Joint.RIGHT_ANKLE: point(6 + lateral + knee_bias, 120),
    }
    joints = {}
    for joint in Joint:
        x, y = coordinates.get(joint, (0.0, 0.0))
        joints[joint] = TemporalJoint2D(
            joint,
            x,
            y,
            x,
            y,
            x,
            y,
            0.95,
            TemporalProvenance.OBSERVED,
            True,
        )
    return StabilizedPoseSample(
        index * step_us,
        index,
        1,
        geometry,
        "target-1",
        1,
        TargetIdentityState.LOCKED,
        joints,
        17,
        0,
        0,
        TemporalPoseQuality.GOOD,
    )


def _run(pattern: list[tuple[float, float, float, float, float]]):
    samples = [
        _pose_sample(
            index,
            lateral=lateral,
            center_x=center_x,
            shoulder_tilt=shoulder_tilt,
            hip_tilt=hip_tilt,
            knee_bias=knee_bias,
        )
        for index, (lateral, center_x, shoulder_tilt, hip_tilt, knee_bias) in enumerate(
            pattern
        )
    ]
    config = TurnSegmentationConfig()
    result = detect_turns_with_evidence_fusion(
        samples, config, TurnEvidenceFusionConfig()
    )
    trace = build_turn_debug_trace(
        samples,
        list(result.signal),
        list(result.peaks),
        list(result.segments),
        list(result.crossings),
        config,
        evidence_samples=list(result.evidence_samples),
        raw_peaks=list(result.raw_peaks),
        rejected_peaks=list(result.rejected_peaks),
        fusion_summary=result.fusion_summary,
        fusion_config=result.fusion_config,
    )
    return samples, result, trace


def test_existing_no_turn_signal_does_not_create_turn() -> None:
    _, result, trace = _run(
        [
            (value, index * 4.0, 0.0, 0.0, 0.0)
            for index, value in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ]
    )

    assert result.raw_peaks == ()
    assert result.peaks == ()
    assert result.segments == ()
    assert trace.to_dict()["turn_debug_summary"]["candidate_count"] == 0


def test_synthetic_left_right_turn_creates_candidate_and_valid_segment() -> None:
    _, result, trace = _run(
        [
            (-9, 0, -1.0, -1.0, -1.0),
            (-6, 2, -0.8, -0.8, -0.7),
            (-2, 5, -0.2, -0.2, -0.2),
            (3, 9, 0.4, 0.4, 0.4),
            (9, 14, 1.0, 1.0, 1.0),
            (4, 17, 0.4, 0.4, 0.4),
            (-2, 18, -0.2, -0.2, -0.2),
            (-8, 16, -1.0, -1.0, -1.0),
            (-3, 12, -0.4, -0.4, -0.3),
            (6, 7, 0.7, 0.7, 0.7),
            (10, 2, 1.0, 1.0, 1.0),
            (2, -1, 0.2, 0.2, 0.2),
            (-7, -3, -0.8, -0.8, -0.8),
        ]
    )

    payload = trace.to_dict()
    assert len(result.raw_peaks) >= 1
    assert len(result.peaks) >= 1
    assert any(segment.status.value == "VALID" for segment in result.segments)
    assert payload["turn_debug_summary"]["candidate_count"] == len(result.raw_peaks)
    assert payload["turn_debug_summary"]["qualified_turn_segment_count"] >= 1
    assert payload["turn_debug_summary"]["dominant_evidence_source"] in {
        "lateral_score",
        "orientation_score",
        "trajectory_score",
        "lower_body_score",
    }
    assert {
        "timestamp_us",
        "lateral_score",
        "orientation_score",
        "trajectory_score",
        "lower_body_score",
        "fused_turn_evidence_score",
        "detector_state",
    } <= payload["samples"][0].keys()


def test_same_input_produces_same_fused_score_sha() -> None:
    pattern = [
        (-9, 0, -1.0, -1.0, -1.0),
        (-6, 2, -0.8, -0.8, -0.7),
        (-2, 5, -0.2, -0.2, -0.2),
        (3, 9, 0.4, 0.4, 0.4),
        (9, 14, 1.0, 1.0, 1.0),
        (4, 17, 0.4, 0.4, 0.4),
        (-2, 18, -0.2, -0.2, -0.2),
        (-8, 16, -1.0, -1.0, -1.0),
        (-3, 12, -0.4, -0.4, -0.3),
        (6, 7, 0.7, 0.7, 0.7),
        (10, 2, 1.0, 1.0, 1.0),
        (2, -1, 0.2, 0.2, 0.2),
        (-7, -3, -0.8, -0.8, -0.8),
    ]
    _, _, first = _run(pattern)
    _, _, second = _run(pattern)

    assert (
        first.to_dict()["turn_debug_summary"]["fused_score_sha256"]
        == second.to_dict()["turn_debug_summary"]["fused_score_sha256"]
    )
    assert first.turn_debug_sha256 == second.turn_debug_sha256


def test_single_direction_movement_does_not_become_turn() -> None:
    _, result, trace = _run(
        [
            (-8, 0, -1.0, -1.0, -1.0),
            (-4, 4, -0.5, -0.5, -0.4),
            (0, 9, 0.0, 0.0, 0.0),
            (5, 15, 0.7, 0.7, 0.6),
            (9, 22, 1.0, 1.0, 1.0),
            (10, 30, 1.0, 1.0, 1.0),
            (10, 39, 1.0, 1.0, 1.0),
        ]
    )

    assert result.raw_peaks == ()
    assert result.peaks == ()
    assert result.segments == ()
    assert trace.to_dict()["turn_debug_summary"]["candidate_count"] == 0
