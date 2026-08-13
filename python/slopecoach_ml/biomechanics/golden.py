"""Independent-data A5 temporal biomechanics Golden runner."""

from __future__ import annotations

import json
import math
from pathlib import Path

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import FrameGeometry, Joint
from slopecoach_ml.temporal import (
    StabilizedPoseSample,
    TemporalJoint2D,
    TemporalPoseQuality,
    TemporalProvenance,
)
from slopecoach_ml.turns import (
    TurnPhaseSign,
    TurnSegment,
    TurnSegmentStatus,
    TurnSignalSample,
    ValidSignalRun,
)

from .pipeline import analyze_temporal_biomechanics


def _sample(item, geometry):
    scale, (tx, ty), angle = item["scale"], item["translate"], math.radians(item["knee_angle_deg"])
    # Shoulder/ankle centers are 100*scale apart; bilateral separations are 20*scale.
    base = {
        Joint.LEFT_SHOULDER: (40, 0),
        Joint.RIGHT_SHOULDER: (60, 0),
        Joint.LEFT_HIP: (-10, 50),
        Joint.RIGHT_HIP: (10, 50),
        Joint.LEFT_KNEE: (-10, 100),
        Joint.RIGHT_KNEE: (10, 100),
    }
    # Hip vector from knee points upward. Rotating clockwise by angle gives each ankle.
    dx, dy = 50 * math.sin(angle), -50 * math.cos(angle)
    base[Joint.LEFT_ANKLE] = (-10 + dx, 100 + dy)
    base[Joint.RIGHT_ANKLE] = (10 + dx, 100 + dy)
    joints = {}
    for joint in Joint:
        x, y = base.get(joint, (0, 0))
        x, y = tx + scale * x, ty + scale * y
        joints[joint] = TemporalJoint2D(
            joint, x, y, x, y, x, y, 0.9, TemporalProvenance.OBSERVED, True
        )
    return StabilizedPoseSample(
        item["timestamp_us"],
        item["timestamp_us"] // 100000,
        item["temporal_segment_id"],
        geometry,
        "golden-target",
        1,
        TargetIdentityState.LOCKED,
        joints,
        17,
        0,
        0,
        TemporalPoseQuality.GOOD,
    )


def run_biomechanics_golden(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    geometry = FrameGeometry.from_dict(data["frame_geometry"])
    samples = [_sample(item, geometry) for item in data["samples"]]
    turn_data = data["turn"]
    turn = TurnSegment(
        turn_data["turn_id"],
        1,
        1,
        turn_data["start_timestamp_us"],
        turn_data["apex_timestamp_us"],
        turn_data["end_timestamp_us"],
        TurnPhaseSign(turn_data["phase_sign"]),
        turn_data["peak_value"],
        turn_data["prominence"],
        turn_data["end_timestamp_us"] - turn_data["start_timestamp_us"],
        3,
        0,
        0.0,
        1.0,
        TurnSegmentStatus.VALID,
    )
    signal = tuple(
        (index, TurnSignalSample(sample.timestamp_us, 1, value, 0.9, "OBSERVED"))
        for index, (sample, value) in enumerate(zip(samples[:3], (-0.2, 0.8, -0.2), strict=True))
    )
    result = analyze_temporal_biomechanics(samples, [turn], [ValidSignalRun(1, 1, signal)])
    frame = {(fact.timestamp_us, fact.feature_id): fact.value for fact in result.frame_facts}
    aggregates = {
        (item.temporal_segment_id, item.feature_id): item
        for item in result.temporal_segment_features
    }
    turn_facts = {fact.feature_id: fact.value for fact in result.turn_features[0].facts}
    expected = data["expected"]
    actual = {
        "frame_1_knee_angle_deg": frame[(0, "left_knee_angle_2d_deg")],
        "frame_1_bilateral_difference_deg": frame[(0, "bilateral_knee_abs_difference_2d_deg")],
        "frame_1_ankle_separation_body_scale": frame[(0, "ankle_separation_body_scale")],
        "segment_knee_median_deg": aggregates[(1, "bilateral_knee_mean_angle_2d_deg")].median,
        "knee_abs_velocity_median_deg_per_s": aggregates[
            (1, "bilateral_knee_mean_angle_abs_velocity_median_deg_per_s")
        ].median,
        "start_to_apex_delta_deg": turn_facts["knee_angle_change_start_to_apex_deg"],
        "apex_to_end_delta_deg": turn_facts["knee_angle_change_apex_to_end_deg"],
        "minimum_angle_offset_from_apex_us": turn_facts[
            "minimum_mean_knee_angle_offset_from_apex_us"
        ],
    }
    passed = all(math.isclose(actual[key], value, abs_tol=1e-8) for key, value in expected.items())
    serialized = result.to_dict()
    json.dumps(serialized, allow_nan=False, sort_keys=True)
    return {
        "golden_passed": passed,
        "fixture_contract_version": data["contract_version"],
        "expected": expected,
        "actual": actual,
        "result": serialized,
    }
