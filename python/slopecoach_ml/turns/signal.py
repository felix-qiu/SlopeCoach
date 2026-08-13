"""Signed, dimensionless lower-body image-space proxy."""

from __future__ import annotations

import math

from slopecoach_ml.pose import Joint
from slopecoach_ml.temporal import StabilizedPoseSample, TemporalProvenance

from .contracts import TurnSignalSample

REQUIRED_TURN_JOINTS = (
    Joint.LEFT_SHOULDER,
    Joint.RIGHT_SHOULDER,
    Joint.LEFT_HIP,
    Joint.RIGHT_HIP,
    Joint.LEFT_KNEE,
    Joint.RIGHT_KNEE,
    Joint.LEFT_ANKLE,
    Joint.RIGHT_ANKLE,
)


def signed_lateral_body_proxy(
    sample: StabilizedPoseSample, *, minimum_confidence: float = 0.30
) -> TurnSignalSample:
    """Cross(lower-body displacement, torso axis), normalized by both lengths."""
    if sample.temporal_segment_id is None:
        return TurnSignalSample(sample.timestamp_us, None, None, None, "MISSING")
    points = {}
    confidences = []
    provenances = []
    for joint in REQUIRED_TURN_JOINTS:
        point = sample.joint(joint)
        if point is None or point.stabilized_x_px is None or point.stabilized_y_px is None:
            return TurnSignalSample(
                sample.timestamp_us, sample.temporal_segment_id, None, None, "MISSING"
            )
        if point.support_confidence is None:
            return TurnSignalSample(
                sample.timestamp_us, sample.temporal_segment_id, None, None, "MISSING"
            )
        if point.support_confidence < minimum_confidence:
            return TurnSignalSample(
                sample.timestamp_us, sample.temporal_segment_id, None, None, "MISSING"
            )
        points[joint] = (point.stabilized_x_px, point.stabilized_y_px)
        confidences.append(point.support_confidence)
        provenances.append(point.provenance)

    def center(left, right):
        return (
            (points[left][0] + points[right][0]) / 2,
            (points[left][1] + points[right][1]) / 2,
        )

    shoulder = center(Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)
    hip = center(Joint.LEFT_HIP, Joint.RIGHT_HIP)
    knee = center(Joint.LEFT_KNEE, Joint.RIGHT_KNEE)
    ankle = center(Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)
    lower = ((knee[0] + ankle[0]) / 2 - hip[0], (knee[1] + ankle[1]) / 2 - hip[1])
    torso = (hip[0] - shoulder[0], hip[1] - shoulder[1])
    scale = math.hypot(*lower) * math.hypot(*torso)
    if not math.isfinite(scale) or scale <= 1e-9:
        return TurnSignalSample(
            sample.timestamp_us, sample.temporal_segment_id, None, None, "MISSING"
        )
    value = (lower[0] * torso[1] - lower[1] * torso[0]) / scale
    if not math.isfinite(value):
        raise ValueError("turn proxy produced non-finite output")
    provenance = (
        "INTERPOLATED_SUPPORT"
        if TemporalProvenance.INTERPOLATED in provenances
        else "STABILIZED_OBSERVED_SUPPORT"
    )
    return TurnSignalSample(
        sample.timestamp_us,
        sample.temporal_segment_id,
        max(-1.0, min(1.0, value)),
        min(confidences),
        provenance,
    )


def build_turn_signal(
    samples: list[StabilizedPoseSample], *, minimum_confidence: float = 0.30
) -> list[TurnSignalSample]:
    return [
        signed_lateral_body_proxy(sample, minimum_confidence=minimum_confidence)
        for sample in samples
    ]
