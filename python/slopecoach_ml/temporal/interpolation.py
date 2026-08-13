"""Timestamp-weighted short-gap interpolation inside identity-safe segments."""

from __future__ import annotations

from dataclasses import dataclass

from slopecoach_ml.pose import Joint, Keypoint2D

from .contracts import TargetPoseSample, TemporalPoseConfig, TemporalProvenance


@dataclass(frozen=True)
class JointSupport:
    raw: Keypoint2D | None
    x_px: float | None
    y_px: float | None
    confidence: float | None
    provenance: TemporalProvenance


def interpolate_segment(
    samples: list[TargetPoseSample], config: TemporalPoseConfig
) -> tuple[list[dict[Joint, JointSupport]], int, int]:
    """Fill only bounded joint gaps; confidence is min(endpoint confidence)."""
    supports: list[dict[Joint, JointSupport]] = []
    for sample in samples:
        joint_map = {}
        for joint in Joint:
            raw = sample.raw_target_pose.joint(joint) if sample.raw_target_pose else None
            usable = raw is not None and raw.confidence >= config.minimum_joint_confidence
            joint_map[joint] = JointSupport(
                raw,
                raw.x_px if usable else None,
                raw.y_px if usable else None,
                raw.confidence if usable else None,
                TemporalProvenance.OBSERVED if usable else TemporalProvenance.MISSING,
            )
        supports.append(joint_map)
    filled = long_unfilled = 0
    for joint in Joint:
        valid = [index for index, item in enumerate(supports) if item[joint].x_px is not None]
        for left, right in zip(valid, valid[1:], strict=False):
            if right == left + 1:
                continue
            t0, t1 = samples[left].timestamp_us, samples[right].timestamp_us
            gap = t1 - t0
            if gap <= 0:
                raise ValueError("interpolation timestamps must strictly increase")
            if gap > config.maximum_interpolation_gap_us:
                long_unfilled += right - left - 1
                continue
            a, b = supports[left][joint], supports[right][joint]
            for index in range(left + 1, right):
                alpha = (samples[index].timestamp_us - t0) / gap
                supports[index][joint] = JointSupport(
                    supports[index][joint].raw,
                    a.x_px + alpha * (b.x_px - a.x_px),
                    a.y_px + alpha * (b.y_px - a.y_px),
                    min(a.confidence, b.confidence),
                    TemporalProvenance.INTERPOLATED,
                )
                filled += 1
    return supports, filled, long_unfilled
