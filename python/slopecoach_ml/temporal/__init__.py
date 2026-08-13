"""A4 temporal pose research/reference implementation."""

from .contracts import (
    StabilizedPoseSample,
    TargetPoseSample,
    TemporalJoint2D,
    TemporalPoseConfig,
    TemporalPoseQuality,
    TemporalPoseRun,
    TemporalProvenance,
)
from .golden import run_temporal_golden, run_turn_golden
from .interpolation import JointSupport, interpolate_segment
from .one_euro import LowPassFilter, OneEuroFilter1D
from .stabilizer import (
    segment_body_scales,
    stabilize_target_pose_stream,
    symmetric_frame_body_scale,
    temporal_stability_metrics,
)

__all__ = [
    "JointSupport",
    "LowPassFilter",
    "OneEuroFilter1D",
    "StabilizedPoseSample",
    "TargetPoseSample",
    "TemporalJoint2D",
    "TemporalPoseConfig",
    "TemporalPoseQuality",
    "TemporalPoseRun",
    "TemporalProvenance",
    "interpolate_segment",
    "run_temporal_golden",
    "run_turn_golden",
    "segment_body_scales",
    "stabilize_target_pose_stream",
    "symmetric_frame_body_scale",
    "temporal_stability_metrics",
]
