"""A4 provisional image-space turn research/reference implementation."""

from .contracts import (
    PeakCandidate,
    PeakDetector,
    TurnPhaseSign,
    TurnSegment,
    TurnSegmentationConfig,
    TurnSegmentStatus,
    TurnSignalSample,
    ZeroCrossing,
)
from .peaks import ReferencePeakDetector, SciPyFindPeaksDetector
from .segmentation import detect_zero_crossings, segment_turns, segmentation_summary
from .signal import REQUIRED_TURN_JOINTS, build_turn_signal, signed_lateral_body_proxy

__all__ = [
    "PeakCandidate",
    "PeakDetector",
    "REQUIRED_TURN_JOINTS",
    "ReferencePeakDetector",
    "SciPyFindPeaksDetector",
    "TurnPhaseSign",
    "TurnSegment",
    "TurnSegmentationConfig",
    "TurnSegmentStatus",
    "TurnSignalSample",
    "ZeroCrossing",
    "build_turn_signal",
    "detect_zero_crossings",
    "segmentation_summary",
    "segment_turns",
    "signed_lateral_body_proxy",
]
