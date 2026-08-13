"""A4 provisional image-space turn research/reference implementation."""

from .contracts import (
    PeakCandidate,
    PeakDetector,
    RealTurnSegmentationStatus,
    TurnPhaseSign,
    TurnSegment,
    TurnSegmentationConfig,
    TurnSegmentStatus,
    TurnSignalSample,
    ZeroCrossing,
)
from .peaks import ReferencePeakDetector, SciPyFindPeaksDetector
from .runs import (
    ValidSignalRun,
    classify_real_turn_status,
    no_qualified_candidate_reason,
    signal_sufficiency_diagnostics,
    valid_signal_runs,
)
from .segmentation import detect_zero_crossings, segment_turns, segmentation_summary
from .signal import REQUIRED_TURN_JOINTS, build_turn_signal, signed_lateral_body_proxy

__all__ = [
    "PeakCandidate",
    "PeakDetector",
    "REQUIRED_TURN_JOINTS",
    "ReferencePeakDetector",
    "RealTurnSegmentationStatus",
    "SciPyFindPeaksDetector",
    "TurnPhaseSign",
    "TurnSegment",
    "TurnSegmentationConfig",
    "TurnSegmentStatus",
    "TurnSignalSample",
    "ValidSignalRun",
    "ZeroCrossing",
    "build_turn_signal",
    "classify_real_turn_status",
    "detect_zero_crossings",
    "no_qualified_candidate_reason",
    "segmentation_summary",
    "segment_turns",
    "signal_sufficiency_diagnostics",
    "signed_lateral_body_proxy",
    "valid_signal_runs",
]
