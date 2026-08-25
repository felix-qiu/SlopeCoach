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
from .debug import TURN_DEBUG_CONTRACT_VERSION, TurnDebugTrace, build_turn_debug_trace
from .fusion import (
    TurnDetectionResult,
    TurnEvidenceFusionConfig,
    TurnEvidenceSample,
    detect_turns_with_evidence_fusion,
    detect_turns_with_reference_pipeline,
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
    "TURN_DEBUG_CONTRACT_VERSION",
    "TurnDebugTrace",
    "TurnDetectionResult",
    "TurnEvidenceFusionConfig",
    "TurnEvidenceSample",
    "TurnSegment",
    "TurnSegmentationConfig",
    "TurnSegmentStatus",
    "TurnSignalSample",
    "ValidSignalRun",
    "ZeroCrossing",
    "build_turn_signal",
    "build_turn_debug_trace",
    "classify_real_turn_status",
    "detect_turns_with_evidence_fusion",
    "detect_turns_with_reference_pipeline",
    "detect_zero_crossings",
    "no_qualified_candidate_reason",
    "segmentation_summary",
    "segment_turns",
    "signal_sufficiency_diagnostics",
    "signed_lateral_body_proxy",
    "valid_signal_runs",
]
