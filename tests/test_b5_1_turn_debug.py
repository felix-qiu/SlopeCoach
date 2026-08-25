from __future__ import annotations

import copy
import json
from dataclasses import dataclass

from slopecoach_ml.cli.__main__ import build_parser
from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.turns import (
    ReferencePeakDetector,
    TurnSegmentationConfig,
    TurnSignalSample,
    build_turn_debug_trace,
    detect_zero_crossings,
    segment_turns,
)


@dataclass(frozen=True)
class _TemporalSample:
    timestamp_us: int
    temporal_segment_id: int | None = 1
    identity_state: TargetIdentityState = TargetIdentityState.LOCKED


def _pipeline(values, *, step_us=500_000):
    config = TurnSegmentationConfig()
    signal = [
        TurnSignalSample(
            timestamp_us=index * step_us,
            temporal_segment_id=1,
            value=value,
            support_confidence=0.9,
            provenance="STABILIZED_OBSERVED_SUPPORT",
        )
        for index, value in enumerate(values)
    ]
    temporal = [_TemporalSample(item.timestamp_us) for item in signal]
    peaks = ReferencePeakDetector().detect(signal, config)
    crossings = detect_zero_crossings(
        signal,
        config.zero_crossing_tolerance,
        minimum_signal_confidence=config.minimum_signal_confidence,
    )
    segments = segment_turns(signal, peaks, crossings, config)
    trace = build_turn_debug_trace(temporal, signal, peaks, segments, crossings, config)
    return trace, peaks, segments


def test_no_turn_trace_explains_maximum_score_below_threshold() -> None:
    trace, peaks, segments = _pipeline([0.0, 0.02, 0.0, -0.02, 0.0])
    payload = trace.to_dict()
    summary = payload["turn_debug_summary"]
    assert peaks == []
    assert segments == []
    assert summary["candidate_generated_count"] == 0
    assert summary["max_turn_score"] == 0.02
    assert summary["threshold"] == 0.08
    assert summary["failure_reason"] == "MAX_SCORE_BELOW_THRESHOLD"
    assert summary["raw_local_extremum_count"] == 2
    assert summary["prominence_eligible_local_extremum_count"] == 0
    assert all(not item["candidate_generated"] for item in payload["samples"])


def test_turn_trace_reports_generated_candidates_without_changing_detector() -> None:
    trace, peaks, segments = _pipeline([-0.2, 0.0, 0.4, 0.0, -0.4, 0.0])
    payload = trace.to_dict()
    assert len(peaks) > 0
    assert len(segments) == len(peaks)
    assert payload["turn_debug_summary"]["candidate_generated_count"] == len(peaks)
    assert sum(item["candidate_generated"] for item in payload["samples"]) == len(peaks)
    assert any(
        item["candidate_state"] == "QUALIFIED_PEAK" for item in payload["samples"]
    )
    assert payload["turn_debug_summary"]["max_local_extremum_prominence"] >= 0.08


def test_turn_debug_sha_is_deterministic_and_json_safe() -> None:
    first, _, _ = _pipeline([-0.2, 0.0, 0.4, 0.0, -0.4, 0.0])
    second, _, _ = _pipeline(copy.deepcopy([-0.2, 0.0, 0.4, 0.0, -0.4, 0.0]))
    assert first.turn_debug_sha256 == second.turn_debug_sha256
    assert first.to_dict() == second.to_dict()
    json.dumps(first.to_dict(), sort_keys=True, allow_nan=False)


def test_analyze_video_parser_accepts_optional_turn_debug_output() -> None:
    args = build_parser().parse_args(
        [
            "analyze-video",
            "ski.mp4",
            "--sport-type",
            "SKI",
            "--turn-debug-output",
            "turn-debug.json",
        ]
    )
    assert args.turn_debug_output == "turn-debug.json"
