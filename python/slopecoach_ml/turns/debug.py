"""Deterministic research-only observability for turn detection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from slopecoach_ml.identity import TargetIdentityState

if TYPE_CHECKING:
    from slopecoach_ml.temporal import StabilizedPoseSample

from .contracts import PeakCandidate, TurnSegment, TurnSegmentationConfig, TurnSignalSample
from .fusion import TurnEvidenceFusionConfig, TurnEvidenceSample
from .runs import (
    classify_real_turn_status,
    no_qualified_candidate_reason,
    signal_sufficiency_diagnostics,
    valid_signal_runs,
)

TURN_DEBUG_CONTRACT_VERSION = "turn-debug-v2"


@dataclass(frozen=True)
class TurnDebugTrace:
    """Debug projection of already-computed turn inputs and outputs."""

    samples: tuple[dict[str, object], ...]
    summary: dict[str, object]
    contract_version: str = TURN_DEBUG_CONTRACT_VERSION

    @property
    def turn_debug_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._semantic_dict()).encode("utf-8")).hexdigest()

    def _semantic_dict(self) -> dict[str, object]:
        fusion_enabled = bool(self.summary.get("fusion_enabled"))
        if fusion_enabled:
            detector_semantics = {
                "implementation": "TurnEvidenceFusionV2",
                "underlying_peak_detector": "ReferencePeakDetector",
                "turn_score_definition": "ABS_FUSED_TURN_EVIDENCE_SCORE",
                "primary_signal": "turn_evidence_fusion_v2",
                "baseline_signal": "signed_lateral_body_proxy",
                "fusion_formula": "weighted_average(lateral, orientation, trajectory, lower_body)",
            }
        else:
            detector_semantics = {
                "implementation": "ReferencePeakDetector",
                "underlying_peak_detector": "ReferencePeakDetector",
                "turn_score_definition": "ABS_SIGNED_LATERAL_BODY_PROXY",
                "primary_signal": "signed_lateral_body_proxy",
                "velocity_proxy_role": "DIAGNOSTIC_ONLY_NOT_USED_BY_DETECTOR",
                "body_inclination_proxy_role": "NOT_USED_BY_CURRENT_TURN_DETECTOR",
                "shoulder_hip_angle_proxy_role": "NOT_USED_BY_CURRENT_TURN_DETECTOR",
            }
        return {
            "contract_version": self.contract_version,
            "detector_semantics": detector_semantics,
            "turn_debug_summary": self.summary,
            "samples": list(self.samples),
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._semantic_dict()
        payload["turn_debug_sha256"] = self.turn_debug_sha256
        _canonical_json(payload)
        return payload


def build_turn_debug_trace(
    temporal_samples: list[StabilizedPoseSample],
    signal: list[TurnSignalSample],
    peaks: list[PeakCandidate],
    segments: list[TurnSegment],
    crossings,
    config: TurnSegmentationConfig,
    *,
    evidence_samples: list[TurnEvidenceSample] | tuple[TurnEvidenceSample, ...] | None = None,
    raw_peaks: list[PeakCandidate] | tuple[PeakCandidate, ...] | None = None,
    rejected_peaks: list[PeakCandidate] | tuple[PeakCandidate, ...] | None = None,
    fusion_summary: dict[str, object] | None = None,
    fusion_config: TurnEvidenceFusionConfig | None = None,
) -> TurnDebugTrace:
    """Project existing detector evidence without rerunning or influencing detection."""

    config.validate()
    if len(temporal_samples) != len(signal):
        raise ValueError("TURN_DEBUG_SAMPLE_ALIGNMENT_MISMATCH")
    for temporal, signal_sample in zip(temporal_samples, signal, strict=True):
        if temporal.timestamp_us != signal_sample.timestamp_us:
            raise ValueError("TURN_DEBUG_TIMESTAMP_ALIGNMENT_MISMATCH")

    evidence_sequence = tuple(evidence_samples or ())
    raw_peak_sequence = tuple(raw_peaks or peaks)
    rejected_peak_sequence = tuple(rejected_peaks or ())
    if evidence_sequence and len(evidence_sequence) != len(signal):
        raise ValueError("TURN_DEBUG_EVIDENCE_ALIGNMENT_MISMATCH")

    diagnostics = signal_sufficiency_diagnostics(signal, peaks, crossings, config)
    status = classify_real_turn_status(diagnostics, segments)
    existing_reason = no_qualified_candidate_reason(status, diagnostics, config)
    trusted_indices = [
        index
        for index, sample in enumerate(temporal_samples)
        if sample.identity_state is TargetIdentityState.LOCKED
        and sample.temporal_segment_id is not None
    ]
    peak_by_index = {peak.sample_index: peak for peak in peaks}
    raw_peak_by_index = {peak.sample_index: peak for peak in raw_peak_sequence}
    rejected_peak_by_index = {peak.sample_index: peak for peak in rejected_peak_sequence}
    evidence_by_index = {index: item for index, item in enumerate(evidence_sequence)}
    segment_by_peak = {segment.apex_timestamp_us: segment for segment in segments}
    raw_extrema = _raw_local_extrema(signal, config)
    trace = []
    previous_signal = None
    previous_state = None
    for index in trusted_indices:
        temporal = temporal_samples[index]
        item = signal[index]
        peak = peak_by_index.get(index)
        raw_peak = raw_peak_by_index.get(index)
        rejected_peak = rejected_peak_by_index.get(index)
        evidence = evidence_by_index.get(index)
        raw_extremum = raw_extrema.get(index)
        segment = segment_by_peak.get(item.timestamp_us) if peak is not None else None
        velocity = _velocity_proxy(previous_signal, item)
        state = _candidate_state(item, peak, raw_peak, rejected_peak, raw_extremum, config)
        lateral_score = evidence.lateral_score if evidence is not None else item.value
        orientation_score = evidence.orientation_score if evidence is not None else None
        trajectory_score = evidence.trajectory_score if evidence is not None else None
        lower_body_score = evidence.lower_body_score if evidence is not None else None
        fused_score = evidence.fused_turn_evidence_score if evidence is not None else item.value
        trace.append(
            {
                "temporal_sample_index": index,
                "timestamp_us": item.timestamp_us,
                "temporal_segment_id": item.temporal_segment_id,
                "trusted_identity_status": {
                    "trusted": True,
                    "identity_state": temporal.identity_state.value,
                },
                "lateral_score": lateral_score,
                "orientation_score": orientation_score,
                "trajectory_score": trajectory_score,
                "lower_body_score": lower_body_score,
                "fused_turn_evidence_score": fused_score,
                "detector_state": state,
                "signals": {
                    "active_turn_signal": item.value,
                    "lateral_movement_proxy": lateral_score,
                    "support_confidence": item.support_confidence,
                    "signal_provenance": item.provenance,
                    "velocity_proxy_per_second": velocity,
                    "velocity_proxy_role": "DIAGNOSTIC_ONLY_NOT_USED_BY_DETECTOR",
                    "body_orientation_change_proxy": orientation_score,
                    "trajectory_direction_change_proxy": trajectory_score,
                    "lower_body_temporal_change_proxy": lower_body_score,
                },
                "turn_score": abs(fused_score) if fused_score is not None else None,
                "turn_score_threshold": config.minimum_peak_amplitude,
                "peak_prominence": raw_peak.prominence if raw_peak is not None else None,
                "peak_prominence_threshold": config.minimum_peak_prominence,
                "raw_local_extremum": raw_extremum,
                "candidate_state": state,
                "candidate_state_transition": {
                    "previous": previous_state,
                    "current": state,
                    "changed": previous_state is not None and previous_state != state,
                },
                "candidate_generated": raw_peak is not None,
                "candidate_accepted": peak is not None,
                "candidate_rejected": rejected_peak is not None,
                "segment_status": segment.status.value if segment is not None else None,
            }
        )
        previous_state = state
        previous_signal = item if item.value is not None else None

    scores = [abs(float(sample.value)) for sample in signal if sample.value is not None]
    qualified = sum(segment.status.value in {"VALID", "PARTIAL"} for segment in segments)
    local_prominences = [float(item["prominence"]) for item in raw_extrema.values()]
    fused_values = [
        evidence.fused_turn_evidence_score
        for evidence in evidence_sequence
        if evidence.fused_turn_evidence_score is not None
    ] or [sample.value for sample in signal if sample.value is not None]
    summary = {
        "trusted_samples": len(trusted_indices),
        "valid_signal_samples": diagnostics["valid_signal_sample_count"],
        "valid_signal_run_count": diagnostics["valid_signal_run_count"],
        "longest_valid_signal_run_sample_count": diagnostics[
            "longest_valid_signal_run_sample_count"
        ],
        "candidate_generated_count": len(raw_peak_sequence),
        "candidate_count": len(raw_peak_sequence),
        "qualified_turn_segment_count": qualified,
        "rejected_candidate_count": max(len(raw_peak_sequence) - qualified, 0),
        "rejected_count": len(rejected_peak_sequence),
        "max_turn_score": max(scores) if scores else None,
        "max_fused_score": max(fused_values) if fused_values else None,
        "min_fused_score": min(fused_values) if fused_values else None,
        "fused_score_range": {
            "min": min(fused_values) if fused_values else None,
            "max": max(fused_values) if fused_values else None,
        },
        "fused_score_sha256": _fused_score_sha256(signal, evidence_sequence),
        "raw_local_extremum_count": len(raw_extrema),
        "amplitude_eligible_local_extremum_count": sum(
            bool(item["amplitude_threshold_passed"]) for item in raw_extrema.values()
        ),
        "prominence_eligible_local_extremum_count": sum(
            bool(item["prominence_threshold_passed"]) for item in raw_extrema.values()
        ),
        "max_local_extremum_prominence": max(local_prominences) if local_prominences else None,
        "threshold": config.minimum_peak_amplitude,
        "thresholds": {
            "minimum_signal_confidence": config.minimum_signal_confidence,
            "minimum_peak_amplitude": config.minimum_peak_amplitude,
            "minimum_peak_prominence": config.minimum_peak_prominence,
            "minimum_peak_separation_us": config.minimum_peak_separation_us,
            "minimum_turn_duration_us": config.minimum_turn_duration_us,
            "maximum_turn_duration_us": config.maximum_turn_duration_us,
            "minimum_valid_samples_per_turn": config.minimum_valid_samples_per_turn,
            "maximum_missing_ratio": config.maximum_missing_ratio,
            "turn_evidence_fusion": (
                asdict(fusion_config) if fusion_config is not None else {"enabled": False}
            ),
        },
        "fusion_enabled": bool(fusion_config is not None and fusion_config.enabled),
        "dominant_evidence_source": (
            fusion_summary.get("dominant_evidence_source") if fusion_summary else "lateral_score"
        ),
        "turn_evidence_fusion_summary": fusion_summary
        or {
            "enabled": False,
            "candidate_count": len(raw_peak_sequence),
            "rejected_count": len(rejected_peak_sequence),
            "dominant_evidence_source": "lateral_score",
        },
        "real_turn_segmentation_status": status.value,
        "existing_no_qualified_candidate_reason": existing_reason,
        "failure_evidence": {
            "maximum_score_below_amplitude_threshold": bool(
                scores and max(scores) < config.minimum_peak_amplitude
            ),
            "maximum_local_extremum_prominence_below_threshold": bool(
                local_prominences and max(local_prominences) < config.minimum_peak_prominence
            ),
        },
        "failure_reason": _failure_reason(
            trusted_sample_count=len(trusted_indices),
            diagnostics=diagnostics,
            peaks=peaks,
            segments=segments,
            max_score=max(scores) if scores else None,
            existing_reason=existing_reason,
            config=config,
        ),
    }
    return TurnDebugTrace(tuple(trace), summary)


def _candidate_state(
    sample: TurnSignalSample,
    peak: PeakCandidate | None,
    raw_peak: PeakCandidate | None,
    rejected_peak: PeakCandidate | None,
    raw_extremum: dict[str, object] | None,
    config: TurnSegmentationConfig,
) -> str:
    if sample.value is None or sample.support_confidence is None:
        return "MISSING_SIGNAL"
    if sample.support_confidence < config.minimum_signal_confidence:
        return "BELOW_SIGNAL_CONFIDENCE"
    if peak is not None:
        return "QUALIFIED_PEAK"
    if rejected_peak is not None:
        return "PEAK_REJECTED_TRANSITION_CONTEXT"
    if raw_peak is not None:
        return "RAW_PEAK_WITHOUT_QUALIFIED_SEGMENT"
    if raw_extremum is not None and not raw_extremum["amplitude_threshold_passed"]:
        return "LOCAL_EXTREMUM_BELOW_PEAK_AMPLITUDE"
    if raw_extremum is not None and not raw_extremum["prominence_threshold_passed"]:
        return "LOCAL_EXTREMUM_BELOW_PEAK_PROMINENCE"
    if abs(sample.value) < config.minimum_peak_amplitude:
        return "BELOW_PEAK_AMPLITUDE"
    return "VALID_SIGNAL_NO_QUALIFIED_EXTREMUM"


def _raw_local_extrema(
    signal: list[TurnSignalSample],
    config: TurnSegmentationConfig,
) -> dict[int, dict[str, object]]:
    """Mirror only the detector's pre-threshold local-extremum observation step."""

    extrema = {}
    for run in valid_signal_runs(signal, config):
        samples = run.indexed_samples
        position = 1
        while position < len(samples) - 1:
            plateau_end = position
            while (
                plateau_end + 1 < len(samples)
                and samples[plateau_end + 1][1].value == samples[position][1].value
            ):
                plateau_end += 1
            index, item = samples[position]
            left = samples[position - 1][1]
            right = samples[plateau_end + 1][1] if plateau_end + 1 < len(samples) else item
            positive = item.value > left.value and item.value > right.value
            negative = item.value < left.value and item.value < right.value
            if positive or negative:
                prominence = min(abs(item.value - left.value), abs(item.value - right.value))
                extrema[index] = {
                    "phase_sign": "POSITIVE_PHASE" if positive else "NEGATIVE_PHASE",
                    "prominence": prominence,
                    "absolute_amplitude": abs(item.value),
                    "amplitude_threshold_passed": abs(item.value) >= config.minimum_peak_amplitude,
                    "prominence_threshold_passed": prominence >= config.minimum_peak_prominence,
                }
            position = plateau_end + 1
    return extrema


def _velocity_proxy(previous: TurnSignalSample | None, current: TurnSignalSample) -> float | None:
    if (
        previous is None
        or previous.value is None
        or current.value is None
        or previous.temporal_segment_id != current.temporal_segment_id
    ):
        return None
    delta_us = current.timestamp_us - previous.timestamp_us
    if delta_us <= 0:
        return None
    velocity = (current.value - previous.value) / (delta_us / 1_000_000)
    return velocity if math.isfinite(velocity) else None


def _failure_reason(
    *,
    trusted_sample_count: int,
    diagnostics: dict[str, object],
    peaks: list[PeakCandidate],
    segments: list[TurnSegment],
    max_score: float | None,
    existing_reason: str | None,
    config: TurnSegmentationConfig,
) -> str | None:
    if trusted_sample_count == 0:
        return "NO_TRUSTED_TEMPORAL_SAMPLES"
    if diagnostics["valid_signal_sample_count"] == 0:
        return "NO_VALID_TURN_SIGNAL"
    if diagnostics["longest_valid_signal_run_sample_count"] < 3:
        return "INSUFFICIENT_WINDOW"
    if not peaks:
        if max_score is not None and max_score < config.minimum_peak_amplitude:
            return "MAX_SCORE_BELOW_THRESHOLD"
        return existing_reason or "NO_QUALIFIED_EXTREMA"
    if segments and all(segment.status.value.startswith("REJECTED_") for segment in segments):
        return "CANDIDATE_REJECTED_BY_SEGMENT_VALIDATION"
    return None


def _fused_score_sha256(
    signal: list[TurnSignalSample],
    evidence_sequence: tuple[TurnEvidenceSample, ...],
) -> str:
    if evidence_sequence:
        payload = [
            {
                "timestamp_us": item.timestamp_us,
                "fused_turn_evidence_score": item.fused_turn_evidence_score,
            }
            for item in evidence_sequence
        ]
    else:
        payload = [
            {
                "timestamp_us": item.timestamp_us,
                "fused_turn_evidence_score": item.value,
            }
            for item in signal
        ]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
