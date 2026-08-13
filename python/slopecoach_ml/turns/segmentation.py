"""Timestamp-based zero crossings and provisional image-space turn segments."""

from __future__ import annotations

from collections import Counter

from .contracts import (
    PeakCandidate,
    TurnSegment,
    TurnSegmentationConfig,
    TurnSegmentStatus,
    TurnSignalSample,
    ZeroCrossing,
)
from .runs import valid_signal_runs


def detect_zero_crossings(
    samples: list[TurnSignalSample],
    tolerance: float = 1e-9,
    *,
    minimum_signal_confidence: float = 0.0,
) -> list[ZeroCrossing]:
    settings = TurnSegmentationConfig(
        minimum_signal_confidence=minimum_signal_confidence,
        zero_crossing_tolerance=tolerance,
    )
    crossings: list[ZeroCrossing] = []
    for run in valid_signal_runs(samples, settings):
        previous_nonzero = None
        first_zero_timestamp = None
        last_zero_timestamp = None
        for _, sample in run.indexed_samples:
            sign = 0 if abs(sample.value) <= tolerance else (1 if sample.value > 0 else -1)
            if sign == 0:
                if previous_nonzero is not None:
                    first_zero_timestamp = (
                        sample.timestamp_us
                        if first_zero_timestamp is None
                        else first_zero_timestamp
                    )
                    last_zero_timestamp = sample.timestamp_us
                continue
            if previous_nonzero is not None and sign != previous_nonzero[0]:
                if first_zero_timestamp is not None:
                    timestamp = (first_zero_timestamp + last_zero_timestamp) // 2
                else:
                    prior_value = previous_nonzero[2]
                    alpha = abs(prior_value) / (abs(prior_value) + abs(sample.value))
                    timestamp = round(
                        previous_nonzero[1] + alpha * (sample.timestamp_us - previous_nonzero[1])
                    )
                crossings.append(
                    ZeroCrossing(
                        timestamp_us=timestamp,
                        temporal_segment_id=run.temporal_segment_id,
                        signal_run_id=run.signal_run_id,
                        direction=("NEGATIVE_TO_POSITIVE" if sign > 0 else "POSITIVE_TO_NEGATIVE"),
                    )
                )
            previous_nonzero = (sign, sample.timestamp_us, sample.value)
            first_zero_timestamp = last_zero_timestamp = None
    return crossings


def segment_turns(
    signal: list[TurnSignalSample],
    peaks: list[PeakCandidate],
    crossings: list[ZeroCrossing],
    config: TurnSegmentationConfig,
) -> list[TurnSegment]:
    config.validate()
    runs = {run.signal_run_id: run for run in valid_signal_runs(signal, config)}
    results = []
    for number, peak in enumerate(
        sorted(peaks, key=lambda candidate: (candidate.timestamp_us, candidate.sample_index)), 1
    ):
        run = runs.get(peak.signal_run_id)
        if run is None:
            raise ValueError(f"peak references unknown signal run {peak.signal_run_id}")
        before = [
            crossing
            for crossing in crossings
            if crossing.signal_run_id == peak.signal_run_id
            and crossing.timestamp_us < peak.timestamp_us
        ]
        after = [
            crossing
            for crossing in crossings
            if crossing.signal_run_id == peak.signal_run_id
            and crossing.timestamp_us > peak.timestamp_us
        ]
        start = before[-1].timestamp_us if before else None
        end = after[0].timestamp_us if after else None
        duration = end - start if start is not None and end is not None else None
        window = [
            sample
            for _, sample in run.indexed_samples
            if sample.temporal_segment_id == peak.temporal_segment_id
            and (start is None or sample.timestamp_us >= start)
            and (end is None or sample.timestamp_us <= end)
        ]
        valid = sum(sample.value is not None for sample in window)
        interpolated = sum(sample.provenance == "INTERPOLATED_SUPPORT" for sample in window)
        missing = sum(sample.value is None for sample in window)
        missing_ratio = missing / len(window) if window else None
        if peak.prominence < config.minimum_peak_prominence:
            status = TurnSegmentStatus.REJECTED_LOW_PROMINENCE
        elif valid < config.minimum_valid_samples_per_turn or (
            missing_ratio is not None and missing_ratio > config.maximum_missing_ratio
        ):
            status = TurnSegmentStatus.REJECTED_LOW_COVERAGE
        elif duration is None:
            status = TurnSegmentStatus.PARTIAL
        elif duration < config.minimum_turn_duration_us:
            status = TurnSegmentStatus.REJECTED_SHORT
        elif duration > config.maximum_turn_duration_us:
            status = TurnSegmentStatus.REJECTED_LONG
        else:
            status = TurnSegmentStatus.VALID
        confidence = min(1.0, peak.prominence / max(config.minimum_peak_prominence, 1e-12))
        results.append(
            TurnSegment(
                f"turn-{number}",
                peak.temporal_segment_id,
                peak.signal_run_id,
                start,
                peak.timestamp_us,
                end,
                peak.phase_sign,
                peak.value,
                peak.prominence,
                duration,
                valid,
                interpolated,
                missing_ratio,
                confidence,
                status,
            )
        )
    return results


def segmentation_summary(segments: list[TurnSegment]) -> dict[str, object]:
    counts = Counter(segment.status.value for segment in segments)
    return {
        "provisional_turn_candidate_count": len(segments),
        "valid_segment_count": counts[TurnSegmentStatus.VALID.value],
        "partial_segment_count": counts[TurnSegmentStatus.PARTIAL.value],
        "rejection_reason_counts": {
            key: value for key, value in sorted(counts.items()) if key.startswith("REJECTED_")
        },
    }
