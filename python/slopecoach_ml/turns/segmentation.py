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


def detect_zero_crossings(
    samples: list[TurnSignalSample], tolerance: float = 1e-9
) -> list[ZeroCrossing]:
    crossings = []
    previous = None
    seen = set()
    for sample in samples:
        if sample.value is None or sample.temporal_segment_id is None:
            previous = None
            continue
        if previous is not None and previous.temporal_segment_id == sample.temporal_segment_id:
            a, b = previous.value, sample.value
            timestamp = direction = None
            if abs(a) <= tolerance:
                timestamp = previous.timestamp_us
                direction = "NEGATIVE_TO_POSITIVE" if b > 0 else "POSITIVE_TO_NEGATIVE"
            elif abs(b) <= tolerance:
                timestamp = sample.timestamp_us
                direction = "NEGATIVE_TO_POSITIVE" if a < 0 else "POSITIVE_TO_NEGATIVE"
            elif a * b < 0:
                alpha = abs(a) / (abs(a) + abs(b))
                timestamp = round(
                    previous.timestamp_us + alpha * (sample.timestamp_us - previous.timestamp_us)
                )
                direction = "NEGATIVE_TO_POSITIVE" if a < b else "POSITIVE_TO_NEGATIVE"
            key = (sample.temporal_segment_id, timestamp)
            if timestamp is not None and key not in seen:
                crossings.append(ZeroCrossing(timestamp, sample.temporal_segment_id, direction))
                seen.add(key)
        previous = sample
    return crossings


def segment_turns(
    signal: list[TurnSignalSample],
    peaks: list[PeakCandidate],
    crossings: list[ZeroCrossing],
    config: TurnSegmentationConfig,
) -> list[TurnSegment]:
    config.validate()
    results = []
    for number, peak in enumerate(peaks, 1):
        before = [
            crossing
            for crossing in crossings
            if crossing.temporal_segment_id == peak.temporal_segment_id
            and crossing.timestamp_us < peak.timestamp_us
        ]
        after = [
            crossing
            for crossing in crossings
            if crossing.temporal_segment_id == peak.temporal_segment_id
            and crossing.timestamp_us > peak.timestamp_us
        ]
        start = before[-1].timestamp_us if before else None
        end = after[0].timestamp_us if after else None
        duration = end - start if start is not None and end is not None else None
        window = [
            sample
            for sample in signal
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
