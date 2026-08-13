"""Valid signal-run boundaries and engineering-only sufficiency diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from .contracts import (
    PeakCandidate,
    RealTurnSegmentationStatus,
    TurnSegment,
    TurnSegmentationConfig,
    TurnSegmentStatus,
    TurnSignalSample,
    ZeroCrossing,
)


@dataclass(frozen=True)
class ValidSignalRun:
    """A maximal confidence-valid signal sequence; not a temporal pose segment."""

    signal_run_id: int
    temporal_segment_id: int
    indexed_samples: tuple[tuple[int, TurnSignalSample], ...]

    @property
    def start_timestamp_us(self) -> int:
        return self.indexed_samples[0][1].timestamp_us

    @property
    def end_timestamp_us(self) -> int:
        return self.indexed_samples[-1][1].timestamp_us

    @property
    def duration_us(self) -> int:
        return self.end_timestamp_us - self.start_timestamp_us

    @property
    def sample_count(self) -> int:
        return len(self.indexed_samples)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(float(sample.value) for _, sample in self.indexed_samples)


def valid_signal_runs(
    samples: list[TurnSignalSample], config: TurnSegmentationConfig
) -> list[ValidSignalRun]:
    """Split on every invalid sample or temporal-segment boundary."""
    config.validate()
    runs: list[ValidSignalRun] = []
    current: list[tuple[int, TurnSignalSample]] = []
    current_segment: int | None = None

    def finish() -> None:
        nonlocal current, current_segment
        if current:
            runs.append(ValidSignalRun(len(runs) + 1, current_segment, tuple(current)))
        current = []
        current_segment = None

    for index, sample in enumerate(samples):
        valid = (
            sample.temporal_segment_id is not None
            and sample.value is not None
            and math.isfinite(sample.value)
            and sample.support_confidence is not None
            and math.isfinite(sample.support_confidence)
            and sample.support_confidence >= config.minimum_signal_confidence
        )
        if not valid:
            finish()
            continue
        if current and sample.temporal_segment_id != current_segment:
            finish()
        if current and sample.timestamp_us <= current[-1][1].timestamp_us:
            raise ValueError("valid signal run timestamps must strictly increase")
        if not current:
            current_segment = sample.temporal_segment_id
        current.append((index, sample))
    finish()
    return runs


def signal_sufficiency_diagnostics(
    samples: list[TurnSignalSample],
    peaks: list[PeakCandidate],
    crossings: list[ZeroCrossing],
    config: TurnSegmentationConfig,
) -> dict[str, object]:
    runs = valid_signal_runs(samples, config)
    values = [value for run in runs for value in run.values]
    absolute_deltas = [
        abs(right - left)
        for run in runs
        for left, right in zip(run.values, run.values[1:], strict=False)
    ]
    longest = max(runs, key=lambda run: (run.sample_count, run.duration_us), default=None)
    positive = sum(peak.value > 0 for peak in peaks)
    negative = sum(peak.value < 0 for peak in peaks)
    value_min = min(values) if values else None
    value_max = max(values) if values else None
    span = value_max - value_min if value_min is not None and value_max is not None else None
    run_summaries = []
    for run in runs:
        run_peaks = [peak for peak in peaks if peak.signal_run_id == run.signal_run_id]
        run_crossings = [
            crossing for crossing in crossings if crossing.signal_run_id == run.signal_run_id
        ]
        run_min, run_max = min(run.values), max(run.values)
        run_summaries.append(
            {
                "signal_run_id": run.signal_run_id,
                "temporal_segment_id": run.temporal_segment_id,
                "start_timestamp_us": run.start_timestamp_us,
                "end_timestamp_us": run.end_timestamp_us,
                "sample_count": run.sample_count,
                "value_min": run_min,
                "value_max": run_max,
                "value_span": run_max - run_min,
                "peak_count": len(run_peaks),
                "zero_crossing_count": len(run_crossings),
            }
        )
    return {
        "valid_signal_sample_count": len(values),
        "missing_signal_sample_count": len(samples) - len(values),
        "valid_signal_run_count": len(runs),
        "longest_valid_signal_run_sample_count": longest.sample_count if longest else 0,
        "longest_valid_signal_run_duration_us": longest.duration_us if longest else 0,
        "signal_value_min": value_min,
        "signal_value_max": value_max,
        "signal_value_span": span,
        "median_absolute_signal_delta": median(absolute_deltas) if absolute_deltas else None,
        "qualified_positive_peak_count": positive,
        "qualified_negative_peak_count": negative,
        "qualified_peak_count": len(peaks),
        "zero_crossing_count": len(crossings),
        "signal_runs": run_summaries,
    }


def classify_real_turn_status(
    diagnostics: dict[str, object], segments: list[TurnSegment]
) -> RealTurnSegmentationStatus:
    if diagnostics["valid_signal_sample_count"] == 0:
        return RealTurnSegmentationStatus.NOT_ANALYZABLE_NO_VALID_TURN_SIGNAL
    if diagnostics["longest_valid_signal_run_sample_count"] < 3:
        return RealTurnSegmentationStatus.NOT_ANALYZABLE_INSUFFICIENT_CONTINUOUS_TARGET_POSE
    if diagnostics["qualified_peak_count"] == 0:
        return RealTurnSegmentationStatus.EXECUTED_NO_QUALIFIED_TURN_CANDIDATES
    if segments and all(segment.status.value.startswith("REJECTED_") for segment in segments):
        return RealTurnSegmentationStatus.EXECUTED_CANDIDATES_REJECTED
    if any(
        segment.status in (TurnSegmentStatus.VALID, TurnSegmentStatus.PARTIAL)
        for segment in segments
    ):
        return RealTurnSegmentationStatus.EXECUTED_PROVISIONAL_CANDIDATES
    return RealTurnSegmentationStatus.EXECUTED_CANDIDATES_REJECTED


def no_qualified_candidate_reason(
    status: RealTurnSegmentationStatus,
    diagnostics: dict[str, object],
    config: TurnSegmentationConfig,
) -> str | None:
    if status is not RealTurnSegmentationStatus.EXECUTED_NO_QUALIFIED_TURN_CANDIDATES:
        return None
    span = diagnostics["signal_value_span"]
    if span is not None and span < config.minimum_peak_amplitude:
        return "SIGNAL_VARIATION_BELOW_AMPLITUDE_THRESHOLD"
    return "NO_QUALIFIED_EXTREMA"
