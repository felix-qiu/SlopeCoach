"""Deterministic reference extrema detection plus optional lazy SciPy adapter."""

from __future__ import annotations

from .contracts import (
    PeakCandidate,
    TurnPhaseSign,
    TurnSegmentationConfig,
    TurnSignalSample,
)


def _valid_segments(samples: list[TurnSignalSample], config: TurnSegmentationConfig):
    current = []
    current_segment = None
    for index, sample in enumerate(samples):
        valid = (
            sample.temporal_segment_id is not None
            and sample.value is not None
            and sample.support_confidence is not None
            and sample.support_confidence >= config.minimum_signal_confidence
        )
        if not valid or (
            current_segment is not None and sample.temporal_segment_id != current_segment
        ):
            if current:
                yield current
            current = []
        if valid:
            current_segment = sample.temporal_segment_id
            current.append((index, sample))
        else:
            current_segment = None
    if current:
        yield current


class ReferencePeakDetector:
    def detect(
        self, samples: list[TurnSignalSample], config: TurnSegmentationConfig
    ) -> list[PeakCandidate]:
        config.validate()
        candidates = []
        for segment in _valid_segments(samples, config):
            position = 1
            while position < len(segment) - 1:
                plateau_end = position
                while (
                    plateau_end + 1 < len(segment)
                    and segment[plateau_end + 1][1].value == segment[position][1].value
                ):
                    plateau_end += 1
                index, item = segment[position]
                left = segment[position - 1][1]
                right = segment[plateau_end + 1][1] if plateau_end + 1 < len(segment) else item
                positive = item.value > left.value and item.value > right.value
                negative = item.value < left.value and item.value < right.value
                if not positive and not negative:
                    position = plateau_end + 1
                    continue
                prominence = min(abs(item.value - left.value), abs(item.value - right.value))
                if (
                    abs(item.value) < config.minimum_peak_amplitude
                    or prominence < config.minimum_peak_prominence
                ):
                    position = plateau_end + 1
                    continue
                candidates.append(
                    PeakCandidate(
                        item.timestamp_us,
                        item.temporal_segment_id,
                        item.value,
                        prominence,
                        TurnPhaseSign.POSITIVE_PHASE if positive else TurnPhaseSign.NEGATIVE_PHASE,
                        index,
                    )
                )
                position = plateau_end + 1
        accepted = []
        for candidate in candidates:
            if (
                accepted
                and candidate.temporal_segment_id == accepted[-1].temporal_segment_id
                and candidate.timestamp_us - accepted[-1].timestamp_us
                < config.minimum_peak_separation_us
            ):
                if abs(candidate.value) > abs(accepted[-1].value):
                    accepted[-1] = candidate
                continue
            if accepted and candidate.phase_sign is accepted[-1].phase_sign:
                if abs(candidate.value) > abs(accepted[-1].value):
                    accepted[-1] = candidate
                continue
            accepted.append(candidate)
        return accepted


class SciPyFindPeaksDetector:
    def detect(
        self, samples: list[TurnSignalSample], config: TurnSegmentationConfig
    ) -> list[PeakCandidate]:
        try:
            from scipy.signal import find_peaks  # noqa: F401
        except ImportError as error:
            raise RuntimeError("SCIPY_PEAK_DETECTOR_NOT_CONFIGURED") from error
        # SciPy's distance is sample-based and conflicts with irregular timestamp
        # semantics. This adapter intentionally delegates acceptance to the timestamp-safe
        # reference implementation after proving SciPy is explicitly configured.
        return ReferencePeakDetector().detect(samples, config)
