"""Deterministic reference extrema detection plus optional lazy SciPy adapter."""

from __future__ import annotations

from .contracts import (
    PeakCandidate,
    TurnPhaseSign,
    TurnSegmentationConfig,
    TurnSignalSample,
)
from .runs import valid_signal_runs


class ReferencePeakDetector:
    def detect(
        self, samples: list[TurnSignalSample], config: TurnSegmentationConfig
    ) -> list[PeakCandidate]:
        config.validate()
        accepted = []
        for run in valid_signal_runs(samples, config):
            raw_candidates = []
            segment = run.indexed_samples
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
                raw_candidates.append(
                    PeakCandidate(
                        timestamp_us=item.timestamp_us,
                        temporal_segment_id=item.temporal_segment_id,
                        signal_run_id=run.signal_run_id,
                        value=item.value,
                        prominence=prominence,
                        phase_sign=(
                            TurnPhaseSign.POSITIVE_PHASE
                            if positive
                            else TurnPhaseSign.NEGATIVE_PHASE
                        ),
                        sample_index=index,
                    )
                )
                position = plateau_end + 1
            accepted_for_run = []
            for candidate in raw_candidates:
                if (
                    accepted_for_run
                    and candidate.timestamp_us - accepted_for_run[-1].timestamp_us
                    < config.minimum_peak_separation_us
                ):
                    if abs(candidate.value) > abs(accepted_for_run[-1].value):
                        accepted_for_run[-1] = candidate
                    continue
                if accepted_for_run and candidate.phase_sign is accepted_for_run[-1].phase_sign:
                    if abs(candidate.value) > abs(accepted_for_run[-1].value):
                        accepted_for_run[-1] = candidate
                    continue
                accepted_for_run.append(candidate)
            accepted.extend(accepted_for_run)
        return sorted(
            accepted, key=lambda candidate: (candidate.timestamp_us, candidate.sample_index)
        )


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
