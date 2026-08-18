"""Manual target initialization for research benchmarks; never Ground Truth."""

from __future__ import annotations

import math
from dataclasses import dataclass

from slopecoach_ml.pose import FrameGeometry
from slopecoach_ml.tracking import TrackObservation, TrackState

from .contracts import PersonCandidate


@dataclass(frozen=True)
class ManualTargetSeed:
    """A SourcePixel2D click used only to initialize TargetIdentity."""

    time_seconds: float
    x_px: float
    y_px: float

    def validate(self) -> None:
        for name in ("time_seconds", "x_px", "y_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.time_seconds < 0:
            raise ValueError("MANUAL_TARGET_SEED_TIME_NEGATIVE")

    @property
    def requested_timestamp_us(self) -> int:
        self.validate()
        return round(self.time_seconds * 1_000_000)


@dataclass(frozen=True)
class ManualTargetSeedMatch:
    track: TrackObservation
    candidate: PersonCandidate
    normalized_center_distance: float


def manual_seed_timestamp_tolerance_us(sample_fps: float) -> float:
    if isinstance(sample_fps, bool) or not isinstance(sample_fps, int | float):
        raise TypeError("sample_fps must be numeric")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and positive")
    return 500_000 / sample_fps + 1e-6


def manual_seed_frame_is_eligible(
    seed: ManualTargetSeed, sample_timestamp_us: int, sample_fps: float
) -> bool:
    return abs(
        sample_timestamp_us - seed.requested_timestamp_us
    ) <= manual_seed_timestamp_tolerance_us(sample_fps)


def manual_seed_window_has_passed(
    seed: ManualTargetSeed, sample_timestamp_us: int, sample_fps: float
) -> bool:
    return sample_timestamp_us > seed.requested_timestamp_us + manual_seed_timestamp_tolerance_us(
        sample_fps
    )


def select_manual_target_seed_match(
    seed: ManualTargetSeed,
    geometry: FrameGeometry,
    tracks: tuple[TrackObservation, ...],
    candidates_by_detection: dict[int, PersonCandidate],
) -> ManualTargetSeedMatch:
    """Resolve a click against current viable tracker/candidate state deterministically."""

    seed.validate()
    geometry.validate()
    if not (0 <= seed.x_px < geometry.width_px and 0 <= seed.y_px < geometry.height_px):
        raise ValueError("MANUAL_TARGET_SEED_POINT_OUT_OF_BOUNDS")

    matches: list[ManualTargetSeedMatch] = []
    for track in tracks:
        if track.state is TrackState.MISSING or track.detection_id is None:
            continue
        candidate = candidates_by_detection.get(track.detection_id)
        if candidate is None or candidate.hard_rejection_reason is not None:
            continue
        bbox = track.bbox
        if not (
            bbox.x_px <= seed.x_px <= bbox.x_px + bbox.width_px
            and bbox.y_px <= seed.y_px <= bbox.y_px + bbox.height_px
        ):
            continue
        center_x = bbox.x_px + bbox.width_px / 2
        center_y = bbox.y_px + bbox.height_px / 2
        normalized_distance = math.hypot(seed.x_px - center_x, seed.y_px - center_y) / max(
            1.0, math.hypot(bbox.width_px, bbox.height_px)
        )
        matches.append(ManualTargetSeedMatch(track, candidate, normalized_distance))

    if not matches:
        raise ValueError("MANUAL_TARGET_SEED_NO_MATCH")
    return min(
        matches,
        key=lambda item: (
            item.normalized_center_distance,
            -item.candidate.quality_score,
            -item.track.confidence,
            item.track.track_id,
        ),
    )
