"""PROVISIONAL A3 identity models for Python research/reference validation only."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Protocol

from slopecoach_ml.pose import BoundingBox2D


class TargetIdentityState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    LOCKED = "LOCKED"
    SUSPECT = "SUSPECT"
    LOST = "LOST"
    RECOVERING = "RECOVERING"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class CandidateFilterConfig:
    minimum_detection_confidence: float = 0.10
    minimum_area_fraction: float = 0.00015
    minimum_visible_fraction: float = 0.20
    minimum_width_px: float = 6.0
    minimum_height_px: float = 12.0
    plausible_aspect_ratio_min: float = 0.12
    plausible_aspect_ratio_max: float = 2.0

    def validate(self) -> None:
        for name in (
            "minimum_detection_confidence",
            "minimum_area_fraction",
            "minimum_visible_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.minimum_width_px <= 0 or self.minimum_height_px <= 0:
            raise ValueError("minimum candidate dimensions must be positive")
        if not 0 < self.plausible_aspect_ratio_min < self.plausible_aspect_ratio_max:
            raise ValueError("candidate aspect ratio bounds are invalid")


@dataclass(frozen=True)
class InitialTargetSelectorConfig:
    initialization_window_us: int = 1_500_000
    minimum_track_observations: int = 3
    minimum_lock_score: float = 0.58
    minimum_winner_margin: float = 0.08
    center_weight: float = 0.15
    area_weight: float = 0.30
    persistence_weight: float = 0.25
    motion_weight: float = 0.10
    detection_weight: float = 0.10
    candidate_quality_weight: float = 0.10
    pose_quality_weight: float = 0.10
    maximum_history_staleness_us: int = 3_000_000

    def validate(self) -> None:
        if self.initialization_window_us < 0 or self.minimum_track_observations < 1:
            raise ValueError("initialization time/observation limits are invalid")
        if self.maximum_history_staleness_us < 0:
            raise ValueError("maximum_history_staleness_us must be non-negative")
        for name in (
            "minimum_lock_score",
            "minimum_winner_margin",
            "center_weight",
            "area_weight",
            "persistence_weight",
            "motion_weight",
            "detection_weight",
            "candidate_quality_weight",
            "pose_quality_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class TargetIdentityConfig:
    minimum_lock_score: float = 0.58
    minimum_winner_margin: float = 0.08
    safe_biomechanics_confidence: float = 0.62
    suspect_timeout_us: int = 750_000
    lost_timeout_us: int = 1_500_000
    recovery_confirmation_observations: int = 2
    track_continuity_weight: float = 0.22
    trajectory_weight: float = 0.18
    spatial_weight: float = 0.18
    bbox_scale_weight: float = 0.12
    body_proportion_weight: float = 0.10
    appearance_weight: float = 0.15
    candidate_quality_weight: float = 0.05
    pose_weight: float = 0.10
    maximum_trajectory_prediction_us: int = 1_500_000

    def validate(self) -> None:
        for name in (
            "minimum_lock_score",
            "minimum_winner_margin",
            "safe_biomechanics_confidence",
            "track_continuity_weight",
            "trajectory_weight",
            "spatial_weight",
            "bbox_scale_weight",
            "body_proportion_weight",
            "appearance_weight",
            "candidate_quality_weight",
            "pose_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.suspect_timeout_us < 0 or self.lost_timeout_us < self.suspect_timeout_us:
            raise ValueError("identity timeouts are invalid")
        if self.recovery_confirmation_observations < 1:
            raise ValueError("recovery_confirmation_observations must be positive")
        if self.maximum_trajectory_prediction_us < 0:
            raise ValueError("maximum_trajectory_prediction_us must be non-negative")


@dataclass(frozen=True)
class PoseSchedulingConfig:
    max_initial_pose_probe_candidates: int = 2
    max_identity_pose_candidates_per_frame: int = 2

    def validate(self) -> None:
        if self.max_initial_pose_probe_candidates < 0:
            raise ValueError("max_initial_pose_probe_candidates must be non-negative")
        if self.max_identity_pose_candidates_per_frame < 1:
            raise ValueError("max_identity_pose_candidates_per_frame must be positive")


@dataclass(frozen=True)
class PersonCandidate:
    detection_id: int
    bbox: BoundingBox2D
    detection_confidence: float
    quality_score: float
    evidence: dict[str, float]
    hard_rejection_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data


@dataclass(frozen=True)
class InitialSelectionEvidence:
    center_score: float | None
    area_score: float | None
    persistence_score: float | None
    motion_score: float | None
    detection_score: float | None
    candidate_quality_score: float | None
    pose_quality_score: float | None = None


@dataclass(frozen=True)
class InitialSelectionResult:
    selected_track_id: int | None
    score: float | None
    runner_up_score: float | None
    margin: float | None
    state: TargetIdentityState
    evidence: dict[int, InitialSelectionEvidence]


@dataclass(frozen=True)
class IdentityEvidence:
    track_continuity: float | None
    trajectory_similarity: float | None
    spatial_similarity: float | None
    bbox_scale_similarity: float | None
    body_proportion_similarity: float | None
    appearance_similarity: float | None
    candidate_quality: float | None
    pose_similarity: float | None
    reliability: float
    predicted_center: tuple[float, float] | None = None
    candidate_center: tuple[float, float] | None = None
    trajectory_normalized_distance: float | None = None


@dataclass(frozen=True)
class IdentityMatch:
    track_id: int
    fused_score: float
    evidence: IdentityEvidence


@dataclass
class TargetIdentity:
    target_id: str
    state: TargetIdentityState = TargetIdentityState.UNINITIALIZED
    active_track_id: int | None = None
    confidence: float = 0.0
    initial_selection_score: float | None = None
    last_bbox: BoundingBox2D | None = None
    last_seen_us: int | None = None
    velocity_x_px_per_s: float = 0.0
    velocity_y_px_per_s: float = 0.0
    velocity_available: bool = False
    trajectory_history: list[tuple[int, float, float]] = field(default_factory=list)
    appearance_gallery: list[Sequence[float]] = field(default_factory=list)


class AppearanceEncoder(Protocol):
    def encode(self, image: object, bbox: BoundingBox2D) -> Sequence[float] | None: ...
