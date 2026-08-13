"""PROVISIONAL A3 research tracking contracts; Rust remains production source of truth."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Protocol

from slopecoach_ml.detection import Detection
from slopecoach_ml.pose import BoundingBox2D, FrameGeometry


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    MISSING = "MISSING"


@dataclass(frozen=True)
class TrackingConfig:
    implementation: str = "REFERENCE_MOTION_IOU"
    minimum_iou: float = 0.05
    maximum_normalized_center_distance: float = 1.25
    maximum_scale_ratio: float = 3.0
    association_iou_weight: float = 0.5
    association_distance_weight: float = 0.35
    association_scale_weight: float = 0.15
    minimum_association_score: float = 0.25
    confirmation_hits: int = 2
    maximum_missed_duration_us: int = 1_500_000

    def validate(self) -> None:
        for name in (
            "minimum_iou",
            "association_iou_weight",
            "association_distance_weight",
            "association_scale_weight",
            "minimum_association_score",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.maximum_normalized_center_distance <= 0 or self.maximum_scale_ratio < 1:
            raise ValueError("tracking distance/scale limits are invalid")
        if self.confirmation_hits < 1 or self.maximum_missed_duration_us < 0:
            raise ValueError("tracking lifecycle limits are invalid")
        if not math.isclose(
            self.association_iou_weight
            + self.association_distance_weight
            + self.association_scale_weight,
            1.0,
        ):
            raise ValueError("association weights must sum to 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    detection_id: int | None
    bbox: BoundingBox2D
    confidence: float
    state: TrackState
    hit_count: int
    first_seen_us: int
    last_seen_us: int
    missed_duration_us: int
    velocity_x_px_per_s: float
    velocity_y_px_per_s: float

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["state"] = self.state.value
        data["bbox"] = self.bbox.to_dict()
        return data


@dataclass(frozen=True)
class TrackingFrame:
    timestamp_us: int
    frame_index: int
    tracks: tuple[TrackObservation, ...]


class Tracker(Protocol):
    def update(
        self,
        detections: tuple[Detection, ...],
        timestamp_us: int,
        frame_index: int,
        geometry: FrameGeometry,
    ) -> TrackingFrame: ...
