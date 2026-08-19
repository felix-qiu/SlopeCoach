from __future__ import annotations

import math
from dataclasses import dataclass

from slopecoach_ml.detection import Detection
from slopecoach_ml.pose import BoundingBox2D

from .contracts import TrackingConfig, TrackingFrame, TrackObservation, TrackState


def bbox_iou(first: BoundingBox2D, second: BoundingBox2D) -> float:
    left = max(first.x_px, second.x_px)
    top = max(first.y_px, second.y_px)
    right = min(first.x_px + first.width_px, second.x_px + second.width_px)
    bottom = min(first.y_px + first.height_px, second.y_px + second.height_px)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width_px * first.height_px + second.width_px * second.height_px - intersection
    return intersection / union if union > 0 else 0.0


def _center(box: BoundingBox2D) -> tuple[float, float]:
    return box.x_px + box.width_px / 2, box.y_px + box.height_px / 2


def _duplicate_like_conflict(
    first: _MutableTrack, second: _MutableTrack, config: TrackingConfig
) -> bool:
    """Use only pre-assignment track history to recognize same-person duplicates."""
    first_center, second_center = _center(first.bbox), _center(second.bbox)
    body_scale = max(
        1.0,
        math.hypot(first.bbox.width_px, first.bbox.height_px),
        math.hypot(second.bbox.width_px, second.bbox.height_px),
    )
    center_distance = math.dist(first_center, second_center) / body_scale
    first_area = first.bbox.width_px * first.bbox.height_px
    second_area = second.bbox.width_px * second.bbox.height_px
    scale_ratio = max(first_area, second_area) / min(first_area, second_area)
    if (
        bbox_iou(first.bbox, second.bbox) < config.duplicate_track_minimum_iou
        or center_distance > config.duplicate_track_maximum_normalized_center_distance
        or scale_ratio > config.duplicate_track_maximum_scale_ratio
    ):
        return False
    first_speed = math.hypot(first.velocity_x, first.velocity_y)
    second_speed = math.hypot(second.velocity_x, second.velocity_y)
    if first.hit_count >= 2 and second.hit_count >= 2 and first_speed > 0 and second_speed > 0:
        velocity_cosine = (
            first.velocity_x * second.velocity_x + first.velocity_y * second.velocity_y
        ) / (first_speed * second_speed)
        if velocity_cosine < config.duplicate_track_minimum_velocity_cosine:
            return False
    return True


@dataclass
class _MutableTrack:
    track_id: int
    bbox: BoundingBox2D
    confidence: float
    hit_count: int
    first_seen_us: int
    last_seen_us: int
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    detection_id: int | None = None


class ReferenceMotionIoUTracker:
    """Small deterministic research tracker; deliberately not branded ByteTrack."""

    implementation = "REFERENCE_MOTION_IOU"

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self.config.validate()
        self._tracks: dict[int, _MutableTrack] = {}
        self._next_id = 1
        self.total_tracks_created = 0
        self.total_tracks_terminated = 0
        self.preferred_association_count = 0
        self.preferred_association_override_count = 0
        self.preferred_association_conflict_count = 0
        self.preferred_association_rejected_non_duplicate_count = 0

    def _score(self, track: _MutableTrack, detection: Detection, timestamp_us: int) -> float | None:
        elapsed = max(0, timestamp_us - track.last_seen_us) / 1_000_000
        cx, cy = _center(track.bbox)
        predicted = (cx + track.velocity_x * elapsed, cy + track.velocity_y * elapsed)
        dcx, dcy = _center(detection.bbox)
        scale = max(1.0, math.hypot(track.bbox.width_px, track.bbox.height_px))
        distance = math.hypot(dcx - predicted[0], dcy - predicted[1]) / scale
        area_first = track.bbox.width_px * track.bbox.height_px
        area_second = detection.bbox.width_px * detection.bbox.height_px
        scale_ratio = max(area_first, area_second) / min(area_first, area_second)
        iou = bbox_iou(track.bbox, detection.bbox)
        if (
            distance > self.config.maximum_normalized_center_distance
            or scale_ratio > self.config.maximum_scale_ratio
            or (iou < self.config.minimum_iou and distance > 0.5)
        ):
            return None
        distance_score = max(0.0, 1 - distance / self.config.maximum_normalized_center_distance)
        scale_score = 1 / scale_ratio
        return (
            self.config.association_iou_weight * iou
            + self.config.association_distance_weight * distance_score
            + self.config.association_scale_weight * scale_score
        )

    def update(
        self,
        detections,
        timestamp_us,
        frame_index,
        geometry,
        *,
        preferred_track_id: int | None = None,
    ) -> TrackingFrame:
        geometry.validate()
        if timestamp_us < 0 or frame_index < 0:
            raise ValueError("timestamp and frame index must be non-negative")
        ordered = sorted(
            detections,
            key=lambda item: (
                item.bbox.x_px,
                item.bbox.y_px,
                item.bbox.width_px,
                item.bbox.height_px,
                -item.confidence,
            ),
        )
        for detection in ordered:
            detection.validate(geometry)
        pairs = []
        for track_id, track in self._tracks.items():
            for index, detection in enumerate(ordered):
                score = self._score(track, detection, timestamp_us)
                if score is not None and score >= self.config.minimum_association_score:
                    pairs.append((-score, track_id, index))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        preferred_decisions: dict[tuple[int, int], str] = {}
        for negative_score, track_id, index in pairs:
            if (
                track_id != preferred_track_id
                or -negative_score < self.config.minimum_preferred_association_score
            ):
                continue
            stronger = [
                (other_negative_score, other_track_id)
                for other_negative_score, other_track_id, other_index in pairs
                if other_index == index
                and other_track_id != track_id
                and other_negative_score < negative_score
            ]
            if not stronger:
                preferred_decisions[(track_id, index)] = "NO_STRONGER_COMPETITOR"
                continue
            self.preferred_association_conflict_count += 1
            preferred_track = self._tracks[track_id]
            duplicate_like = all(
                _duplicate_like_conflict(preferred_track, self._tracks[other_track_id], self.config)
                for _, other_track_id in stronger
            )
            if duplicate_like:
                preferred_decisions[(track_id, index)] = "DUPLICATE_OVERRIDE"
            else:
                # Continuity is optional; correct physical identity is mandatory.
                preferred_decisions[(track_id, index)] = "REJECTED_NON_DUPLICATE"
                self.preferred_association_rejected_non_duplicate_count += 1

        def association_order(pair: tuple[float, int, int]) -> tuple[bool, float, int, int]:
            negative_score, track_id, index = pair
            preferred = preferred_decisions.get((track_id, index)) in {
                "NO_STRONGER_COMPETITOR",
                "DUPLICATE_OVERRIDE",
            }
            return (not preferred, negative_score, track_id, index)

        for _negative_score, track_id, index in sorted(pairs, key=association_order):
            if track_id in matched_tracks or index in matched_detections:
                continue
            preferred_decision = preferred_decisions.get((track_id, index))
            preferred = preferred_decision in {
                "NO_STRONGER_COMPETITOR",
                "DUPLICATE_OVERRIDE",
            }
            if preferred:
                self.preferred_association_count += 1
                if preferred_decision == "DUPLICATE_OVERRIDE":
                    self.preferred_association_override_count += 1
            track, detection = self._tracks[track_id], ordered[index]
            elapsed = (timestamp_us - track.last_seen_us) / 1_000_000
            if elapsed > 0:
                old_center, new_center = _center(track.bbox), _center(detection.bbox)
                track.velocity_x = (new_center[0] - old_center[0]) / elapsed
                track.velocity_y = (new_center[1] - old_center[1]) / elapsed
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.hit_count += 1
            track.last_seen_us = timestamp_us
            track.detection_id = detection.detection_id
            matched_tracks.add(track_id)
            matched_detections.add(index)
        for index, detection in enumerate(ordered):
            if index in matched_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            self.total_tracks_created += 1
            self._tracks[track_id] = _MutableTrack(
                track_id,
                detection.bbox,
                detection.confidence,
                1,
                timestamp_us,
                timestamp_us,
                detection_id=detection.detection_id,
            )
            matched_tracks.add(track_id)
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if timestamp_us - track.last_seen_us > self.config.maximum_missed_duration_us
        ]
        for track_id in expired:
            del self._tracks[track_id]
            self.total_tracks_terminated += 1
        observations = []
        for track_id, track in sorted(self._tracks.items()):
            missed = timestamp_us - track.last_seen_us
            state = (
                TrackState.MISSING
                if missed
                else TrackState.CONFIRMED
                if track.hit_count >= self.config.confirmation_hits
                else TrackState.TENTATIVE
            )
            observations.append(
                TrackObservation(
                    track_id,
                    track.detection_id if not missed else None,
                    track.bbox,
                    track.confidence,
                    state,
                    track.hit_count,
                    track.first_seen_us,
                    track.last_seen_us,
                    missed,
                    track.velocity_x,
                    track.velocity_y,
                )
            )
        return TrackingFrame(timestamp_us, frame_index, tuple(observations))
