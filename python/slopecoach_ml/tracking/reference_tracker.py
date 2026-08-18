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

        def association_order(pair: tuple[float, int, int]) -> tuple[bool, float, int, int]:
            negative_score, track_id, index = pair
            preferred = (
                track_id == preferred_track_id
                and -negative_score >= self.config.minimum_preferred_association_score
            )
            return (not preferred, negative_score, track_id, index)

        for negative_score, track_id, index in sorted(pairs, key=association_order):
            if track_id in matched_tracks or index in matched_detections:
                continue
            preferred = (
                track_id == preferred_track_id
                and -negative_score >= self.config.minimum_preferred_association_score
            )
            if preferred:
                self.preferred_association_count += 1
                if any(
                    other_index == index
                    and other_track_id != track_id
                    and other_negative_score < negative_score
                    for other_negative_score, other_track_id, other_index in pairs
                ):
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
