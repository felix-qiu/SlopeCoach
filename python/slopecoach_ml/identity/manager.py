from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from slopecoach_ml.tracking import TrackObservation, TrackState

from .appearance import descriptor_similarity, update_gallery
from .contracts import (
    IdentityEvidence,
    IdentityMatch,
    PersonCandidate,
    TargetIdentity,
    TargetIdentityConfig,
    TargetIdentityState,
)
from .selector import weighted_available


@dataclass(frozen=True)
class RecoveryEvent:
    old_track_id: int | None
    new_track_id: int
    lost_duration_us: int
    recovery_confidence: float


class TargetIdentityManager:
    def __init__(
        self, config: TargetIdentityConfig | None = None, *, target_id: str = "target-1"
    ) -> None:
        self.config = config or TargetIdentityConfig()
        self.config.validate()
        self.identity = TargetIdentity(target_id)
        self.recovery_events: list[RecoveryEvent] = []
        self.active_track_id_change_count = 0
        self.relock_count = 0
        self._recovery_candidate: int | None = None
        self._recovery_hits = 0
        self._old_track_id: int | None = None

    def initialize(
        self, track: TrackObservation, score: float, timestamp_us: int, descriptor=None
    ) -> None:
        self.identity.state = TargetIdentityState.LOCKED
        self.identity.active_track_id = track.track_id
        self.identity.confidence = score
        self.identity.initial_selection_score = score
        self._remember(track, timestamp_us)
        update_gallery(self.identity.appearance_gallery, descriptor, quality=score)

    def _remember(self, track: TrackObservation, timestamp_us: int) -> None:
        self.identity.last_bbox = track.bbox
        self.identity.last_seen_us = timestamp_us
        self.identity.velocity_x_px_per_s = track.velocity_x_px_per_s
        self.identity.velocity_y_px_per_s = track.velocity_y_px_per_s
        center = (
            track.bbox.x_px + track.bbox.width_px / 2,
            track.bbox.y_px + track.bbox.height_px / 2,
        )
        self.identity.trajectory_history.append((timestamp_us, *center))
        del self.identity.trajectory_history[:-20]

    def _match(
        self,
        track: TrackObservation,
        candidate: PersonCandidate,
        descriptor=None,
        pose_similarity: float | None = None,
    ) -> IdentityMatch:
        last = self.identity.last_bbox
        if last is None:
            spatial = trajectory = scale = proportion = None
        else:
            last_center = (last.x_px + last.width_px / 2, last.y_px + last.height_px / 2)
            center = (
                track.bbox.x_px + track.bbox.width_px / 2,
                track.bbox.y_px + track.bbox.height_px / 2,
            )
            body_scale = max(1.0, math.hypot(last.width_px, last.height_px))
            distance = (
                math.hypot(center[0] - last_center[0], center[1] - last_center[1]) / body_scale
            )
            spatial = max(0.0, 1 - distance / 2)
            trajectory = max(0.0, 1 - distance)
            areas = (last.width_px * last.height_px, track.bbox.width_px * track.bbox.height_px)
            scale = min(areas) / max(areas)
            proportions = (
                last.width_px / last.height_px,
                track.bbox.width_px / track.bbox.height_px,
            )
            proportion = min(proportions) / max(proportions)
        appearance = None
        if descriptor is not None and self.identity.appearance_gallery:
            values = [
                descriptor_similarity(item, descriptor) for item in self.identity.appearance_gallery
            ]
            usable = [item for item in values if item is not None]
            appearance = max(usable) if usable else None
        continuity = 1.0 if track.track_id == self.identity.active_track_id else 0.0
        evidence = IdentityEvidence(
            continuity,
            trajectory,
            spatial,
            scale,
            proportion,
            appearance,
            candidate.quality_score,
            pose_similarity,
            candidate.quality_score,
        )
        values = {
            "track": continuity,
            "trajectory": trajectory,
            "spatial": spatial,
            "scale": scale,
            "proportion": proportion,
            "appearance": appearance,
            "quality": candidate.quality_score,
            "pose": pose_similarity,
        }
        weights = {
            "track": self.config.track_continuity_weight,
            "trajectory": self.config.trajectory_weight,
            "spatial": self.config.spatial_weight,
            "scale": self.config.bbox_scale_weight,
            "proportion": self.config.body_proportion_weight,
            "appearance": self.config.appearance_weight,
            "quality": self.config.candidate_quality_weight,
            "pose": self.config.pose_weight,
        }
        score = weighted_available(values, weights) or 0.0
        return IdentityMatch(track.track_id, score * evidence.reliability, evidence)

    def update(
        self,
        tracks,
        candidates_by_detection,
        timestamp_us,
        descriptors=None,
        pose_similarities=None,
    ):
        viable = []
        for track in tracks:
            if track.state is TrackState.MISSING or track.detection_id is None:
                continue
            candidate = candidates_by_detection.get(track.detection_id)
            if candidate is None or candidate.hard_rejection_reason:
                continue
            viable.append((track, candidate))
        active = next(
            (track for track, _ in viable if track.track_id == self.identity.active_track_id), None
        )
        if self.identity.state is TargetIdentityState.LOCKED and active is not None:
            candidate = next(candidate for track, candidate in viable if track is active)
            match = self._match(
                active,
                candidate,
                (descriptors or {}).get(active.track_id),
                (pose_similarities or {}).get(active.track_id),
            )
            self.identity.confidence = match.fused_score
            self._remember(active, timestamp_us)
            update_gallery(
                self.identity.appearance_gallery,
                (descriptors or {}).get(active.track_id),
                quality=match.evidence.reliability,
            )
            if match.fused_score < self.config.minimum_lock_score:
                self.identity.state = TargetIdentityState.SUSPECT
            return (match,)
        if self.identity.state is TargetIdentityState.SUSPECT and active is not None:
            candidate = next(candidate for track, candidate in viable if track is active)
            match = self._match(
                active,
                candidate,
                (descriptors or {}).get(active.track_id),
                (pose_similarities or {}).get(active.track_id),
            )
            if match.fused_score >= self.config.minimum_lock_score:
                self.identity.state = TargetIdentityState.LOCKED
                self.identity.confidence = match.fused_score
                self._remember(active, timestamp_us)
                update_gallery(
                    self.identity.appearance_gallery,
                    (descriptors or {}).get(active.track_id),
                    quality=match.evidence.reliability,
                )
            return (match,)
        if self.identity.state is TargetIdentityState.LOCKED:
            self.identity.state = TargetIdentityState.SUSPECT
            self._old_track_id = self.identity.active_track_id
            return ()
        last_seen = self.identity.last_seen_us
        missing_duration = timestamp_us - (last_seen if last_seen is not None else timestamp_us)
        if (
            self.identity.state is TargetIdentityState.SUSPECT
            and missing_duration <= self.config.suspect_timeout_us
        ):
            return ()
        if self.identity.state is TargetIdentityState.SUSPECT:
            self.identity.state = TargetIdentityState.LOST
            self.identity.active_track_id = None
            self.identity.confidence = 0.0
        if not viable:
            if (
                self.identity.state
                in {TargetIdentityState.RECOVERING, TargetIdentityState.AMBIGUOUS}
                and missing_duration > self.config.lost_timeout_us
            ):
                self.identity.state = TargetIdentityState.LOST
            return ()
        matches = sorted(
            (
                self._match(
                    track,
                    candidate,
                    (descriptors or {}).get(track.track_id),
                    (pose_similarities or {}).get(track.track_id),
                )
                for track, candidate in viable
            ),
            key=lambda item: (-item.fused_score, item.track_id),
        )
        best = matches[0]
        runner = matches[1].fused_score if len(matches) > 1 else None
        margin = best.fused_score - runner if runner is not None else best.fused_score
        if best.fused_score < self.config.minimum_lock_score:
            self.identity.state = TargetIdentityState.LOST
            return tuple(matches)
        if margin < self.config.minimum_winner_margin:
            self.identity.state = TargetIdentityState.AMBIGUOUS
            self.identity.active_track_id = None
            self.identity.confidence = best.fused_score
            return tuple(matches)
        self.identity.state = TargetIdentityState.RECOVERING
        if self._recovery_candidate == best.track_id:
            self._recovery_hits += 1
        else:
            self._recovery_candidate, self._recovery_hits = best.track_id, 1
        self.identity.confidence = best.fused_score
        if self._recovery_hits >= self.config.recovery_confirmation_observations:
            track = next(track for track, _ in viable if track.track_id == best.track_id)
            old = self._old_track_id
            if old != best.track_id:
                self.active_track_id_change_count += 1
            self.identity.active_track_id = best.track_id
            self.identity.state = TargetIdentityState.LOCKED
            self._remember(track, timestamp_us)
            self.relock_count += 1
            self.recovery_events.append(
                RecoveryEvent(old, best.track_id, missing_duration, best.fused_score)
            )
            descriptor = (descriptors or {}).get(best.track_id)
            update_gallery(
                self.identity.appearance_gallery, descriptor, quality=best.evidence.reliability
            )
        return tuple(matches)

    def to_dict(self):
        data = asdict(self.identity)
        data["state"] = self.identity.state.value
        data["last_bbox"] = self.identity.last_bbox.to_dict() if self.identity.last_bbox else None
        return data
