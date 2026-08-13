from __future__ import annotations

import math
from dataclasses import dataclass, field

from slopecoach_ml.pose import FrameGeometry
from slopecoach_ml.tracking import TrackObservation, TrackState

from .contracts import (
    InitialSelectionEvidence,
    InitialSelectionResult,
    InitialTargetSelectorConfig,
    PersonCandidate,
    TargetIdentityState,
)


def weighted_available(values: dict[str, float | None], weights: dict[str, float]) -> float | None:
    available = [(value, weights[name]) for name, value in values.items() if value is not None]
    denominator = sum(weight for _, weight in available)
    return sum(value * weight for value, weight in available) / denominator if denominator else None


@dataclass
class _History:
    observations: int = 0
    first_seen_us: int = 0
    last_seen_us: int = 0
    centers: list[tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0
    quality: float = 0.0
    area_fraction: float = 0.0
    center_score: float = 0.0


class AutoInitialTargetSelector:
    def __init__(self, config: InitialTargetSelectorConfig | None = None) -> None:
        self.config = config or InitialTargetSelectorConfig()
        self.config.validate()
        self._started_us: int | None = None
        self._history: dict[int, _History] = {}

    def observe(
        self,
        tracks: tuple[TrackObservation, ...],
        candidates_by_detection: dict[int, PersonCandidate],
        geometry: FrameGeometry,
        timestamp_us: int,
        pose_quality: dict[int, float] | None = None,
    ) -> InitialSelectionResult:
        if self._started_us is None:
            self._started_us = timestamp_us
        expired = [
            track_id
            for track_id, history in self._history.items()
            if timestamp_us - history.last_seen_us > self.config.maximum_history_staleness_us
        ]
        for track_id in expired:
            del self._history[track_id]
        current_eligible_track_ids: set[int] = set()
        frame_center = (geometry.width_px / 2, geometry.height_px / 2)
        frame_diag = math.hypot(geometry.width_px, geometry.height_px)
        for track in tracks:
            if track.state is TrackState.MISSING or track.detection_id is None:
                continue
            candidate = candidates_by_detection.get(track.detection_id)
            if candidate is None or candidate.hard_rejection_reason:
                continue
            current_eligible_track_ids.add(track.track_id)
            history = self._history.setdefault(track.track_id, _History(first_seen_us=timestamp_us))
            center = (
                track.bbox.x_px + track.bbox.width_px / 2,
                track.bbox.y_px + track.bbox.height_px / 2,
            )
            history.observations += 1
            history.last_seen_us = timestamp_us
            history.centers.append(center)
            history.confidence = track.confidence
            history.quality = candidate.quality_score
            history.area_fraction = candidate.evidence["area_fraction"]
            history.center_score = max(
                0.0,
                1
                - 2
                * math.hypot(center[0] - frame_center[0], center[1] - frame_center[1])
                / frame_diag,
            )
        scores: list[tuple[float, int]] = []
        evidence: dict[int, InitialSelectionEvidence] = {}
        weights = {
            "center": self.config.center_weight,
            "area": self.config.area_weight,
            "persistence": self.config.persistence_weight,
            "motion": self.config.motion_weight,
            "detection": self.config.detection_weight,
            "quality": self.config.candidate_quality_weight,
            "pose": self.config.pose_quality_weight,
        }
        maximum_area_fraction = max(
            (history.area_fraction for history in self._history.values()), default=0.0
        )
        for track_id, history in self._history.items():
            motion = None
            if len(history.centers) >= 2:
                displacement = (
                    math.hypot(
                        history.centers[-1][0] - history.centers[0][0],
                        history.centers[-1][1] - history.centers[0][1],
                    )
                    / frame_diag
                )
                motion = min(1.0, displacement / 0.12)
            area = (
                history.area_fraction / maximum_area_fraction if maximum_area_fraction > 0 else None
            )
            persistence = min(1.0, history.observations / self.config.minimum_track_observations)
            pose = pose_quality.get(track_id) if pose_quality else None
            item = InitialSelectionEvidence(
                history.center_score,
                area,
                persistence,
                motion,
                history.confidence,
                history.quality,
                pose,
            )
            evidence[track_id] = item
            score = weighted_available(
                {
                    "center": item.center_score,
                    "area": item.area_score,
                    "persistence": item.persistence_score,
                    "motion": item.motion_score,
                    "detection": item.detection_score,
                    "quality": item.candidate_quality_score,
                    "pose": item.pose_quality_score,
                },
                weights,
            )
            if score is not None:
                scores.append((score, track_id))
        scores.sort(reverse=True)
        eligible_scores = [item for item in scores if item[1] in current_eligible_track_ids]
        elapsed = timestamp_us - self._started_us
        if elapsed < self.config.initialization_window_us:
            return InitialSelectionResult(
                None,
                eligible_scores[0][0] if eligible_scores else None,
                eligible_scores[1][0] if len(eligible_scores) > 1 else None,
                None,
                TargetIdentityState.UNINITIALIZED,
                evidence,
            )
        best = eligible_scores[0] if eligible_scores else None
        runner = eligible_scores[1][0] if len(eligible_scores) > 1 else None
        margin = best[0] - runner if best and runner is not None else best[0] if best else None
        eligible = (
            best and self._history[best[1]].observations >= self.config.minimum_track_observations
        )
        if (
            eligible
            and best[0] >= self.config.minimum_lock_score
            and margin is not None
            and margin >= self.config.minimum_winner_margin
        ):
            return InitialSelectionResult(
                best[1], best[0], runner, margin, TargetIdentityState.LOCKED, evidence
            )
        state = (
            TargetIdentityState.AMBIGUOUS
            if len(eligible_scores) > 1
            else TargetIdentityState.UNINITIALIZED
        )
        return InitialSelectionResult(
            None, best[0] if best else None, runner, margin, state, evidence
        )
