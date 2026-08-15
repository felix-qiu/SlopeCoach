"""Single-pass collector for A6.1 LOCKED target equipment contexts."""

from __future__ import annotations

from typing import Any

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import BoundingBox2D, PoseFrame
from slopecoach_ml.sport_type import TargetSportFrameContext

from .temporal_turns import TemporalTurnCollector


class SportTypeBenchmarkCollector(TemporalTurnCollector):
    """Keeps raw pixels only for LOCKED target frames; debug images are JPEG-compressed."""

    def __init__(self, *, keep_images: bool = False) -> None:
        super().__init__(keep_images=False)
        self.keep_sport_images = keep_images
        self.sport_contexts: list[TargetSportFrameContext] = []

    def observe(self, frame, observation: dict[str, Any], pose_frame: PoseFrame | None) -> None:
        super().observe(frame, observation, pose_frame)
        if (
            observation["identity_state"] != TargetIdentityState.LOCKED.value
            or observation["selected_bbox"] is None
            or observation["target_id"] is None
            or observation["active_track_id"] is None
        ):
            return
        context = TargetSportFrameContext(
            timestamp_us=observation["timestamp_us"],
            frame_index=observation["frame_index"],
            geometry=frame.geometry,
            target_id=observation["target_id"],
            active_track_id=observation["active_track_id"],
            target_bbox=BoundingBox2D.from_dict(observation["selected_bbox"]),
            identity_state=TargetIdentityState.LOCKED,
            frame_reference=frame.image.copy(),
        )
        self.sport_contexts.append(context)
        if self.keep_sport_images:
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError("DEBUG_DEPENDENCY_MISSING: opencv-python") from error
            ok, encoded = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise RuntimeError("SPORT_TYPE_DEBUG_FRAME_ENCODE_FAILED")
            self.images[frame.frame_index] = encoded.tobytes()

    def release_frame_contexts(self) -> None:
        self.sport_contexts.clear()
