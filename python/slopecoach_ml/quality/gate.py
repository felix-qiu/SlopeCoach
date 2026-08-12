from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from slopecoach_ml.video import VideoMetadata


class VideoQualityStatus(StrEnum):
    READY = "READY"
    PARTIAL_ANALYSIS = "PARTIAL_ANALYSIS"
    NOT_ANALYZABLE = "NOT_ANALYZABLE"


@dataclass(frozen=True)
class VideoQuality:
    status: VideoQualityStatus
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["reasons"] = list(self.reasons)
        return data


class VideoQualityGate:
    def __init__(self, *, minimum_usable_frame_ratio: float = 0.8) -> None:
        if not 0.0 <= minimum_usable_frame_ratio <= 1.0:
            raise ValueError("minimum usable frame ratio must be in [0, 1]")
        self.minimum_usable_frame_ratio = minimum_usable_frame_ratio

    def evaluate(self, metadata: VideoMetadata) -> VideoQuality:
        fatal: list[str] = []
        partial: list[str] = []
        if not metadata.readable:
            fatal.append(metadata.error or "video is unreadable")
        if metadata.width_px is None or metadata.height_px is None:
            fatal.append("video dimensions are unavailable")
        elif metadata.width_px <= 0 or metadata.height_px <= 0:
            fatal.append("video dimensions must be positive")
        if metadata.duration_seconds is None:
            partial.append("duration is unavailable")
        elif metadata.duration_seconds <= 0:
            fatal.append("duration must be positive")
        if metadata.frame_count is None:
            partial.append("frame count is unavailable")
        elif metadata.frame_count <= 0:
            fatal.append("frame count must be positive")
        if metadata.usable_frame_ratio is None:
            partial.append("usable frame ratio is not measured")
        elif not 0.0 <= metadata.usable_frame_ratio <= 1.0:
            fatal.append("usable frame ratio must be in [0, 1]")
        elif metadata.usable_frame_ratio < self.minimum_usable_frame_ratio:
            partial.append("usable frame ratio is below the ready threshold")
        if fatal:
            return VideoQuality(VideoQualityStatus.NOT_ANALYZABLE, tuple(fatal + partial))
        if partial:
            return VideoQuality(VideoQualityStatus.PARTIAL_ANALYSIS, tuple(partial))
        return VideoQuality(VideoQualityStatus.READY, ())
