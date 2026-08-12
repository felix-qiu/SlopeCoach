from __future__ import annotations

from slopecoach_ml.quality import VideoQualityGate, VideoQualityStatus
from slopecoach_ml.video import VideoMetadata, inspect_video


def test_unreadable_input(tmp_path) -> None:
    metadata = inspect_video(tmp_path / "missing.mp4")
    assert (
        VideoQualityGate().evaluate(metadata).status
        is VideoQualityStatus.NOT_ANALYZABLE
    )


def test_invalid_dimensions() -> None:
    metadata = VideoMetadata(None, True, 2, 60, 0, 1080, 1.0)
    assert (
        VideoQualityGate().evaluate(metadata).status
        is VideoQualityStatus.NOT_ANALYZABLE
    )


def test_valid_metadata_is_ready() -> None:
    metadata = VideoMetadata(None, True, 2, 60, 1920, 1080, 0.95)
    assert VideoQualityGate().evaluate(metadata).status is VideoQualityStatus.READY


def test_unknown_usable_ratio_is_partial_not_fabricated() -> None:
    metadata = VideoMetadata(None, True, 2, 60, 1920, 1080, None)
    quality = VideoQualityGate().evaluate(metadata)
    assert quality.status is VideoQualityStatus.PARTIAL_ANALYSIS
    assert "usable frame ratio is not measured" in quality.reasons
