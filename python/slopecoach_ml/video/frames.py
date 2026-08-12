from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slopecoach_ml.pose import FrameGeometry


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_us: int
    geometry: FrameGeometry
    image: Any


def probe_rotation(path: str | Path) -> int | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        payload = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    if result.returncode != 0 or not payload.get("streams"):
        return None
    stream = payload["streams"][0]
    values = [stream.get("tags", {}).get("rotate")]
    values.extend(item.get("rotation") for item in stream.get("side_data_list", []))
    for value in values:
        if value is not None:
            return int(round(float(value))) % 360
    return 0


class OpenCVVideoSampler:
    """Research sampler using decoded timestamps, not frame_index/FPS as authority."""

    def __init__(self, path: str | Path, *, sample_fps: float = 2.0) -> None:
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        self.path = Path(path)
        self.sample_fps = sample_fps

    def __iter__(self) -> Iterator[SampledFrame]:
        rotation = probe_rotation(self.path)
        if rotation not in {0, 90, 180, 270}:
            raise RuntimeError("ORIENTATION_NORMALIZATION_UNSUPPORTED")
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OPENMMLAB_DEPENDENCY_MISSING: opencv-python") from error
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError("VIDEO_DECODE_FAILED")
        next_timestamp_us = 0
        interval_us = round(1_000_000 / self.sample_fps)
        frame_index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                timestamp_us = max(0, round(capture.get(cv2.CAP_PROP_POS_MSEC) * 1000))
                if timestamp_us < next_timestamp_us:
                    frame_index += 1
                    continue
                if rotation == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif rotation == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                height, width = image.shape[:2]
                yield SampledFrame(frame_index, timestamp_us, FrameGeometry(width, height), image)
                next_timestamp_us = timestamp_us + interval_us
                frame_index += 1
        finally:
            capture.release()
