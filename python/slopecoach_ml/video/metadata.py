from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoMetadata:
    path: str | None
    readable: bool
    duration_seconds: float | None
    frame_count: int | None
    width_px: int | None
    height_px: int | None
    usable_frame_ratio: float | None = None
    error: str | None = None
    container: str | None = None
    codec: str | None = None
    nominal_fps: float | None = None
    average_fps: float | None = None
    pixel_aspect_ratio: float | None = None
    rotation_degrees: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    normalized = value.replace(":", "/")
    numerator, separator, denominator = normalized.partition("/")
    try:
        result = float(numerator) / float(denominator) if separator else float(value)
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def inspect_video(path: str | Path, *, ffprobe: str = "ffprobe") -> VideoMetadata:
    video_path = Path(path)
    if not video_path.is_file():
        return VideoMetadata(str(video_path), False, None, None, None, None, error="file not found")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,codec_name,nb_read_frames,nb_frames,r_frame_rate,avg_frame_rate,"
        "sample_aspect_ratio,duration:stream_tags=rotate:stream_side_data=rotation:"
        "format=format_name,duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    except FileNotFoundError:
        return VideoMetadata(
            str(video_path), False, None, None, None, None, error="ffprobe not found"
        )
    except subprocess.TimeoutExpired:
        return VideoMetadata(
            str(video_path), False, None, None, None, None, error="ffprobe timed out"
        )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        return VideoMetadata(
            str(video_path),
            False,
            None,
            None,
            None,
            None,
            error=detail[-1] if detail else "ffprobe failed",
        )
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        duration_value = stream.get("duration") or payload.get("format", {}).get("duration")
        duration = float(duration_value) if duration_value is not None else None
        frame_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
        frame_count = int(frame_raw) if frame_raw not in (None, "N/A") else None
        if frame_count is None and duration is not None:
            frame_rate = _parse_rate(stream.get("avg_frame_rate"))
            frame_count = round(duration * frame_rate) if frame_rate is not None else None
        if duration is not None and not math.isfinite(duration):
            duration = None
        nominal_fps = _parse_rate(stream.get("r_frame_rate"))
        average_fps = _parse_rate(stream.get("avg_frame_rate"))
        par = _parse_rate(stream.get("sample_aspect_ratio"))
        rotation = None
        rotation_values = [stream.get("tags", {}).get("rotate")]
        rotation_values.extend(item.get("rotation") for item in stream.get("side_data_list", []))
        for value in rotation_values:
            if value is not None:
                rotation = int(round(float(value))) % 360
                break
        if rotation is None:
            rotation = 0
        container = payload.get("format", {}).get("format_name")
        codec = stream.get("codec_name")
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return VideoMetadata(
            str(video_path), False, None, None, None, None, error=f"invalid ffprobe output: {error}"
        )
    return VideoMetadata(
        str(video_path),
        True,
        duration,
        frame_count,
        width,
        height,
        container=container,
        codec=codec,
        nominal_fps=nominal_fps,
        average_fps=average_fps,
        pixel_aspect_ratio=par,
        rotation_degrees=rotation,
    )
