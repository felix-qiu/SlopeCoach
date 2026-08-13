from __future__ import annotations

from pathlib import Path
from typing import Any

from slopecoach_ml.pose import PoseFrame, overlay_primitives
from slopecoach_ml.video import SampledFrame


def select_target_debug_indices(
    observations: list[dict[str, Any]], max_frames: int = 12
) -> list[int]:
    if max_frames <= 0:
        return []
    selected: list[int] = []

    def add(item: dict[str, Any] | None) -> None:
        if item and item["frame_index"] not in selected and len(selected) < max_frames:
            selected.append(item["frame_index"])

    add(observations[0] if observations else None)
    for state in ("LOCKED", "SUSPECT", "LOST", "RECOVERING", "AMBIGUOUS"):
        add(next((item for item in observations if item["identity_state"] == state), None))
    locked = [item for item in observations if item["identity_state"] == "LOCKED"]
    if locked:
        add(locked[len(locked) // 2])
        add(min(locked, key=lambda item: item["identity_confidence"]))
    changes = [
        item
        for previous, item in zip(observations, observations[1:], strict=False)
        if previous["active_track_id"] != item["active_track_id"]
    ]
    add(changes[0] if changes else None)
    return selected


class TargetIdentityDebugCollector:
    def __init__(self) -> None:
        self._records: dict[int, tuple[bytes, dict[str, Any], PoseFrame | None]] = {}

    def observe(
        self, frame: SampledFrame, observation: dict[str, Any], pose: PoseFrame | None
    ) -> None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("DEBUG_DEPENDENCY_MISSING: opencv-python") from error
        ok, encoded = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("DEBUG_FRAME_ENCODE_FAILED")
        self._records[frame.frame_index] = (encoded.tobytes(), observation, pose)

    def write(self, output_dir: str | Path, observations, *, max_frames: int = 12):
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise RuntimeError("DEBUG_DEPENDENCY_MISSING: opencv-python/numpy") from error
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        indices = select_target_debug_indices(observations, max_frames)
        paths, images = [], []
        for frame_index in indices:
            encoded, observation, pose = self._records[frame_index]
            canvas = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            active = observation["active_track_id"]
            for track in observation["tracks"]:
                box = track["bbox"]
                left, top = round(box["x_px"]), round(box["y_px"])
                right = round(box["x_px"] + box["width_px"])
                bottom = round(box["y_px"] + box["height_px"])
                is_target = (
                    track["track_id"] == active and observation["identity_state"] == "LOCKED"
                )
                color, thickness = ((0, 255, 0), 4) if is_target else ((160, 160, 160), 1)
                cv2.rectangle(canvas, (left, top), (right, bottom), color, thickness)
                cv2.putText(
                    canvas,
                    f"T{track['track_id']}",
                    (left, max(12, top - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
            if pose is not None and observation["identity_state"] == "LOCKED":
                primitives = overlay_primitives(pose)
                for x1, y1, x2, y2 in primitives["lines"]:
                    cv2.line(
                        canvas,
                        (round(x1), round(y1)),
                        (round(x2), round(y2)),
                        (0, 200, 255),
                        2,
                    )
            label = (
                f"t={observation['timestamp_us'] / 1_000_000:.2f}s "
                f"state={observation['identity_state']} target={observation['target_id']} "
                f"track={active} conf={observation['identity_confidence']:.3f}"
            )
            cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            path = destination / f"frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(path), canvas):
                raise RuntimeError("DEBUG_OVERLAY_WRITE_FAILED")
            paths.append(str(path))
            tile_width = 360
            tile_height = max(1, round(canvas.shape[0] * tile_width / canvas.shape[1]))
            images.append(cv2.resize(canvas, (tile_width, tile_height)))
        contact = None
        if images:
            height = max(image.shape[0] for image in images)
            padded = [
                cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                for image in images
            ]
            rows = []
            for offset in range(0, len(padded), 3):
                row = padded[offset : offset + 3]
                while len(row) < 3:
                    row.append(np.zeros_like(padded[0]))
                rows.append(cv2.hconcat(row))
            contact = destination / "contact_sheet.jpg"
            cv2.imwrite(str(contact), cv2.vconcat(rows))
        return {
            "selected_frame_indices": indices,
            "overlay_paths": paths,
            "contact_sheet": str(contact) if contact else None,
        }
