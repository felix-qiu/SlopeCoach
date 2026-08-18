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
            match = observation.get("latest_identity_match_score")
            observed_age = observation.get("last_observed_age_us")
            diagnostics = (f"match={match:.3f}" if match is not None else "match=null") + (
                f" observed_age_ms={observed_age / 1000:.0f}"
                if observed_age is not None
                else " observed_age_ms=null"
            )
            cv2.putText(
                canvas,
                diagnostics,
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
            )
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


def write_ground_truth_comparison(
    output_dir: str | Path,
    debug_artifacts: dict[str, Any],
    observations: list[dict[str, Any]],
    ground_truth,
    frame_classifications: list[dict[str, Any]],
) -> dict[str, Any]:
    """Add a separate GT/model comparison layer; never used to author Ground Truth."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("DEBUG_DEPENDENCY_MISSING: opencv-python") from error
    destination = Path(output_dir) / "gt_comparison"
    destination.mkdir(parents=True, exist_ok=True)
    observations_by_frame = {item["frame_index"]: item for item in observations}
    gt_by_timestamp = {item.timestamp_us: item for item in ground_truth.frames}
    classification_by_timestamp = {item["timestamp_us"]: item for item in frame_classifications}
    outputs = []
    contact_entries = []
    for source in debug_artifacts.get("overlay_paths", []):
        frame_index = int(Path(source).stem.split("_")[-1])
        observation = observations_by_frame.get(frame_index)
        if observation is None:
            continue
        classification = classification_by_timestamp.get(observation["timestamp_us"])
        if classification is None:
            continue
        gt_frame = gt_by_timestamp.get(classification["gt_timestamp_us"])
        image = cv2.imread(source)
        if image is None:
            continue
        if gt_frame and gt_frame.bbox:
            box = gt_frame.bbox
            cv2.rectangle(
                image,
                (round(box.x_px), round(box.y_px)),
                (round(box.x_px + box.width_px), round(box.y_px + box.height_px)),
                (255, 120, 0),
                3,
            )
            cv2.putText(
                image,
                "GT",
                (round(box.x_px), max(15, round(box.y_px) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 120, 0),
                2,
            )
        label = classification["classification"]
        iou = classification["selected_target_iou"]
        cv2.putText(
            image,
            f"GT RESULT={label} IoU={iou if iou is not None else 'null'}",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        output = destination / Path(source).name
        cv2.imwrite(str(output), image)
        outputs.append(str(output))
        priority = (
            0
            if label in {"WRONG_TARGET_LOCK", "TARGET_NOT_LOCKED", "FALSE_LOCK_WHEN_TARGET_ABSENT"}
            else 1
        )
        contact_entries.append((priority, image))
    contact = None
    if contact_entries:
        import numpy as np

        tiles = []
        for _, image in sorted(contact_entries, key=lambda item: item[0]):
            width = 360
            height = max(1, round(image.shape[0] * width / image.shape[1]))
            tiles.append(cv2.resize(image, (width, height)))
        row_height = max(tile.shape[0] for tile in tiles)
        rows = []
        for offset in range(0, len(tiles), 3):
            row = [
                cv2.copyMakeBorder(tile, 0, row_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                for tile in tiles[offset : offset + 3]
            ]
            while len(row) < 3:
                row.append(np.zeros_like(row[0]))
            rows.append(cv2.hconcat(row))
        contact = destination / "error_prioritized_contact_sheet.jpg"
        if not cv2.imwrite(str(contact), cv2.vconcat(rows)):
            raise RuntimeError("GT_COMPARISON_CONTACT_SHEET_WRITE_FAILED")
    return {
        "comparison_overlays": outputs,
        "gt_error_prioritized_contact_sheet": str(contact) if contact else None,
    }
