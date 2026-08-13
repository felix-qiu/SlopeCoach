from __future__ import annotations

from pathlib import Path

from slopecoach_ml.video import OpenCVVideoSampler, inspect_video

from .ground_truth import GT_CONTRACT_VERSION, video_sha256


def prepare_target_gt_template(
    video_path: str | Path,
    *,
    sample_fps: float,
    output_path: str | Path,
    review_dir: str | Path | None = None,
) -> dict[str, object]:
    metadata = inspect_video(video_path)
    if not metadata.readable or metadata.width_px is None or metadata.height_px is None:
        raise ValueError("VIDEO_NOT_ANALYZABLE_FOR_GT_TEMPLATE")
    frames = []
    canonical_width = canonical_height = None
    review_images = []
    destination = Path(review_dir) if review_dir else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("GT_REVIEW_DEPENDENCY_MISSING: opencv-python") from error
    for sampled in OpenCVVideoSampler(video_path, sample_fps=sample_fps):
        canonical_width = sampled.geometry.width_px
        canonical_height = sampled.geometry.height_px
        frames.append(
            {
                "timestamp_us": sampled.timestamp_us,
                "frame_index": sampled.frame_index,
                "target_state": "UNLABELED",
                "bbox": None,
                "notes": None,
            }
        )
        if destination:
            canvas = sampled.image.copy()
            cv2.putText(
                canvas,
                f"HUMAN GT REVIEW t={sampled.timestamp_us / 1_000_000:.3f}s "
                f"frame={sampled.frame_index}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            path = destination / f"frame_{sampled.frame_index:06d}.jpg"
            if not cv2.imwrite(str(path), canvas):
                raise RuntimeError("GT_REVIEW_FRAME_WRITE_FAILED")
            review_images.append(canvas)
    payload = {
        "contract_version": GT_CONTRACT_VERSION,
        "video_sha256": video_sha256(video_path),
        "video_path_hint": Path(video_path).name,
        "coordinate_space": "SourcePixel2D",
        "annotation_source": "USER_MANUAL",
        "sample_fps": sample_fps,
        "width_px": canonical_width or metadata.width_px,
        "height_px": canonical_height or metadata.height_px,
        "duration_seconds": metadata.duration_seconds,
        "frames": frames,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    import json

    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    contact = None
    if destination and review_images:
        contact = _write_contact_pages(review_images, destination)
    return {
        "status": "TEMPLATE_CREATED_REQUIRES_HUMAN_LABELING",
        "template_path": str(output),
        "review_dir": str(destination) if destination else None,
        "contact_sheets": contact,
        "sampled_frame_count": len(frames),
        "video_sha256": payload["video_sha256"],
    }


def _write_contact_pages(images, destination: Path) -> list[str]:
    import cv2
    import numpy as np

    paths = []
    for page_number, offset in enumerate(range(0, len(images), 12), 1):
        page = images[offset : offset + 12]
        tiles = []
        for image in page:
            width = 240
            height = max(1, round(image.shape[0] * width / image.shape[1]))
            tiles.append(cv2.resize(image, (width, height)))
        row_height = max(tile.shape[0] for tile in tiles)
        padded = [
            cv2.copyMakeBorder(tile, 0, row_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            for tile in tiles
        ]
        rows = []
        for row_offset in range(0, len(padded), 3):
            row = padded[row_offset : row_offset + 3]
            while len(row) < 3:
                row.append(np.zeros_like(padded[0]))
            rows.append(cv2.hconcat(row))
        path = destination / f"contact_sheet_{page_number:02d}.jpg"
        if not cv2.imwrite(str(path), cv2.vconcat(rows)):
            raise RuntimeError("GT_REVIEW_CONTACT_SHEET_WRITE_FAILED")
        paths.append(str(path))
    return paths
