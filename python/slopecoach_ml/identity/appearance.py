from __future__ import annotations

import math
from collections.abc import Sequence

from slopecoach_ml.pose import BoundingBox2D


def clipped_crop_bounds(
    image_width: int,
    image_height: int,
    bbox: BoundingBox2D,
    *,
    minimum_crop_pixels: int = 64,
) -> tuple[int, int, int, int] | None:
    left = max(0, min(image_width, math.floor(bbox.x_px)))
    top = max(0, min(image_height, math.floor(bbox.y_px)))
    right = max(0, min(image_width, math.ceil(bbox.x_px + bbox.width_px)))
    bottom = max(0, min(image_height, math.ceil(bbox.y_px + bbox.height_px)))
    if right <= left or bottom <= top or (right - left) * (bottom - top) < minimum_crop_pixels:
        return None
    return left, top, right, bottom


def descriptor_similarity(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or not first:
        return None
    if not all(math.isfinite(value) and value >= 0 for value in (*first, *second)):
        return None
    norm_first = math.sqrt(sum(value * value for value in first))
    norm_second = math.sqrt(sum(value * value for value in second))
    if norm_first == 0 or norm_second == 0:
        return None
    return max(
        0.0,
        min(
            1.0, sum(a * b for a, b in zip(first, second, strict=True)) / (norm_first * norm_second)
        ),
    )


def update_gallery(
    gallery: list[Sequence[float]],
    descriptor: Sequence[float] | None,
    *,
    quality: float,
    minimum_quality: float = 0.5,
    maximum_length: int = 8,
) -> None:
    if descriptor is None or quality < minimum_quality:
        return
    gallery.append(tuple(float(value) for value in descriptor))
    del gallery[:-maximum_length]


class HSVHistogramAppearanceEncoder:
    """Lightweight color evidence, not a deep ReID model."""

    def __init__(self, *, bins: tuple[int, int] = (12, 8), minimum_crop_pixels: int = 64) -> None:
        self.bins = bins
        self.minimum_crop_pixels = minimum_crop_pixels

    def encode(self, image: object, bbox: BoundingBox2D) -> Sequence[float] | None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("APPEARANCE_DEPENDENCY_MISSING: opencv-python") from error
        height, width = image.shape[:2]
        bounds = clipped_crop_bounds(
            width, height, bbox, minimum_crop_pixels=self.minimum_crop_pixels
        )
        if bounds is None:
            return None
        left, top, right, bottom = bounds
        crop = image[top:bottom, left:right]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, list(self.bins), [0, 180, 0, 256])
        total = float(histogram.sum())
        if not math.isfinite(total) or total <= 0:
            return None
        return tuple(float(value) for value in (histogram / total).reshape(-1))
