"""Deterministic reference geometry transforms; no pixels or inference are handled here."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import CoordinateSpace, FrameGeometry, Keypoint2D


def _finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ModelPoint2D:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PreprocessTransform2D:
    """Recorded source crop, resize, and letterbox transform.

    Orientation and mirror correction are deliberately outside this reference adapter. The
    caller must provide canonical-upright, non-mirrored source geometry.
    """

    crop_x_px: float
    crop_y_px: float
    crop_width_px: float
    crop_height_px: float
    scale_x: float
    scale_y: float
    pad_left_px: float
    pad_top_px: float
    model_width_px: float
    model_height_px: float

    def validate(self, source: FrameGeometry) -> None:
        source.validate()
        for name, value in vars(self).items():
            _finite_number(value, name)
        if self.crop_x_px < 0 or self.crop_y_px < 0:
            raise ValueError("crop origin must be inside the source frame")
        if self.crop_width_px <= 0 or self.crop_height_px <= 0:
            raise ValueError("crop dimensions must be positive")
        if self.crop_x_px + self.crop_width_px > source.width_px:
            raise ValueError("crop exceeds source width")
        if self.crop_y_px + self.crop_height_px > source.height_px:
            raise ValueError("crop exceeds source height")
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("resize scales must be positive")
        if self.pad_left_px < 0 or self.pad_top_px < 0:
            raise ValueError("letterbox padding must be non-negative")
        if self.model_width_px <= 0 or self.model_height_px <= 0:
            raise ValueError("model dimensions must be positive")
        content_right = self.pad_left_px + self.crop_width_px * self.scale_x
        content_bottom = self.pad_top_px + self.crop_height_px * self.scale_y
        if content_right > self.model_width_px or content_bottom > self.model_height_px:
            raise ValueError("resized crop and padding do not fit model geometry")

    def forward(self, point: Keypoint2D, source: FrameGeometry) -> ModelPoint2D:
        self.validate(source)
        point.validate(source)
        return ModelPoint2D(
            x=(point.x_px - self.crop_x_px) * self.scale_x + self.pad_left_px,
            y=(point.y_px - self.crop_y_px) * self.scale_y + self.pad_top_px,
            confidence=point.confidence,
        )

    def inverse(self, point: ModelPoint2D, source: FrameGeometry) -> Keypoint2D:
        self.validate(source)
        for name, value in vars(point).items():
            _finite_number(value, f"model_point.{name}")
        if not 0.0 <= point.confidence <= 1.0:
            raise ValueError("model point confidence must be in [0, 1]")
        result = Keypoint2D(
            x_px=(point.x - self.pad_left_px) / self.scale_x + self.crop_x_px,
            y_px=(point.y - self.pad_top_px) / self.scale_y + self.crop_y_px,
            confidence=float(point.confidence),
            coordinate_space=CoordinateSpace.SOURCE_PIXEL_2D,
        )
        result.validate(source)
        return result
