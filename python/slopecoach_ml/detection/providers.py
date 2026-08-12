from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from slopecoach_ml.pose import BoundingBox2D, FrameGeometry


@dataclass(frozen=True)
class Detection:
    detection_id: int
    bbox: BoundingBox2D
    confidence: float

    def validate(self, geometry: FrameGeometry) -> None:
        if isinstance(self.detection_id, bool) or not isinstance(self.detection_id, int):
            raise TypeError("detection_id must be an integer")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be in [0, 1]")
        self.bbox.validate(geometry)


class DetectorProvider(Protocol):
    name: str

    def detect(self, frame: object, geometry: FrameGeometry) -> tuple[Detection, ...]: ...


class MockDetectorProvider:
    name = "mock-detector"

    def __init__(self, detections: tuple[Detection, ...] = ()) -> None:
        self._detections = detections

    def detect(self, frame: object, geometry: FrameGeometry) -> tuple[Detection, ...]:
        geometry.validate()
        return self._detections
