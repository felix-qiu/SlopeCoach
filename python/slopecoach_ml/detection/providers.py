from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from slopecoach_ml.pose import BoundingBox2D, FrameGeometry


@dataclass(frozen=True)
class Detection:
    detection_id: int
    bbox: BoundingBox2D
    confidence: float


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
