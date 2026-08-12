"""Provisional Python pose contracts for research and Rust parity fixtures.

These types are not the production cross-language source of truth. That role is
reserved for the future Rust contracts crate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class CoordinateSpace(StrEnum):
    SOURCE_PIXEL_2D = "SourcePixel2D"


class FrameOrientation(StrEnum):
    CANONICAL_UPRIGHT = "CanonicalUpright"


class Joint(StrEnum):
    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


COCO17_V1 = "COCO17_V1"
COCO17_JOINTS = frozenset(Joint)


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class FrameGeometry:
    width_px: int
    height_px: int
    pixel_aspect_ratio: float = 1.0
    coordinate_space: CoordinateSpace = CoordinateSpace.SOURCE_PIXEL_2D
    orientation: FrameOrientation = FrameOrientation.CANONICAL_UPRIGHT
    mirrored: bool = False

    def validate(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("frame dimensions must be positive")
        _finite(self.pixel_aspect_ratio, "pixel_aspect_ratio")
        if self.pixel_aspect_ratio <= 0:
            raise ValueError("pixel_aspect_ratio must be positive")
        if self.coordinate_space is not CoordinateSpace.SOURCE_PIXEL_2D:
            raise ValueError("biomechanics requires SourcePixel2D coordinates")
        if self.orientation is not FrameOrientation.CANONICAL_UPRIGHT:
            raise ValueError("frame orientation must be CanonicalUpright")
        if self.mirrored:
            raise ValueError("mirroring must be corrected before the canonical boundary")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameGeometry:
        geometry = cls(
            width_px=int(data["width_px"]),
            height_px=int(data["height_px"]),
            pixel_aspect_ratio=float(data.get("pixel_aspect_ratio", 1.0)),
            coordinate_space=CoordinateSpace(data["coordinate_space"]),
            orientation=FrameOrientation(data["orientation"]),
            mirrored=bool(data["mirrored"]),
        )
        geometry.validate()
        return geometry

    def to_dict(self) -> dict[str, Any]:
        return {
            "width_px": self.width_px,
            "height_px": self.height_px,
            "pixel_aspect_ratio": self.pixel_aspect_ratio,
            "coordinate_space": self.coordinate_space.value,
            "orientation": self.orientation.value,
            "mirrored": self.mirrored,
        }


@dataclass(frozen=True)
class Keypoint2D:
    x_px: float
    y_px: float
    confidence: float
    coordinate_space: CoordinateSpace = CoordinateSpace.SOURCE_PIXEL_2D

    def validate(self, geometry: FrameGeometry) -> None:
        for value, name in (
            (self.x_px, "x_px"),
            (self.y_px, "y_px"),
            (self.confidence, "confidence"),
        ):
            _finite(value, name)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("keypoint confidence must be in [0, 1]")
        if self.coordinate_space is not geometry.coordinate_space:
            raise ValueError("keypoint and frame coordinate spaces differ")
        if not 0.0 <= self.x_px < geometry.width_px or not 0.0 <= self.y_px < geometry.height_px:
            raise ValueError("keypoint is outside source-frame bounds")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Keypoint2D:
        return cls(
            x_px=float(data["x_px"]),
            y_px=float(data["y_px"]),
            confidence=float(data["confidence"]),
            coordinate_space=CoordinateSpace(data.get("coordinate_space", "SourcePixel2D")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_px": self.x_px,
            "y_px": self.y_px,
            "confidence": self.confidence,
            "coordinate_space": self.coordinate_space.value,
        }


@dataclass(frozen=True)
class BoundingBox2D:
    x_px: float
    y_px: float
    width_px: float
    height_px: float
    coordinate_space: CoordinateSpace = CoordinateSpace.SOURCE_PIXEL_2D

    def validate(self, geometry: FrameGeometry) -> None:
        for value, name in (
            (self.x_px, "bbox.x_px"),
            (self.y_px, "bbox.y_px"),
            (self.width_px, "bbox.width_px"),
            (self.height_px, "bbox.height_px"),
        ):
            _finite(value, name)
        if self.coordinate_space is not geometry.coordinate_space:
            raise ValueError("bbox and frame coordinate spaces differ")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("bbox dimensions must be positive")
        if self.x_px < 0 or self.y_px < 0:
            raise ValueError("bbox origin is outside source-frame bounds")
        if (
            self.x_px + self.width_px > geometry.width_px
            or self.y_px + self.height_px > geometry.height_px
        ):
            raise ValueError("bbox extends outside source-frame bounds")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundingBox2D:
        return cls(
            x_px=float(data["x_px"]),
            y_px=float(data["y_px"]),
            width_px=float(data["width_px"]),
            height_px=float(data["height_px"]),
            coordinate_space=CoordinateSpace(data.get("coordinate_space", "SourcePixel2D")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_px": self.x_px,
            "y_px": self.y_px,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "coordinate_space": self.coordinate_space.value,
        }


@dataclass(frozen=True)
class PersonPose2D:
    bbox: BoundingBox2D
    person_confidence: float
    keypoints: dict[Joint, Keypoint2D]
    detection_id: int | None = None

    def joint(self, joint: Joint) -> Keypoint2D | None:
        return self.keypoints.get(joint)

    def validate(self, geometry: FrameGeometry) -> None:
        _finite(self.person_confidence, "person_confidence")
        if not 0.0 <= self.person_confidence <= 1.0:
            raise ValueError("person confidence must be in [0, 1]")
        self.bbox.validate(geometry)
        for joint, keypoint in self.keypoints.items():
            if joint not in COCO17_JOINTS:
                raise ValueError(f"joint {joint!r} is not part of COCO17_V1")
            keypoint.validate(geometry)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonPose2D:
        raw_keypoints = data["keypoints"]
        if not isinstance(raw_keypoints, dict):
            raise ValueError("keypoints must be an object keyed by canonical joint identity")
        try:
            keypoints = {
                Joint(name): Keypoint2D.from_dict(value) for name, value in raw_keypoints.items()
            }
        except (TypeError, KeyError) as error:
            raise ValueError("malformed keypoint data") from error
        return cls(
            detection_id=data.get("detection_id"),
            bbox=BoundingBox2D.from_dict(data["bbox"]),
            person_confidence=float(data["person_confidence"]),
            keypoints=keypoints,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "bbox": self.bbox.to_dict(),
            "person_confidence": self.person_confidence,
            "keypoints": {joint.value: point.to_dict() for joint, point in self.keypoints.items()},
        }


@dataclass(frozen=True)
class PoseFrame:
    contract_version: str
    timestamp_us: int
    frame_index: int
    geometry: FrameGeometry
    joint_schema: str
    persons: tuple[PersonPose2D, ...]

    def validate(self) -> None:
        self.geometry.validate()
        if self.joint_schema != COCO17_V1:
            raise ValueError(f"unsupported joint schema: {self.joint_schema}")
        if self.timestamp_us < 0 or self.frame_index < 0:
            raise ValueError("timestamp and frame index must be non-negative")
        for person in self.persons:
            person.validate(self.geometry)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoseFrame:
        frame = cls(
            contract_version=str(data["contract_version"]),
            timestamp_us=int(data["timestamp_us"]),
            frame_index=int(data.get("frame_index", 0)),
            geometry=FrameGeometry.from_dict(data["frame_geometry"]),
            joint_schema=str(data["joint_schema"]),
            persons=tuple(PersonPose2D.from_dict(person) for person in data["persons"]),
        )
        frame.validate()
        return frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "timestamp_us": self.timestamp_us,
            "frame_index": self.frame_index,
            "frame_geometry": self.geometry.to_dict(),
            "joint_schema": self.joint_schema,
            "persons": [person.to_dict() for person in self.persons],
        }


def adapt_model_schema_to_coco17(*_: object, **__: object) -> PoseFrame:
    """Explicit adapter boundary; real model mappings are intentionally not implemented."""
    raise NotImplementedError("a concrete model-to-COCO17 adapter is not configured")
