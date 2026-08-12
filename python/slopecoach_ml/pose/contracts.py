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


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    _finite(value, name)
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _finite(value: int | float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
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
        _integer(self.width_px, "width_px")
        _integer(self.height_px, "height_px")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("frame dimensions must be positive")
        _finite(self.pixel_aspect_ratio, "pixel_aspect_ratio")
        if self.pixel_aspect_ratio <= 0:
            raise ValueError("pixel_aspect_ratio must be positive")
        if self.coordinate_space is not CoordinateSpace.SOURCE_PIXEL_2D:
            raise ValueError("biomechanics requires SourcePixel2D coordinates")
        if self.orientation is not FrameOrientation.CANONICAL_UPRIGHT:
            raise ValueError("frame orientation must be CanonicalUpright")
        _boolean(self.mirrored, "mirrored")
        if self.mirrored:
            raise ValueError("mirroring must be corrected before the canonical boundary")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameGeometry:
        geometry = cls(
            width_px=_integer(data["width_px"], "width_px"),
            height_px=_integer(data["height_px"], "height_px"),
            pixel_aspect_ratio=_number(data.get("pixel_aspect_ratio", 1.0), "pixel_aspect_ratio"),
            coordinate_space=CoordinateSpace(data["coordinate_space"]),
            orientation=FrameOrientation(data["orientation"]),
            mirrored=_boolean(data["mirrored"], "mirrored"),
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

    def is_inside_frame(self, geometry: FrameGeometry) -> bool:
        self.validate(geometry)
        return 0.0 <= self.x_px < geometry.width_px and 0.0 <= self.y_px < geometry.height_px

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Keypoint2D:
        return cls(
            x_px=_number(data["x_px"], "x_px"),
            y_px=_number(data["y_px"], "y_px"),
            confidence=_number(data["confidence"], "confidence"),
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

    def intersection_area(self, geometry: FrameGeometry) -> float:
        self.validate(geometry)
        left = max(0.0, self.x_px)
        top = max(0.0, self.y_px)
        right = min(float(geometry.width_px), self.x_px + self.width_px)
        bottom = min(float(geometry.height_px), self.y_px + self.height_px)
        return max(0.0, right - left) * max(0.0, bottom - top)

    def intersects_frame(self, geometry: FrameGeometry) -> bool:
        return self.intersection_area(geometry) > 0.0

    def visible_fraction(self, geometry: FrameGeometry) -> float:
        return self.intersection_area(geometry) / (self.width_px * self.height_px)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundingBox2D:
        return cls(
            x_px=_number(data["x_px"], "bbox.x_px"),
            y_px=_number(data["y_px"], "bbox.y_px"),
            width_px=_number(data["width_px"], "bbox.width_px"),
            height_px=_number(data["height_px"], "bbox.height_px"),
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
        detection_id = data.get("detection_id")
        if detection_id is not None:
            detection_id = _integer(detection_id, "detection_id")
        person = cls(
            detection_id=detection_id,
            bbox=BoundingBox2D.from_dict(data["bbox"]),
            person_confidence=_number(data["person_confidence"], "person_confidence"),
            keypoints=keypoints,
        )
        return person

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
        _integer(self.timestamp_us, "timestamp_us")
        _integer(self.frame_index, "frame_index")
        if self.joint_schema != COCO17_V1:
            raise ValueError(f"unsupported joint schema: {self.joint_schema}")
        if self.timestamp_us < 0 or self.frame_index < 0:
            raise ValueError("timestamp and frame index must be non-negative")
        for person in self.persons:
            person.validate(self.geometry)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoseFrame:
        persons = data["persons"]
        if not isinstance(persons, list):
            raise TypeError("persons must be an array")
        frame = cls(
            contract_version=_string(data["contract_version"], "contract_version"),
            timestamp_us=_integer(data["timestamp_us"], "timestamp_us"),
            frame_index=_integer(data.get("frame_index", 0), "frame_index"),
            geometry=FrameGeometry.from_dict(data["frame_geometry"]),
            joint_schema=_string(data["joint_schema"], "joint_schema"),
            persons=tuple(PersonPose2D.from_dict(person) for person in persons),
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
