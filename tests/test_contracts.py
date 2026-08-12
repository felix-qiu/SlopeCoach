from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from slopecoach_ml.pose import (
    COCO17_V1,
    BoundingBox2D,
    FrameGeometry,
    Joint,
    Keypoint2D,
    PersonPose2D,
    PoseFrame,
)


def geometry() -> FrameGeometry:
    return FrameGeometry(640, 480)


def person(**keypoints: Keypoint2D) -> PersonPose2D:
    return PersonPose2D(
        bbox=BoundingBox2D(0, 0, 400, 400),
        person_confidence=0.9,
        keypoints={Joint(name): point for name, point in keypoints.items()},
    )


def test_valid_source_pixel_2d_contract() -> None:
    pose = person(left_knee=Keypoint2D(100, 200, 0.9))
    pose.validate(geometry())
    assert pose.joint(Joint.LEFT_KNEE) == Keypoint2D(100, 200, 0.9)
    assert pose.joint(Joint.RIGHT_KNEE) is None


@pytest.mark.parametrize("width,height", [(0, 480), (640, -1)])
def test_invalid_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        FrameGeometry(width, height).validate()


def test_invalid_orientation_state() -> None:
    with pytest.raises(ValueError):
        FrameGeometry.from_dict(
            {
                "width_px": 640,
                "height_px": 480,
                "coordinate_space": "SourcePixel2D",
                "orientation": "Rotated90",
                "mirrored": False,
            }
        )


def test_mirrored_boundary_is_rejected() -> None:
    with pytest.raises(ValueError, match="mirroring"):
        FrameGeometry(640, 480, mirrored=True).validate()


def test_bbox_keypoint_coordinate_consistency() -> None:
    invalid_bbox = BoundingBox2D(0, 0, 100, 100, coordinate_space="ModelCoordinate")  # type: ignore[arg-type]
    pose = PersonPose2D(invalid_bbox, 0.9, {Joint.LEFT_HIP: Keypoint2D(20, 20, 0.9)})
    with pytest.raises(ValueError, match="coordinate spaces"):
        pose.validate(geometry())


def test_non_finite_keypoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        Keypoint2D(math.nan, 20, 0.9).validate(geometry())


@pytest.mark.parametrize(
    "field,value",
    [
        ("width_px", "640"),
        ("width_px", True),
        ("height_px", 480.0),
        ("mirrored", "false"),
    ],
)
def test_strict_geometry_parsing_rejects_coercion(field: str, value: object) -> None:
    data = {
        "width_px": 640,
        "height_px": 480,
        "pixel_aspect_ratio": 1.0,
        "coordinate_space": "SourcePixel2D",
        "orientation": "CanonicalUpright",
        "mirrored": False,
    }
    data[field] = value
    with pytest.raises((TypeError, ValueError)):
        FrameGeometry.from_dict(data)


@pytest.mark.parametrize("value", ["1.0", True, math.nan, math.inf])
def test_strict_numeric_keypoint_parsing(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Keypoint2D.from_dict({"x_px": value, "y_px": 2, "confidence": 0.9})


@pytest.mark.parametrize("value", ["0.9", True, math.nan, 1.1])
def test_malformed_confidence_rejected(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        point = Keypoint2D.from_dict({"x_px": 1, "y_px": 2, "confidence": value})
        point.validate(geometry())


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_invalid_pixel_aspect_ratio_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        FrameGeometry(640, 480, pixel_aspect_ratio=value).validate()


def test_out_of_frame_coordinates_are_preserved_and_visible_separately() -> None:
    negative = Keypoint2D(-3.5, 20, 0.9)
    beyond = Keypoint2D(700, 20, 0.9)
    negative.validate(geometry())
    beyond.validate(geometry())
    assert negative.x_px == -3.5
    assert not negative.is_inside_frame(geometry())
    assert not beyond.is_inside_frame(geometry())


def test_partially_outside_bbox_has_visibility_without_clamping() -> None:
    bbox = BoundingBox2D(-10, 20, 100, 100)
    bbox.validate(geometry())
    assert bbox.x_px == -10
    assert bbox.intersects_frame(geometry())
    assert bbox.visible_fraction(geometry()) == pytest.approx(0.9)
    outside = BoundingBox2D(-200, 20, 50, 50)
    assert outside.visible_fraction(geometry()) == 0


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_bbox_size_rejected(size: float) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        BoundingBox2D(0, 0, size, 10).validate(geometry())


@pytest.mark.parametrize("field", ["timestamp_us", "frame_index"])
def test_string_integer_frame_fields_rejected(field: str) -> None:
    data = json.loads(
        (Path(__file__).parents[1] / "fixtures/golden_pose_001.json").read_text()
    )["pose_frame"]
    data[field] = "1000"
    with pytest.raises(TypeError):
        PoseFrame.from_dict(data)


def test_coco17_schema_and_lookup() -> None:
    frame = PoseFrame(
        "reference-v1",
        0,
        0,
        geometry(),
        COCO17_V1,
        (person(left_hip=Keypoint2D(10, 10, 1)),),
    )
    frame.validate()
    assert frame.persons[0].joint(Joint.LEFT_HIP) is not None


def test_malformed_joint_data_fails() -> None:
    with pytest.raises((ValueError, TypeError, KeyError)):
        PersonPose2D.from_dict(
            {
                "bbox": {"x_px": 0, "y_px": 0, "width_px": 10, "height_px": 10},
                "person_confidence": 1,
                "keypoints": {"not_a_joint": {"x_px": 1, "y_px": 1, "confidence": 1}},
            }
        )
