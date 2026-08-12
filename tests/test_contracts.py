from __future__ import annotations

import math

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
