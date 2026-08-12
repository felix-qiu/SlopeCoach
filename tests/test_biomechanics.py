from __future__ import annotations

import pytest

from slopecoach_ml.biomechanics import knee_angle_2d
from slopecoach_ml.pose import (
    BoundingBox2D,
    FrameGeometry,
    Joint,
    Keypoint2D,
    PersonPose2D,
)


GEOMETRY = FrameGeometry(500, 500)
ARGS = {"minimum_confidence": 0.5, "square_pixel_tolerance": 1e-6}


def make_person(points: dict[Joint, tuple[float, float, float]]) -> PersonPose2D:
    return PersonPose2D(
        bbox=BoundingBox2D(0, 0, 500, 500),
        person_confidence=1.0,
        keypoints={joint: Keypoint2D(*values) for joint, values in points.items()},
    )


def standard_points() -> dict[Joint, tuple[float, float, float]]:
    return {
        Joint.LEFT_HIP: (100, 100, 1),
        Joint.LEFT_KNEE: (100, 200, 1),
        Joint.LEFT_ANKLE: (200, 200, 1),
    }


def test_known_90_degree_case() -> None:
    assert knee_angle_2d(
        make_person(standard_points()), GEOMETRY, **ARGS
    ) == pytest.approx(90.0, abs=1e-12)


def test_known_straight_leg_case() -> None:
    points = standard_points()
    points[Joint.LEFT_ANKLE] = (100, 300, 1)
    assert knee_angle_2d(make_person(points), GEOMETRY, **ARGS) == pytest.approx(
        180.0, abs=1e-12
    )


@pytest.mark.parametrize("missing", [Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE])
def test_missing_joint_returns_none(missing: Joint) -> None:
    points = standard_points()
    del points[missing]
    assert knee_angle_2d(make_person(points), GEOMETRY, **ARGS) is None


def test_low_confidence_returns_none() -> None:
    points = standard_points()
    points[Joint.LEFT_KNEE] = (100, 200, 0.49)
    assert knee_angle_2d(make_person(points), GEOMETRY, **ARGS) is None


def test_degenerate_vector_returns_none() -> None:
    points = standard_points()
    points[Joint.LEFT_HIP] = points[Joint.LEFT_KNEE]
    assert knee_angle_2d(make_person(points), GEOMETRY, **ARGS) is None


def test_near_square_pixels_are_supported() -> None:
    near_square = FrameGeometry(500, 500, pixel_aspect_ratio=1.0000005)
    assert knee_angle_2d(
        make_person(standard_points()), near_square, **ARGS
    ) == pytest.approx(90)


def test_non_square_pixels_are_valid_but_unsupported() -> None:
    non_square = FrameGeometry(500, 500, pixel_aspect_ratio=1.2)
    non_square.validate()
    assert knee_angle_2d(make_person(standard_points()), non_square, **ARGS) is None


def test_unrelated_out_of_frame_joint_does_not_affect_angle() -> None:
    points = standard_points()
    points[Joint.LEFT_WRIST] = (-20, 700, 0.9)
    assert knee_angle_2d(make_person(points), GEOMETRY, **ARGS) == pytest.approx(90)


def test_required_out_of_frame_joint_returns_none_without_clamping() -> None:
    points = standard_points()
    points[Joint.LEFT_KNEE] = (-1, 200, 1)
    pose = make_person(points)
    assert pose.joint(Joint.LEFT_KNEE).x_px == -1
    assert knee_angle_2d(pose, GEOMETRY, **ARGS) is None
