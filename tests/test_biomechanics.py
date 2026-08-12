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
    assert knee_angle_2d(make_person(standard_points()), GEOMETRY) == pytest.approx(
        90.0, abs=1e-12
    )


def test_known_straight_leg_case() -> None:
    points = standard_points()
    points[Joint.LEFT_ANKLE] = (100, 300, 1)
    assert knee_angle_2d(make_person(points), GEOMETRY) == pytest.approx(
        180.0, abs=1e-12
    )


@pytest.mark.parametrize("missing", [Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE])
def test_missing_joint_returns_none(missing: Joint) -> None:
    points = standard_points()
    del points[missing]
    assert knee_angle_2d(make_person(points), GEOMETRY) is None


def test_low_confidence_returns_none() -> None:
    points = standard_points()
    points[Joint.LEFT_KNEE] = (100, 200, 0.49)
    assert knee_angle_2d(make_person(points), GEOMETRY) is None


def test_degenerate_vector_returns_none() -> None:
    points = standard_points()
    points[Joint.LEFT_HIP] = points[Joint.LEFT_KNEE]
    assert knee_angle_2d(make_person(points), GEOMETRY) is None
