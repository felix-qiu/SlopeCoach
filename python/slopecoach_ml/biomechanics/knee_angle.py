from __future__ import annotations

import math

from slopecoach_ml.pose import FrameGeometry, Joint, PersonPose2D


def knee_angle_2d(
    person: PersonPose2D,
    geometry: FrameGeometry,
    *,
    side: str = "left",
    minimum_confidence: float,
    square_pixel_tolerance: float,
) -> float | None:
    """Return the image-plane hip-knee-ankle angle, or None if evidence is insufficient."""
    geometry.validate()
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum confidence must be in [0, 1]")
    if not math.isfinite(square_pixel_tolerance) or square_pixel_tolerance < 0:
        raise ValueError("square pixel tolerance must be finite and non-negative")
    if not math.isclose(
        geometry.pixel_aspect_ratio, 1.0, rel_tol=0.0, abs_tol=square_pixel_tolerance
    ):
        return None
    person.validate(geometry)
    joints = {
        "left": (Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
        "right": (Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
    }[side]
    hip, knee, ankle = (person.joint(joint) for joint in joints)
    if hip is None or knee is None or ankle is None:
        return None
    if min(hip.confidence, knee.confidence, ankle.confidence) < minimum_confidence:
        return None
    if not all(point.is_inside_frame(geometry) for point in (hip, knee, ankle)):
        return None
    first = (hip.x_px - knee.x_px, hip.y_px - knee.y_px)
    second = (ankle.x_px - knee.x_px, ankle.y_px - knee.y_px)
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length == 0.0 or second_length == 0.0:
        return None
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_length * second_length)
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.degrees(math.acos(cosine))
    return angle if math.isfinite(angle) else None
