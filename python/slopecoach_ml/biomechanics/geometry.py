"""Finite-checked geometry helpers for canonical SourcePixel2D coordinates."""

from __future__ import annotations

import math

Point2D = tuple[float, float]


def _finite(*points: Point2D) -> None:
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("geometry coordinates must be finite")


def midpoint_2d(left: Point2D, right: Point2D) -> Point2D:
    _finite(left, right)
    return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)


def distance_2d(left: Point2D, right: Point2D) -> float:
    _finite(left, right)
    return math.hypot(right[0] - left[0], right[1] - left[1])


def angle_three_points_2d(first: Point2D, vertex: Point2D, third: Point2D) -> float | None:
    _finite(first, vertex, third)
    a = (first[0] - vertex[0], first[1] - vertex[1])
    b = (third[0] - vertex[0], third[1] - vertex[1])
    denominator = math.hypot(*a) * math.hypot(*b)
    if denominator <= 1e-12:
        return None
    cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / denominator))
    return math.degrees(math.acos(cosine))


def signed_screen_angle_from_vertical(start: Point2D, end: Point2D) -> float | None:
    """Positive means the downward vector leans toward screen-right."""
    _finite(start, end)
    dx, dy = end[0] - start[0], end[1] - start[1]
    if math.hypot(dx, dy) <= 1e-12:
        return None
    return math.degrees(math.atan2(dx, dy))


def undirected_axis_difference_deg(
    first_start: Point2D,
    first_end: Point2D,
    second_start: Point2D,
    second_end: Point2D,
) -> float | None:
    _finite(first_start, first_end, second_start, second_end)
    first = math.degrees(math.atan2(first_end[1] - first_start[1], first_end[0] - first_start[0]))
    second = math.degrees(
        math.atan2(second_end[1] - second_start[1], second_end[0] - second_start[0])
    )
    if (
        distance_2d(first_start, first_end) <= 1e-12
        or distance_2d(second_start, second_end) <= 1e-12
    ):
        return None
    difference = abs(first - second) % 180
    return min(difference, 180 - difference)


def normalized_screen_x_offset(first: Point2D, second: Point2D, scale: float) -> float | None:
    _finite(first, second)
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    return (first[0] - second[0]) / scale
