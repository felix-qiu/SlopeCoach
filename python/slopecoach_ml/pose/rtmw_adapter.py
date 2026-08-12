"""Verified COCO-WholeBody body-keypoint identity mapping for RTMW output.

The official COCO-WholeBody metainfo defines body keypoints 0..16 with COCO identities.
This named map intentionally avoids slicing and makes left/right parity testable.
"""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import Joint, Keypoint2D

RTMW_COCO_WHOLEBODY_INDEX_BY_JOINT: dict[Joint, int] = {
    Joint.NOSE: 0,
    Joint.LEFT_EYE: 1,
    Joint.RIGHT_EYE: 2,
    Joint.LEFT_EAR: 3,
    Joint.RIGHT_EAR: 4,
    Joint.LEFT_SHOULDER: 5,
    Joint.RIGHT_SHOULDER: 6,
    Joint.LEFT_ELBOW: 7,
    Joint.RIGHT_ELBOW: 8,
    Joint.LEFT_WRIST: 9,
    Joint.RIGHT_WRIST: 10,
    Joint.LEFT_HIP: 11,
    Joint.RIGHT_HIP: 12,
    Joint.LEFT_KNEE: 13,
    Joint.RIGHT_KNEE: 14,
    Joint.LEFT_ANKLE: 15,
    Joint.RIGHT_ANKLE: 16,
}


def map_rtmw_wholebody_to_coco17(
    coordinates: Sequence[Sequence[float]], scores: Sequence[float]
) -> dict[Joint, Keypoint2D]:
    if len(coordinates) < 133 or len(scores) < 133:
        raise ValueError("RTMW whole-body output must contain at least 133 keypoints and scores")
    result: dict[Joint, Keypoint2D] = {}
    for joint, index in RTMW_COCO_WHOLEBODY_INDEX_BY_JOINT.items():
        coordinate = coordinates[index]
        if len(coordinate) < 2:
            raise ValueError(f"RTMW keypoint {index} has no 2D coordinate")
        result[joint] = Keypoint2D(float(coordinate[0]), float(coordinate[1]), float(scores[index]))
    return result
