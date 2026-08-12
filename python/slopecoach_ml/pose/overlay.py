from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import Joint, PoseFrame

COCO17_EDGES = (
    (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    (Joint.LEFT_SHOULDER, Joint.LEFT_HIP),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_HIP),
    (Joint.LEFT_HIP, Joint.RIGHT_HIP),
    (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
)


def overlay_primitives(frame: PoseFrame, *, min_confidence: float = 0.3) -> dict[str, Any]:
    """Build drawing primitives exclusively from canonical SourcePixel2D output."""
    frame.validate()
    boxes = []
    points = []
    lines = []
    for person in frame.persons:
        boxes.append(
            (person.bbox.x_px, person.bbox.y_px, person.bbox.width_px, person.bbox.height_px)
        )
        for joint, point in person.keypoints.items():
            if point.confidence >= min_confidence:
                points.append((point.x_px, point.y_px, joint.value, point.confidence))
        for first, second in COCO17_EDGES:
            a, b = person.joint(first), person.joint(second)
            if a and b and a.confidence >= min_confidence and b.confidence >= min_confidence:
                lines.append((a.x_px, a.y_px, b.x_px, b.y_px))
    return {"boxes": boxes, "points": points, "lines": lines}


def render_debug_overlay(
    image: object,
    frame: PoseFrame,
    output: str | Path,
    *,
    provider_name: str,
    min_confidence: float = 0.3,
) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OPENMMLAB_DEPENDENCY_MISSING: opencv-python") from error
    canvas = image.copy()
    primitives = overlay_primitives(frame, min_confidence=min_confidence)
    for x, y, width, height in primitives["boxes"]:
        cv2.rectangle(
            canvas, (round(x), round(y)), (round(x + width), round(y + height)), (0, 255, 0), 2
        )
    for x1, y1, x2, y2 in primitives["lines"]:
        cv2.line(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (0, 200, 255), 2)
    for x, y, _, confidence in primitives["points"]:
        cv2.circle(canvas, (round(x), round(y)), 3, (255, 0, 255), -1)
        cv2.putText(
            canvas,
            f"{confidence:.2f}",
            (round(x) + 3, round(y) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )
    cv2.putText(
        canvas,
        f"{provider_name} t={frame.timestamp_us}us",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError("DEBUG_OVERLAY_WRITE_FAILED")
