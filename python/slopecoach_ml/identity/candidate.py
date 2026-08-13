from __future__ import annotations

import math

from slopecoach_ml.detection import Detection
from slopecoach_ml.pose import FrameGeometry

from .contracts import CandidateFilterConfig, PersonCandidate


def evaluate_candidates(
    detections: tuple[Detection, ...],
    geometry: FrameGeometry,
    config: CandidateFilterConfig | None = None,
) -> tuple[PersonCandidate, ...]:
    settings = config or CandidateFilterConfig()
    settings.validate()
    frame_area = geometry.width_px * geometry.height_px
    results = []
    for detection in detections:
        reason = None
        try:
            detection.validate(geometry)
            visible = detection.bbox.visible_fraction(geometry)
            area_fraction = detection.bbox.width_px * detection.bbox.height_px / frame_area
            aspect = detection.bbox.width_px / detection.bbox.height_px
        except (TypeError, ValueError):
            visible, area_fraction, aspect = 0.0, 0.0, 0.0
            reason = "INVALID_GEOMETRY"
        if reason is None and detection.confidence < settings.minimum_detection_confidence:
            reason = "LOW_DETECTION_CONFIDENCE"
        elif reason is None and (
            detection.bbox.width_px < settings.minimum_width_px
            or detection.bbox.height_px < settings.minimum_height_px
            or area_fraction < settings.minimum_area_fraction
        ):
            reason = "TOO_SMALL"
        elif reason is None and visible < settings.minimum_visible_fraction:
            reason = "NEARLY_INVISIBLE"
        elif reason is None and not (
            settings.plausible_aspect_ratio_min <= aspect <= settings.plausible_aspect_ratio_max
        ):
            reason = "IMPLAUSIBLE_ASPECT_RATIO"
        size_score = min(1.0, math.sqrt(max(0.0, area_fraction) / 0.08))
        aspect_score = math.exp(-abs(math.log(max(aspect, 1e-9) / 0.45)))
        evidence = {
            "detection_confidence": detection.confidence,
            "area_fraction": area_fraction,
            "visible_fraction": visible,
            "aspect_plausibility": aspect_score,
        }
        quality = max(
            0.0,
            min(
                1.0,
                0.45 * detection.confidence
                + 0.2 * size_score
                + 0.2 * visible
                + 0.15 * aspect_score,
            ),
        )
        results.append(
            PersonCandidate(
                detection.detection_id,
                detection.bbox,
                detection.confidence,
                quality,
                evidence,
                reason,
            )
        )
    return tuple(results)
