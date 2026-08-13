from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

from slopecoach_ml.detection.providers import Detection

from .contracts import COCO17_V1, FrameGeometry, PersonPose2D, PoseFrame
from .rtmw_adapter import map_rtmw_wholebody_to_coco17


class MMPoseBackend(Protocol):
    def infer(
        self, image: object, bboxes_xyxy: Sequence[tuple[float, float, float, float]]
    ) -> Sequence[tuple[Sequence[Sequence[float]], Sequence[float]]]: ...


class MMPoseRTMWPoseProvider:
    """Real-provider boundary for RTMW-L; official API outputs original-image coordinates.

    MMPose `inference_topdown` decodes SimCC and transforms predictions back with each
    data sample's input center/scale. Therefore coordinates at this boundary already align
    with the canonical-upright input image and must not be inverse-transformed again.
    """

    name = "openmmlab-rtmw-l-cocktail14-256x192"

    def __init__(self, backend: MMPoseBackend) -> None:
        self._backend = backend
        self.last_backend_seconds = 0.0
        self.last_adapter_seconds = 0.0
        self.last_raw_joint_count: int | None = None

    def estimate_detections(
        self,
        image: object,
        detections: Sequence[Detection],
        geometry: FrameGeometry,
        *,
        timestamp_us: int,
        frame_index: int,
    ) -> PoseFrame:
        geometry.validate()
        for detection in detections:
            detection.validate(geometry)
        boxes = [
            (
                detection.bbox.x_px,
                detection.bbox.y_px,
                detection.bbox.x_px + detection.bbox.width_px,
                detection.bbox.y_px + detection.bbox.height_px,
            )
            for detection in detections
        ]
        started = time.perf_counter()
        predictions = self._backend.infer(image, boxes) if boxes else ()
        self.last_backend_seconds = time.perf_counter() - started
        if len(predictions) != len(detections):
            raise RuntimeError("POSE_INFERENCE_FAILED: prediction/detection count mismatch")
        self.last_raw_joint_count = len(predictions[0][0]) if predictions else None
        started = time.perf_counter()
        persons = tuple(
            PersonPose2D(
                detection_id=detection.detection_id,
                bbox=detection.bbox,
                person_confidence=detection.confidence,
                keypoints=map_rtmw_wholebody_to_coco17(coordinates, scores),
            )
            for detection, (coordinates, scores) in zip(detections, predictions, strict=True)
        )
        self.last_adapter_seconds = time.perf_counter() - started
        frame = PoseFrame(
            contract_version="python-reference-pose-v1",
            timestamp_us=timestamp_us,
            frame_index=frame_index,
            geometry=geometry,
            joint_schema=COCO17_V1,
            persons=persons,
        )
        frame.validate()
        return frame


class OpenMMLabMMPoseBackend:
    def __init__(self, config_path: str, checkpoint_path: str, *, device: str = "cpu") -> None:
        self.device = device
        try:
            from mmpose.apis import inference_topdown, init_model
        except ImportError as error:
            raise RuntimeError("OPENMMLAB_DEPENDENCY_MISSING: mmpose") from error
        try:
            self._model = init_model(config_path, checkpoint_path, device=device)
        except Exception as error:
            raise RuntimeError(f"MODEL_LOAD_FAILED: pose: {error}") from error
        self._infer = inference_topdown

    def infer(
        self, image: object, bboxes_xyxy: Sequence[tuple[float, float, float, float]]
    ) -> Sequence[tuple[Sequence[Sequence[float]], Sequence[float]]]:
        try:
            results = self._infer(self._model, image, bboxes=list(bboxes_xyxy), bbox_format="xyxy")
            return [
                (
                    result.pred_instances.keypoints[0].tolist(),
                    result.pred_instances.keypoint_scores[0].tolist(),
                )
                for result in results
            ]
        except Exception as error:
            raise RuntimeError(f"POSE_INFERENCE_FAILED: pose: {error}") from error
