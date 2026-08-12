from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from slopecoach_ml.pose import BoundingBox2D, FrameGeometry

from .providers import Detection


class MMDetBackend(Protocol):
    def infer(self, image: object) -> Sequence[tuple[float, float, float, float, float, int]]: ...


@dataclass(frozen=True)
class MMDetProviderConfig:
    score_threshold: float = 0.3
    person_label: int = 0


class MMDetPersonDetectorProvider:
    """Research MMDetection person provider with a lazy, injectable backend."""

    name = "openmmlab-rtmdet-m-640-coco-obj365-person"

    def __init__(self, backend: MMDetBackend, config: MMDetProviderConfig | None = None) -> None:
        self._backend = backend
        self.config = config or MMDetProviderConfig()
        if not 0.0 <= self.config.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")

    def detect(self, frame: object, geometry: FrameGeometry) -> tuple[Detection, ...]:
        geometry.validate()
        detections: list[Detection] = []
        for x1, y1, x2, y2, score, label in self._backend.infer(frame):
            if label != self.config.person_label or score < self.config.score_threshold:
                continue
            detection = Detection(
                detection_id=len(detections),
                bbox=BoundingBox2D(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                confidence=float(score),
            )
            detection.validate(geometry)
            detections.append(detection)
        return tuple(detections)


class OpenMMLabMMDetBackend:
    """Adapter over official MMDetection APIs; imports only when explicitly configured."""

    def __init__(self, config_path: str, checkpoint_path: str, *, device: str = "cpu") -> None:
        try:
            from mmdet.apis import inference_detector, init_detector
        except ImportError as error:
            raise RuntimeError("OPENMMLAB_DEPENDENCY_MISSING: mmdet") from error
        try:
            self._model = init_detector(config_path, checkpoint_path, device=device)
        except Exception as error:
            raise RuntimeError(f"MODEL_LOAD_FAILED: detector: {error}") from error
        self._infer: Callable[[Any, Any], Any] = inference_detector

    def infer(self, image: object) -> Sequence[tuple[float, float, float, float, float, int]]:
        try:
            instances = self._infer(self._model, image).pred_instances.cpu()
            boxes = instances.bboxes.numpy().tolist()
            scores = instances.scores.numpy().tolist()
            labels = instances.labels.numpy().tolist()
        except Exception as error:
            raise RuntimeError(f"POSE_INFERENCE_FAILED: detector: {error}") from error
        return [
            (*box, score, label) for box, score, label in zip(boxes, scores, labels, strict=True)
        ]
