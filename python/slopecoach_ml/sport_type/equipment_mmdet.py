"""Target-gated full-COCO RTMDet equipment evidence for A6.1 research."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Protocol

from slopecoach_ml.pose import BoundingBox2D

from .contracts import (
    SportEvidenceKind,
    SportEvidenceObservation,
    SportEvidenceProviderResult,
    SportEvidenceProviderStatus,
    SportEvidenceScope,
)

EQUIPMENT_PROVIDER_NAME = "openmmlab-rtmdet-tiny-coco-equipment"
EQUIPMENT_CONFIG_PROFILE = "RESEARCH_DEFAULTS_A6_1"
MINIMUM_P95_SAMPLES = 20


class EquipmentDetectorBackend(Protocol):
    class_names: tuple[str, ...]

    def infer(self, image: object) -> tuple[tuple[float, float, float, float, float, int], ...]: ...


@dataclass(frozen=True)
class EquipmentSportEvidenceConfig:
    max_frame_contexts: int = 12
    minimum_target_bbox_height_ratio: float = 0.08
    equipment_score_threshold: float = 0.25
    crop_width_scale: float = 2.0
    crop_height_scale: float = 1.6
    crop_vertical_center_offset_ratio: float = 0.15
    association_width_scale: float = 1.5
    association_top_ratio: float = 0.40
    association_bottom_extension_ratio: float = 0.40
    minimum_associated_equipment_observations: int = 2

    def __post_init__(self) -> None:
        for name in ("max_frame_contexts", "minimum_associated_equipment_observations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("minimum_target_bbox_height_ratio", "equipment_score_threshold"):
            _ratio(getattr(self, name), name)
        for name in (
            "crop_width_scale",
            "crop_height_scale",
            "association_width_scale",
        ):
            _positive(getattr(self, name), name)
        for name in ("crop_vertical_center_offset_ratio",):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite numeric")
        _ratio(self.association_top_ratio, "association_top_ratio")
        for name in ("association_bottom_extension_ratio",):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        return {"profile": EQUIPMENT_CONFIG_PROFILE, **asdict(self)}


class OpenMMLabEquipmentBackend:
    """Lazy MMDetection adapter with model-metadata class discovery."""

    def __init__(self, config_path: str, checkpoint_path: str, *, device: str = "cpu") -> None:
        self.config_path = str(Path(config_path).resolve())
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.device = device
        started = time.perf_counter()
        try:
            from mmdet.apis import inference_detector, init_detector
        except ImportError as error:
            raise RuntimeError("OPENMMLAB_DEPENDENCY_MISSING: mmdet") from error
        try:
            self._model = init_detector(self.config_path, self.checkpoint_path, device=device)
        except Exception as error:
            raise RuntimeError(f"MODEL_LOAD_FAILED: equipment detector: {error}") from error
        self.model_load_seconds = time.perf_counter() - started
        self._infer = inference_detector
        classes = self._model.dataset_meta.get("classes") if self._model.dataset_meta else None
        if not classes:
            raise RuntimeError("EQUIPMENT_CLASS_MAP_UNSUPPORTED: dataset classes missing")
        self.class_names = tuple(str(item) for item in classes)

    def infer(self, image: object):
        try:
            from mmengine.registry import init_default_scope

            init_default_scope("mmdet")
            instances = self._infer(self._model, image).pred_instances.cpu()
            boxes = instances.bboxes.numpy().tolist()
            scores = instances.scores.numpy().tolist()
            labels = instances.labels.numpy().tolist()
        except Exception as error:
            raise RuntimeError(f"EQUIPMENT_DETECTOR_INFERENCE_FAILED: {error}") from error
        return tuple(
            (*box, float(score), int(label))
            for box, score, label in zip(boxes, scores, labels, strict=True)
        )


class MMDetEquipmentSportEvidenceProvider:
    name = EQUIPMENT_PROVIDER_NAME
    kind = SportEvidenceKind.EQUIPMENT
    execution_scope = "FRAME"

    def __init__(
        self,
        backend: EquipmentDetectorBackend,
        config: EquipmentSportEvidenceConfig | None = None,
        *,
        model_id: str = "rtmdet-tiny-640-coco-equipment",
        device: str = "cpu",
        config_path: str | None = None,
        config_source: str | None = None,
        checkpoint_path: str | None = None,
        checkpoint_source: str | None = None,
        checkpoint_sha256: str | None = None,
        model_load_seconds: float | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or EquipmentSportEvidenceConfig()
        self.model_id = model_id
        self.device = device
        self.config_path = config_path
        self.config_source = config_source
        self.checkpoint_path = checkpoint_path
        self.checkpoint_source = checkpoint_source
        self.checkpoint_sha256 = checkpoint_sha256
        self.model_load_seconds = model_load_seconds
        try:
            self.skis_label_index = backend.class_names.index("skis")
            self.snowboard_label_index = backend.class_names.index("snowboard")
        except ValueError as error:
            raise RuntimeError(
                "EQUIPMENT_CLASS_MAP_UNSUPPORTED: required classes 'skis' and 'snowboard'"
            ) from error
        self.last_debug_frames: list[dict[str, object]] = []
        self.last_performance: dict[str, object] = {}
        self.last_summary: dict[str, object] = {}

    def infer(self, contexts=None) -> SportEvidenceProviderResult:
        started = time.perf_counter()
        self.last_debug_frames = []
        inference_latencies = []
        association_seconds = 0.0
        try:
            eligible, selected, below = select_equipment_contexts(
                tuple(contexts or ()), self.config
            )
            observations = []
            for context in selected:
                crop_bbox = equipment_crop_bbox(context, self.config)
                zone = equipment_association_zone(context, self.config)
                crop = _crop_image(context.frame_reference, crop_bbox)
                stage = time.perf_counter()
                detections = tuple(self.backend.infer(crop))
                inference_latencies.append(time.perf_counter() - stage)
                stage = time.perf_counter()
                diagnostic, observation = self._associate(context, crop_bbox, zone, detections)
                association_seconds += time.perf_counter() - stage
                self.last_debug_frames.append(diagnostic)
                if observation is not None:
                    observations.append(observation)
            supports = [max(item.ski_support, item.snowboard_support) for item in observations]
            self.last_summary = {
                "eligible_locked_context_count": len(eligible),
                "selected_equipment_context_count": len(selected),
                "contexts_below_target_size_threshold": below,
                "equipment_inference_context_count": len(inference_latencies),
                "frames_with_associated_skis": sum(item.ski_support > 0 for item in observations),
                "frames_with_associated_snowboard": sum(
                    item.snowboard_support > 0 for item in observations
                ),
                "frames_with_both": sum(
                    item.ski_support > 0 and item.snowboard_support > 0 for item in observations
                ),
                "equipment_observation_count": len(observations),
                "mean_detector_support": mean(supports) if supports else None,
                "median_detector_support": median(supports) if supports else None,
            }
            self.last_performance = {
                "equipment_model_load_seconds": self.model_load_seconds,
                "equipment_inference_total_seconds": sum(inference_latencies),
                "equipment_mean_inference_seconds": mean(inference_latencies)
                if inference_latencies
                else None,
                "equipment_p95_inference_seconds": _p95(inference_latencies),
                "equipment_association_total_seconds": association_seconds,
                "equipment_provider_total_seconds": time.perf_counter() - started,
            }
            return SportEvidenceProviderResult(
                provider_name=self.name,
                evidence_kind=self.kind,
                status=(
                    SportEvidenceProviderStatus.EXECUTED_WITH_EVIDENCE
                    if observations
                    else SportEvidenceProviderStatus.EXECUTED_NO_EVIDENCE
                ),
                observations=tuple(observations),
                limitations=("TARGET_EQUIPMENT_ASSOCIATION_IS_GEOMETRIC_ONLY",),
            )
        except Exception as error:
            self.last_performance = {
                "equipment_model_load_seconds": self.model_load_seconds,
                "equipment_inference_total_seconds": sum(inference_latencies),
                "equipment_mean_inference_seconds": mean(inference_latencies)
                if inference_latencies
                else None,
                "equipment_p95_inference_seconds": _p95(inference_latencies),
                "equipment_association_total_seconds": association_seconds,
                "equipment_provider_total_seconds": time.perf_counter() - started,
            }
            return SportEvidenceProviderResult(
                provider_name=self.name,
                evidence_kind=self.kind,
                status=SportEvidenceProviderStatus.FAILED,
                error=f"{type(error).__name__}: {error}",
            )

    def _associate(self, context, crop_bbox, zone, detections):
        raw_ski = raw_snowboard = associated_ski = associated_snowboard = 0
        accepted = []
        rejected = []
        for x1, y1, x2, y2, score, label in detections:
            _validate_backend_detection(x1, y1, x2, y2, score, label)
            if label not in {self.skis_label_index, self.snowboard_label_index}:
                continue
            name = "skis" if label == self.skis_label_index else "snowboard"
            if name == "skis":
                raw_ski += 1
            else:
                raw_snowboard += 1
            mapped = BoundingBox2D(
                crop_bbox.x_px + float(x1),
                crop_bbox.y_px + float(y1),
                float(x2 - x1),
                float(y2 - y1),
            )
            mapped.validate(context.geometry)
            center = (mapped.x_px + mapped.width_px / 2, mapped.y_px + mapped.height_px / 2)
            record = {"class_name": name, "score": float(score), "bbox": mapped.to_dict()}
            if score >= self.config.equipment_score_threshold and _contains(zone, *center):
                accepted.append(record)
                if name == "skis":
                    associated_ski += 1
                else:
                    associated_snowboard += 1
            else:
                rejected.append(record)
        ski_support = max(
            (item["score"] for item in accepted if item["class_name"] == "skis"), default=0.0
        )
        snowboard_support = max(
            (item["score"] for item in accepted if item["class_name"] == "snowboard"),
            default=0.0,
        )
        emitted = ski_support > 0 or snowboard_support > 0
        diagnostic = {
            "timestamp_us": context.timestamp_us,
            "frame_index": context.frame_index,
            "target_bbox": context.target_bbox.to_dict(),
            "crop_bbox": crop_bbox.to_dict(),
            "association_zone": zone.to_dict(),
            "raw_skis_detection_count": raw_ski,
            "raw_snowboard_detection_count": raw_snowboard,
            "associated_skis_count": associated_ski,
            "associated_snowboard_count": associated_snowboard,
            "max_ski_support": ski_support,
            "max_snowboard_support": snowboard_support,
            "observation_emitted": emitted,
            "associated_detections": accepted,
            "rejected_equipment_detections": rejected,
        }
        observation = (
            SportEvidenceObservation(
                evidence_id=f"equipment-{context.timestamp_us}-{context.frame_index}",
                kind=SportEvidenceKind.EQUIPMENT,
                provider_name=self.name,
                timestamp_us=context.timestamp_us,
                temporal_segment_id=None,
                ski_support=ski_support,
                snowboard_support=snowboard_support,
                quality=1.0,
                scope=SportEvidenceScope.FRAME,
                reason="ASSOCIATED_COCO_EQUIPMENT_DETECTION",
                limitations=(
                    "DETECTOR_SUPPORT_NOT_CALIBRATED_SPORT_PROBABILITY",
                    "TARGET_EQUIPMENT_ASSOCIATION_IS_GEOMETRIC_ONLY",
                    "COCO_EQUIPMENT_DETECTION_NOT_SPORT_TYPE_GT",
                ),
            )
            if emitted
            else None
        )
        return diagnostic, observation

    def provenance(self) -> dict[str, object]:
        return {
            "provider_name": self.name,
            "model_id": self.model_id,
            "config_path": self.config_path,
            "config_source": self.config_source,
            "checkpoint_filename": Path(self.checkpoint_path).name
            if self.checkpoint_path
            else None,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_source": self.checkpoint_source,
            "device": self.device,
            "class_count": len(self.backend.class_names),
            "class_map_names": list(self.backend.class_names),
            "resolved_skis_label_index": self.skis_label_index,
            "resolved_snowboard_label_index": self.snowboard_label_index,
            "equipment_score_threshold": self.config.equipment_score_threshold,
            "equipment_config": self.config.to_dict(),
        }


def select_equipment_contexts(contexts, config: EquipmentSportEvidenceConfig):
    by_timestamp = {}
    for context in sorted(contexts, key=lambda item: (item.timestamp_us, item.frame_index)):
        by_timestamp.setdefault(context.timestamp_us, context)
    distinct = tuple(by_timestamp.values())
    eligible = tuple(
        item
        for item in distinct
        if item.target_bbox.height_px / item.geometry.height_px
        >= config.minimum_target_bbox_height_ratio
    )
    below = len(distinct) - len(eligible)
    if len(eligible) <= config.max_frame_contexts:
        return eligible, eligible, below
    count = config.max_frame_contexts
    if count == 1:
        return eligible, (eligible[0],), below
    indices = tuple(index * (len(eligible) - 1) // (count - 1) for index in range(count))
    return eligible, tuple(eligible[index] for index in indices), below


def equipment_crop_bbox(context, config: EquipmentSportEvidenceConfig) -> BoundingBox2D:
    target = context.target_bbox
    width = target.width_px * config.crop_width_scale
    height = target.height_px * config.crop_height_scale
    center_x = target.x_px + target.width_px / 2
    center_y = (
        target.y_px
        + target.height_px / 2
        + target.height_px * config.crop_vertical_center_offset_ratio
    )
    left = max(0, math.floor(center_x - width / 2))
    top = max(0, math.floor(center_y - height / 2))
    right = min(context.geometry.width_px, math.ceil(center_x + width / 2))
    bottom = min(context.geometry.height_px, math.ceil(center_y + height / 2))
    if right <= left or bottom <= top:
        raise ValueError("EQUIPMENT_CROP_INVALID")
    return BoundingBox2D(float(left), float(top), float(right - left), float(bottom - top))


def equipment_association_zone(context, config: EquipmentSportEvidenceConfig) -> BoundingBox2D:
    target = context.target_bbox
    width = target.width_px * config.association_width_scale
    center_x = target.x_px + target.width_px / 2
    left = max(0.0, center_x - width / 2)
    right = min(float(context.geometry.width_px), center_x + width / 2)
    top = max(0.0, target.y_px + target.height_px * config.association_top_ratio)
    bottom = min(
        float(context.geometry.height_px),
        target.y_px + target.height_px * (1 + config.association_bottom_extension_ratio),
    )
    if right <= left or bottom <= top:
        raise ValueError("EQUIPMENT_ASSOCIATION_ZONE_INVALID")
    return BoundingBox2D(left, top, right - left, bottom - top)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def equipment_provider_doctor(
    config_path: str | None,
    checkpoint_path: str | None,
    *,
    device: str = "cpu",
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    report = {
        "EQUIPMENT_SPORT_PROVIDER_READINESS": "NOT_CONFIGURED",
        "device": device,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": None,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_sha256_matches_expected": None,
        "class_count": None,
        "class_names": None,
        "resolved_skis_label_index": None,
        "resolved_snowboard_label_index": None,
        "model_load_seconds": None,
        "error": None,
    }
    if not config_path or not checkpoint_path:
        return report
    if not Path(config_path).is_file() or not Path(checkpoint_path).is_file():
        report["EQUIPMENT_SPORT_PROVIDER_READINESS"] = "MODEL_UNAVAILABLE"
        report["error"] = "equipment config or checkpoint does not exist"
        return report
    actual_sha = sha256_file(checkpoint_path)
    report["checkpoint_sha256"] = actual_sha
    report["checkpoint_sha256_matches_expected"] = (
        actual_sha == expected_checkpoint_sha256 if expected_checkpoint_sha256 else None
    )
    if expected_checkpoint_sha256 and actual_sha != expected_checkpoint_sha256:
        report["EQUIPMENT_SPORT_PROVIDER_READINESS"] = "CHECKPOINT_SHA256_MISMATCH"
        report["error"] = "equipment checkpoint SHA256 does not match registry"
        return report
    try:
        backend = OpenMMLabEquipmentBackend(config_path, checkpoint_path, device=device)
        provider = MMDetEquipmentSportEvidenceProvider(
            backend,
            device=device,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=actual_sha,
            model_load_seconds=backend.model_load_seconds,
        )
    except RuntimeError as error:
        report["EQUIPMENT_SPORT_PROVIDER_READINESS"] = (
            "CLASS_MAP_UNSUPPORTED"
            if "EQUIPMENT_CLASS_MAP_UNSUPPORTED" in str(error)
            else "MODEL_LOAD_FAILED"
        )
        report["error"] = str(error)
        return report
    report.update(
        {
            "EQUIPMENT_SPORT_PROVIDER_READINESS": "READY_CPU" if device == "cpu" else "READY",
            "class_count": len(backend.class_names),
            "class_names": list(backend.class_names),
            "resolved_skis_label_index": provider.skis_label_index,
            "resolved_snowboard_label_index": provider.snowboard_label_index,
            "model_load_seconds": backend.model_load_seconds,
        }
    )
    return report


def _crop_image(image, bbox):
    left, top = round(bbox.x_px), round(bbox.y_px)
    right, bottom = round(bbox.x_px + bbox.width_px), round(bbox.y_px + bbox.height_px)
    crop = image[top:bottom, left:right]
    if getattr(crop, "size", 0) == 0:
        raise ValueError("EQUIPMENT_CROP_EMPTY")
    return crop


def _contains(bbox, x, y):
    return (
        bbox.x_px <= x <= bbox.x_px + bbox.width_px and bbox.y_px <= y <= bbox.y_px + bbox.height_px
    )


def _ratio(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _validate_backend_detection(x1, y1, x2, y2, score, label):
    for name, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"equipment detection {name} must be finite numeric")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("equipment detection bbox must have positive dimensions")
    _ratio(score, "equipment detection score")
    if isinstance(label, bool) or not isinstance(label, int) or label < 0:
        raise ValueError("equipment detection label must be a non-negative integer")


def _p95(values):
    if len(values) < MINIMUM_P95_SAMPLES:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]
