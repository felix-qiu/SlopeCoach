"""Optional target-crop OpenAI CLIP SportType evidence for A6.2 research."""

from __future__ import annotations

import hashlib
import json
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
from .equipment_mmdet import MINIMUM_P95_SAMPLES, sha256_file
from .providers import select_sport_frame_contexts, sport_evidence_id

VISUAL_PROVIDER_NAME = "openai-clip-vit-b32-visual-sport"
VISUAL_MODEL_ID = VISUAL_PROVIDER_NAME
VISUAL_MODEL_NAME = "ViT-B/32"
VISUAL_CONFIG_PROFILE = "RESEARCH_DEFAULTS_A6_2"
VISUAL_SPORT_PROMPT_SCHEMA_VERSION = "visual-sport-prompts-v1"
VISUAL_IMPLEMENTATION_SOURCE = "https://github.com/openai/CLIP"
VISUAL_IMPLEMENTATION_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"

VISUAL_SPORT_PROMPTS = {
    "SKI": (
        "a photo of a skier",
        "a person skiing on two skis",
        "an alpine skier riding skis on snow",
        "a person wearing skis on a snowy slope",
    ),
    "SNOWBOARD": (
        "a photo of a snowboarder",
        "a person riding a snowboard on snow",
        "a snowboard rider on a snowy slope",
        "a person standing on a snowboard",
    ),
    "NEUTRAL": (
        "a person on a snowy slope",
        "a person standing on snow",
        "a distant person on snow",
        "a person whose snow sport equipment is unclear",
    ),
}


def visual_prompt_payload() -> dict[str, object]:
    return {
        "schema_version": VISUAL_SPORT_PROMPT_SCHEMA_VERSION,
        "classes": [
            {"name": name, "prompts": list(prompts)}
            for name, prompts in VISUAL_SPORT_PROMPTS.items()
        ],
    }


def visual_prompt_sha256() -> str:
    encoded = json.dumps(
        visual_prompt_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VisualSportScores:
    ski_support: float
    snowboard_support: float
    neutral_support: float
    top_class: str | None = None
    raw_class_logits: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        values = (self.ski_support, self.snowboard_support, self.neutral_support)
        for name, value in zip(
            ("ski_support", "snowboard_support", "neutral_support"), values, strict=True
        ):
            _unit_interval(value, name)
        if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("visual softmax supports must sum to 1")
        expected_top = ("SKI", "SNOWBOARD", "NEUTRAL")[max(range(3), key=values.__getitem__)]
        if self.top_class is not None and self.top_class != expected_top:
            raise ValueError("top_class does not match visual supports")
        if self.raw_class_logits is not None:
            if len(self.raw_class_logits) != 3:
                raise ValueError("raw_class_logits must contain exactly three values")
            for value in self.raw_class_logits:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                ):
                    raise ValueError("raw_class_logits must be finite numeric")

    def to_dict(self) -> dict[str, object]:
        return {
            "ski_support": self.ski_support,
            "snowboard_support": self.snowboard_support,
            "neutral_support": self.neutral_support,
            "top_class": self.top_class
            or ("SKI", "SNOWBOARD", "NEUTRAL")[
                max(
                    range(3),
                    key=(
                        self.ski_support,
                        self.snowboard_support,
                        self.neutral_support,
                    ).__getitem__,
                )
            ],
            "raw_class_logits": list(self.raw_class_logits)
            if self.raw_class_logits is not None
            else None,
        }


class VisualSportClassifierBackend(Protocol):
    model_name: str

    def infer(self, image: object) -> VisualSportScores: ...


class FakeVisualSportBackend:
    model_name = "fake-visual-sport-backend"

    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.calls: list[object] = []

    def infer(self, image: object) -> VisualSportScores:
        self.calls.append(image)
        if not self.outputs:
            raise RuntimeError("fake visual backend has no output")
        output = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        if isinstance(output, Exception):
            raise output
        return output


class OpenAIClipVisualSportBackend:
    """Lazy official OpenAI CLIP adapter using one explicit local checkpoint."""

    model_name = VISUAL_MODEL_NAME

    def __init__(self, checkpoint_path: str, *, device: str = "cpu") -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise RuntimeError("CHECKPOINT_MISSING: visual CLIP checkpoint")
        self.checkpoint_path = str(checkpoint.resolve())
        self.device = device
        try:
            import clip
            import torch
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(f"VISUAL_DEPENDENCY_MISSING: {error.name}") from error
        started = time.perf_counter()
        try:
            self._model, self._preprocess = clip.load(
                self.checkpoint_path, device=device, jit=False
            )
            self._model.eval()
        except Exception as error:
            raise RuntimeError(f"MODEL_LOAD_FAILED: visual CLIP: {error}") from error
        self.model_load_seconds = time.perf_counter() - started
        self._torch = torch
        self._image_type = Image
        self.input_resolution = int(self._model.visual.input_resolution)
        started = time.perf_counter()
        self._class_prototypes = self._encode_class_prototypes(clip)
        self.text_prototype_seconds = time.perf_counter() - started

    def _encode_class_prototypes(self, clip_module):
        torch = self._torch
        prototypes = []
        with torch.no_grad():
            for prompts in VISUAL_SPORT_PROMPTS.values():
                tokens = clip_module.tokenize(list(prompts)).to(self.device)
                embeddings = self._model.encode_text(tokens).float()
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                prototype = embeddings.mean(dim=0)
                prototype = prototype / prototype.norm()
                prototypes.append(prototype)
        return torch.stack(prototypes)

    def infer(self, image: object) -> VisualSportScores:
        torch = self._torch
        rgb = bgr_to_rgb(image)
        pil_image = self._image_type.fromarray(rgb)
        tensor = self._preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self._model.encode_image(tensor).float()
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            logits = self._model.logit_scale.exp().float() * embedding @ self._class_prototypes.T
            probabilities = logits.softmax(dim=-1)[0].cpu().tolist()
            raw_logits = logits[0].cpu().tolist()
        top = ("SKI", "SNOWBOARD", "NEUTRAL")[max(range(3), key=probabilities.__getitem__)]
        return VisualSportScores(
            float(probabilities[0]),
            float(probabilities[1]),
            float(probabilities[2]),
            top_class=top,
            raw_class_logits=tuple(float(value) for value in raw_logits),
        )


@dataclass(frozen=True)
class VisualSportEvidenceConfig:
    max_frame_contexts: int = 12
    minimum_target_bbox_height_ratio: float = 0.08
    crop_width_scale: float = 1.8
    crop_height_scale: float = 1.5
    crop_vertical_center_offset_ratio: float = 0.08

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_frame_contexts, bool)
            or not isinstance(self.max_frame_contexts, int)
            or self.max_frame_contexts < 1
        ):
            raise ValueError("max_frame_contexts must be a positive integer")
        _unit_interval(self.minimum_target_bbox_height_ratio, "minimum_target_bbox_height_ratio")
        for name in ("crop_width_scale", "crop_height_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        value = self.crop_vertical_center_offset_ratio
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError("crop_vertical_center_offset_ratio must be finite numeric")

    def to_dict(self) -> dict[str, object]:
        return {"profile": VISUAL_CONFIG_PROFILE, **asdict(self)}


class ClipVisualSportEvidenceProvider:
    name = VISUAL_PROVIDER_NAME
    kind = SportEvidenceKind.VISUAL_CLASSIFIER
    execution_scope = "FRAME"

    def __init__(
        self,
        backend: VisualSportClassifierBackend,
        config: VisualSportEvidenceConfig | None = None,
        *,
        device: str = "cpu",
        implementation_commit: str | None = VISUAL_IMPLEMENTATION_COMMIT,
        checkpoint_path: str | None = None,
        checkpoint_sha256: str | None = None,
        model_load_seconds: float | None = None,
        text_prototype_seconds: float | None = None,
        input_resolution: int | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or VisualSportEvidenceConfig()
        self.device = device
        self.implementation_commit = implementation_commit
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.model_load_seconds = model_load_seconds
        self.text_prototype_seconds = text_prototype_seconds
        self.input_resolution = input_resolution
        self.last_debug_frames: list[dict[str, object]] = []
        self.last_summary: dict[str, object] = {}
        self.last_performance: dict[str, object] = {}

    def infer(self, contexts=None) -> SportEvidenceProviderResult:
        started = time.perf_counter()
        self.last_debug_frames = []
        latencies = []
        try:
            eligible, selected, below = select_sport_frame_contexts(
                tuple(contexts or ()),
                max_frame_contexts=self.config.max_frame_contexts,
                minimum_target_bbox_height_ratio=self.config.minimum_target_bbox_height_ratio,
            )
            observations = []
            scores = []
            for context in selected:
                crop_bbox = visual_crop_bbox(context, self.config)
                crop = crop_source_image(context.frame_reference, crop_bbox)
                stage = time.perf_counter()
                result = self.backend.infer(crop)
                latencies.append(time.perf_counter() - stage)
                if not isinstance(result, VisualSportScores):
                    raise ValueError("visual backend must return VisualSportScores")
                scores.append(result)
                observations.append(
                    SportEvidenceObservation(
                        evidence_id=sport_evidence_id(
                            self.kind, self.name, context.timestamp_us, context.frame_index
                        ),
                        kind=self.kind,
                        provider_name=self.name,
                        timestamp_us=context.timestamp_us,
                        ski_support=result.ski_support,
                        snowboard_support=result.snowboard_support,
                        quality=1.0,
                        scope=SportEvidenceScope.FRAME,
                        reason="TARGET_CROP_ZERO_SHOT_VISUAL_CLASSIFICATION",
                        limitations=(
                            "ZERO_SHOT_SUPPORT_NOT_CALIBRATED_PROBABILITY",
                            "SUPPORT_DEPENDS_ON_FIXED_PROMPT_TAXONOMY",
                            "TARGET_CROP_VISUAL_CLASSIFICATION_ONLY",
                            "NO_SPORT_TYPE_GT",
                        ),
                    )
                )
                self.last_debug_frames.append(
                    {
                        "timestamp_us": context.timestamp_us,
                        "frame_index": context.frame_index,
                        "target_bbox": context.target_bbox.to_dict(),
                        "visual_crop_bbox": crop_bbox.to_dict(),
                        **result.to_dict(),
                        "observation_emitted": True,
                    }
                )
            self.last_summary = _visual_summary(eligible, selected, below, scores)
            self.last_performance = _visual_performance(
                started,
                latencies,
                self.model_load_seconds,
                self.text_prototype_seconds,
            )
            return SportEvidenceProviderResult(
                provider_name=self.name,
                evidence_kind=self.kind,
                status=(
                    SportEvidenceProviderStatus.EXECUTED_WITH_EVIDENCE
                    if observations
                    else SportEvidenceProviderStatus.EXECUTED_NO_EVIDENCE
                ),
                observations=tuple(observations),
                limitations=(
                    "ZERO_SHOT_SUPPORT_NOT_CALIBRATED_PROBABILITY",
                    "SUPPORT_DEPENDS_ON_FIXED_PROMPT_TAXONOMY",
                ),
            )
        except Exception as error:
            self.last_performance = _visual_performance(
                started,
                latencies,
                self.model_load_seconds,
                self.text_prototype_seconds,
            )
            return SportEvidenceProviderResult(
                provider_name=self.name,
                evidence_kind=self.kind,
                status=SportEvidenceProviderStatus.FAILED,
                error=f"{type(error).__name__}: {error}",
            )

    def provenance(self) -> dict[str, object]:
        return {
            "provider_name": self.name,
            "model_id": VISUAL_MODEL_ID,
            "model_name": self.backend.model_name,
            "implementation_repository": "OpenAI CLIP",
            "implementation_source": VISUAL_IMPLEMENTATION_SOURCE,
            "implementation_commit": self.implementation_commit,
            "checkpoint_filename": Path(self.checkpoint_path).name
            if self.checkpoint_path
            else None,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "input_resolution": self.input_resolution,
            "visual_prompt_schema_version": VISUAL_SPORT_PROMPT_SCHEMA_VERSION,
            "visual_prompt_sha256": visual_prompt_sha256(),
            "prompt_class_names": list(VISUAL_SPORT_PROMPTS),
            "visual_config": self.config.to_dict(),
        }


def visual_crop_bbox(context, config: VisualSportEvidenceConfig) -> BoundingBox2D:
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
        raise ValueError("VISUAL_CROP_INVALID")
    return BoundingBox2D(float(left), float(top), float(right - left), float(bottom - top))


def crop_source_image(image, bbox):
    left, top = round(bbox.x_px), round(bbox.y_px)
    right, bottom = round(bbox.x_px + bbox.width_px), round(bbox.y_px + bbox.height_px)
    crop = image[top:bottom, left:right]
    if getattr(crop, "size", 0) == 0:
        raise ValueError("VISUAL_CROP_EMPTY")
    return crop.copy()


def bgr_to_rgb(image):
    """Convert an OpenCV BGR array to the RGB array expected by Pillow/CLIP."""

    if getattr(image, "ndim", None) != 3 or image.shape[2] != 3:
        raise ValueError("visual input must be an HxWx3 BGR image")
    return image[..., ::-1].copy()


def visual_provider_doctor(
    checkpoint_path: str | None,
    *,
    device: str = "cpu",
    expected_checkpoint_sha256: str | None = None,
    implementation_commit: str | None = VISUAL_IMPLEMENTATION_COMMIT,
) -> dict[str, object]:
    report = {
        "VISUAL_SPORT_PROVIDER_READINESS": "NOT_CONFIGURED",
        "provider_name": VISUAL_PROVIDER_NAME,
        "model_name": VISUAL_MODEL_NAME,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": None,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_sha256_matches_expected": None,
        "implementation_source": VISUAL_IMPLEMENTATION_SOURCE,
        "implementation_repository": "OpenAI CLIP",
        "implementation_commit": implementation_commit,
        "visual_prompt_schema_version": VISUAL_SPORT_PROMPT_SCHEMA_VERSION,
        "visual_prompt_sha256": visual_prompt_sha256(),
        "input_resolution": None,
        "model_load_seconds": None,
        "visual_text_prototype_seconds": None,
        "synthetic_scores": None,
        "error": None,
    }
    try:
        import clip  # noqa: F401
        import torch
        import torchvision
    except ImportError as error:
        report["VISUAL_SPORT_PROVIDER_READINESS"] = "DEPENDENCY_MISSING"
        report["error"] = f"{error.name} is not installed"
        return report
    report["torch_version"] = torch.__version__
    report["torchvision_version"] = torchvision.__version__
    if not checkpoint_path:
        return report
    if not Path(checkpoint_path).is_file():
        report["VISUAL_SPORT_PROVIDER_READINESS"] = "CHECKPOINT_MISSING"
        report["error"] = "visual checkpoint does not exist"
        return report
    actual_sha = sha256_file(checkpoint_path)
    report["checkpoint_sha256"] = actual_sha
    report["checkpoint_sha256_matches_expected"] = (
        actual_sha == expected_checkpoint_sha256 if expected_checkpoint_sha256 else None
    )
    if expected_checkpoint_sha256 and actual_sha != expected_checkpoint_sha256:
        report["VISUAL_SPORT_PROVIDER_READINESS"] = "CHECKPOINT_SHA256_MISMATCH"
        report["error"] = "visual checkpoint SHA256 does not match registry"
        return report
    try:
        backend = OpenAIClipVisualSportBackend(checkpoint_path, device=device)
        import numpy as np

        scores = backend.infer(
            np.zeros((backend.input_resolution, backend.input_resolution, 3), dtype=np.uint8)
        )
    except Exception as error:
        report["VISUAL_SPORT_PROVIDER_READINESS"] = "MODEL_LOAD_FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
        return report
    report.update(
        {
            "VISUAL_SPORT_PROVIDER_READINESS": "READY_CPU" if device == "cpu" else "READY",
            "input_resolution": backend.input_resolution,
            "model_load_seconds": backend.model_load_seconds,
            "visual_text_prototype_seconds": backend.text_prototype_seconds,
            "synthetic_scores": scores.to_dict(),
        }
    )
    return report


def prepare_visual_sport_model(destination: str | Path) -> dict[str, object]:
    """Explicitly download the official ViT-B/32 checkpoint into an ignored directory."""

    try:
        from clip.clip import _MODELS, _download
    except ImportError as error:
        raise RuntimeError("VISUAL_DEPENDENCY_MISSING: clip") from error
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(_download(_MODELS[VISUAL_MODEL_NAME], str(destination)))
    return {
        "provider_name": VISUAL_PROVIDER_NAME,
        "model_name": VISUAL_MODEL_NAME,
        "implementation_repository": "OpenAI CLIP",
        "implementation_commit": VISUAL_IMPLEMENTATION_COMMIT,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _visual_summary(eligible, selected, below, scores):
    ski = [item.ski_support for item in scores]
    snowboard = [item.snowboard_support for item in scores]
    neutral = [item.neutral_support for item in scores]
    tops = [
        item.top_class
        or ("SKI", "SNOWBOARD", "NEUTRAL")[
            max(
                range(3),
                key=(
                    item.ski_support,
                    item.snowboard_support,
                    item.neutral_support,
                ).__getitem__,
            )
        ]
        for item in scores
    ]
    return {
        "eligible_locked_context_count": len(eligible),
        "selected_visual_context_count": len(selected),
        "contexts_below_target_size_threshold": below,
        "visual_inference_context_count": len(scores),
        "frames_visual_favors_ski": tops.count("SKI"),
        "frames_visual_favors_snowboard": tops.count("SNOWBOARD"),
        "frames_visual_favors_neutral": tops.count("NEUTRAL"),
        "visual_observation_count": len(scores),
        "mean_ski_support": mean(ski) if ski else None,
        "median_ski_support": median(ski) if ski else None,
        "mean_snowboard_support": mean(snowboard) if snowboard else None,
        "median_snowboard_support": median(snowboard) if snowboard else None,
        "mean_neutral_support": mean(neutral) if neutral else None,
        "median_neutral_support": median(neutral) if neutral else None,
    }


def _visual_performance(started, latencies, model_load_seconds, prototype_seconds):
    return {
        "visual_model_load_seconds": model_load_seconds,
        "visual_text_prototype_seconds": prototype_seconds,
        "visual_inference_total_seconds": sum(latencies),
        "visual_mean_inference_seconds": mean(latencies) if latencies else None,
        "visual_p95_inference_seconds": _p95(latencies),
        "visual_provider_total_seconds": time.perf_counter() - started,
    }


def _p95(values):
    if len(values) < MINIMUM_P95_SAMPLES:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _unit_interval(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")
