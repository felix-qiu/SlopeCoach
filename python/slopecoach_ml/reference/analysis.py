"""PROVISIONAL REFERENCE MODEL for research, golden tests, and Rust parity validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slopecoach_ml.biomechanics import knee_angle_2d
from slopecoach_ml.pose import PoseFrame
from slopecoach_ml.quality import VideoQuality
from slopecoach_ml.video import VideoMetadata

NON_SQUARE_PIXEL_ASPECT_RATIO_UNSUPPORTED = "NON_SQUARE_PIXEL_ASPECT_RATIO_UNSUPPORTED"
MULTIPLE_PERSONS_TARGET_IDENTITY_UNRESOLVED = "MULTIPLE_PERSONS_TARGET_IDENTITY_UNRESOLVED"


@dataclass(frozen=True)
class ReferenceAnalysisConfig:
    """PROVISIONAL / REFERENCE ONLY; not a production contract source of truth."""

    min_joint_confidence: float = 0.5
    square_pixel_tolerance: float = 1e-6

    def validate(self) -> None:
        if isinstance(self.min_joint_confidence, bool) or not isinstance(
            self.min_joint_confidence, int | float
        ):
            raise TypeError("min_joint_confidence must be numeric")
        if not math.isfinite(self.min_joint_confidence):
            raise ValueError("min_joint_confidence must be finite")
        if not 0.0 <= self.min_joint_confidence <= 1.0:
            raise ValueError("min_joint_confidence must be in [0, 1]")
        if isinstance(self.square_pixel_tolerance, bool) or not isinstance(
            self.square_pixel_tolerance, int | float
        ):
            raise TypeError("square_pixel_tolerance must be numeric")
        if not math.isfinite(self.square_pixel_tolerance):
            raise ValueError("square_pixel_tolerance must be finite")
        if self.square_pixel_tolerance < 0:
            raise ValueError("square_pixel_tolerance must be non-negative")


@dataclass(frozen=True)
class ReferenceAnalysisContext:
    analysis_id: str
    provider_name: str | None = None
    model_id: str | None = None
    model_version: str | None = None


@dataclass(frozen=True)
class ReferenceAnalysisResult:
    analysis_id: str
    reference_contract_version: str
    video_metadata: VideoMetadata | None
    video_quality: VideoQuality | None
    pose_summary: dict[str, Any]
    features: dict[str, float | None]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    model_versions: dict[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["video_metadata"] = self.video_metadata.to_dict() if self.video_metadata else None
        data["video_quality"] = self.video_quality.to_dict() if self.video_quality else None
        data["warnings"] = list(self.warnings)
        data["limitations"] = list(self.limitations)
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)


def load_golden_fixture(path: str | Path) -> tuple[PoseFrame, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PoseFrame.from_dict(data["pose_frame"]), data["expected"]


def analyze_pose_frame(
    frame: PoseFrame,
    *,
    context: ReferenceAnalysisContext,
    config: ReferenceAnalysisConfig,
) -> ReferenceAnalysisResult:
    frame.validate()
    config.validate()
    warnings: list[str] = []
    limitations = ["IMAGE_2D_ONLY_NOT_PHYSICAL_3D"]
    person = frame.persons[0] if len(frame.persons) == 1 else None
    if not frame.persons:
        warnings.append("no person pose available")
    elif len(frame.persons) > 1:
        warnings.append(MULTIPLE_PERSONS_TARGET_IDENTITY_UNRESOLVED)
        limitations.append(MULTIPLE_PERSONS_TARGET_IDENTITY_UNRESOLVED)
    if not math.isclose(
        frame.geometry.pixel_aspect_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=config.square_pixel_tolerance,
    ):
        limitations.append(NON_SQUARE_PIXEL_ASPECT_RATIO_UNSUPPORTED)
    angle = (
        knee_angle_2d(
            person,
            frame.geometry,
            minimum_confidence=config.min_joint_confidence,
            square_pixel_tolerance=config.square_pixel_tolerance,
        )
        if person is not None
        else None
    )
    return ReferenceAnalysisResult(
        analysis_id=context.analysis_id,
        reference_contract_version="python-reference-v1",
        video_metadata=None,
        video_quality=None,
        pose_summary={
            "frame_count": 1,
            "person_count": len(frame.persons),
            "joint_schema": frame.joint_schema,
        },
        features={"left_knee_angle_2d_degrees": angle},
        warnings=tuple(warnings),
        limitations=tuple(limitations),
        model_versions={
            "provider": context.provider_name,
            "model_id": context.model_id,
            "model_version": context.model_version,
        },
    )
