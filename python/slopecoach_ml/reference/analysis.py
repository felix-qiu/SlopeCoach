"""PROVISIONAL REFERENCE MODEL for research, golden tests, and Rust parity validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slopecoach_ml.biomechanics import knee_angle_2d
from slopecoach_ml.pose import PoseFrame
from slopecoach_ml.quality import VideoQuality
from slopecoach_ml.video import VideoMetadata


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
    frame: PoseFrame, *, analysis_id: str = "golden-pose-001"
) -> ReferenceAnalysisResult:
    frame.validate()
    person = frame.persons[0] if frame.persons else None
    angle = knee_angle_2d(person, frame.geometry) if person is not None else None
    warnings = () if person is not None else ("no person pose available",)
    return ReferenceAnalysisResult(
        analysis_id=analysis_id,
        reference_contract_version="python-reference-v1",
        video_metadata=None,
        video_quality=None,
        pose_summary={
            "frame_count": 1,
            "person_count": len(frame.persons),
            "joint_schema": frame.joint_schema,
        },
        features={"left_knee_angle_2d_degrees": angle},
        warnings=warnings,
        limitations=("Image2D measurement only; not Physical3D or physical edge angle.",),
        model_versions={"detector": None, "pose": "golden-fixture"},
    )
