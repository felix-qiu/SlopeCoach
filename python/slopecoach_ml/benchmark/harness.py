from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from slopecoach_ml.quality import VideoQualityGate
from slopecoach_ml.reference import (
    ReferenceAnalysisConfig,
    ReferenceAnalysisContext,
    analyze_pose_frame,
    load_golden_fixture,
)
from slopecoach_ml.video import inspect_video


def _performance(*, total: float, stages: dict[str, float], samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    p95 = None
    if len(ordered) >= 20:
        p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "total_processing_seconds": total,
        "per_stage_seconds": stages,
        "mean_latency_seconds": sum(samples) / len(samples) if samples else None,
        "p95_latency_seconds": p95,
    }


def _base() -> dict[str, Any]:
    return {
        "benchmark_contract_version": "ski-bench-reference-v1",
        "REAL_GT_STATUS": "NOT_AVAILABLE",
        "ground_truth_metrics": {
            "diagnosis_precision": None,
            "diagnosis_recall": None,
            "diagnosis_f1": None,
        },
    }


def benchmark_golden(
    path: str | Path, *, clock: Callable[[], float] = time.perf_counter
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = clock()
    stage_start = clock()
    frame, expected = load_golden_fixture(path)
    parse_time = clock() - stage_start
    stage_start = clock()
    result = analyze_pose_frame(
        frame,
        context=ReferenceAnalysisContext(
            analysis_id="golden-pose-001",
            provider_name="golden-fixture",
            model_id="golden-pose-001",
            model_version="golden-pose-v1",
        ),
        config=ReferenceAnalysisConfig(),
    )
    analysis_time = clock() - stage_start
    finished = clock()
    angle = result.features["left_knee_angle_2d_degrees"]
    expected_angle = float(expected["left_knee_angle_2d_degrees"])
    tolerance = float(expected["absolute_tolerance"])
    passed = angle is not None and math.isclose(angle, expected_angle, abs_tol=tolerance)
    report = _base()
    report.update(
        {
            "input_kind": "GOLDEN_FIXTURE",
            "pose_provider": "GOLDEN_FIXTURE",
            "pose": {
                "coverage": 1.0,
                "confidence": min(
                    point.confidence for point in frame.persons[0].keypoints.values()
                ),
                "stability": None,
            },
            "biomechanics": {"knee_angle_2d_coverage": 1.0 if angle is not None else 0.0},
            "performance": _performance(
                total=finished - started,
                stages={"parse": parse_time, "analysis": analysis_time},
                samples=[analysis_time],
            ),
            "golden_passed": passed,
        }
    )
    return report, result.to_dict()


def benchmark_video(
    path: str | Path,
    *,
    input_kind: str = "REAL_VIDEO",
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if input_kind not in {"REAL_VIDEO", "SYNTHETIC_METADATA_SMOKE"}:
        raise ValueError("video input_kind must be REAL_VIDEO or SYNTHETIC_METADATA_SMOKE")
    started = clock()
    stage_start = clock()
    metadata = inspect_video(path)
    inspect_time = clock() - stage_start
    stage_start = clock()
    quality = VideoQualityGate().evaluate(metadata)
    quality_time = clock() - stage_start
    finished = clock()
    report = _base()
    report.update(
        {
            "input_kind": input_kind,
            "pose_provider": "NOT_CONFIGURED",
            "video_metadata": metadata.to_dict(),
            "video_quality": quality.to_dict(),
            "pose": {"coverage": None, "confidence": None, "stability": None},
            "biomechanics": {"knee_angle_2d_coverage": None},
            "performance": _performance(
                total=finished - started,
                stages={"video_inspection": inspect_time, "quality_gate": quality_time},
                samples=[inspect_time],
            ),
        }
    )
    return report
