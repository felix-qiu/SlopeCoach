from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from slopecoach_ml.benchmark import benchmark_golden, benchmark_real_pose_frames, benchmark_video
from slopecoach_ml.detection.mmdet_provider import (
    MMDetPersonDetectorProvider,
    OpenMMLabMMDetBackend,
)
from slopecoach_ml.models import load_model_registry
from slopecoach_ml.openmmlab import openmmlab_preflight
from slopecoach_ml.pose import render_debug_overlay
from slopecoach_ml.pose.mmpose_provider import MMPoseRTMWPoseProvider, OpenMMLabMMPoseBackend
from slopecoach_ml.quality import VideoQualityGate, VideoQualityStatus
from slopecoach_ml.reference import (
    ReferenceAnalysisConfig,
    ReferenceAnalysisContext,
    analyze_pose_frame,
    load_golden_fixture,
)
from slopecoach_ml.video import OpenCVVideoSampler, inspect_video


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(payload: dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SlopeCoach research/reference CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    golden = subparsers.add_parser("golden", help="run the deterministic golden pose pipeline")
    golden.add_argument("--fixture", default=str(_root() / "fixtures/golden_pose_001.json"))
    golden.add_argument("--output")
    inspect = subparsers.add_parser(
        "inspect-video", help="inspect real video metadata with ffprobe"
    )
    inspect.add_argument("video")
    inspect.add_argument("--output")
    benchmark = subparsers.add_parser(
        "benchmark", help="benchmark a golden JSON fixture or real video"
    )
    benchmark.add_argument("input")
    benchmark.add_argument("--output")
    subparsers.add_parser("pose-doctor", help="inspect optional OpenMMLab provider readiness")
    pose_image = subparsers.add_parser(
        "pose-image", help="run configured RTMDet + RTMW-L on an image"
    )
    pose_image.add_argument("image")
    pose_image.add_argument("--output")
    pose_image.add_argument("--overlay")
    pose_image.add_argument("--input-non-mirrored", action="store_true")
    real_video = subparsers.add_parser(
        "benchmark-real-pose", help="run configured real-pose video benchmark"
    )
    real_video.add_argument("video")
    real_video.add_argument("--sample-fps", type=float, default=2.0)
    real_video.add_argument("--output")
    real_video.add_argument("--input-non-mirrored", action="store_true")
    return parser


def _real_providers():
    required = {
        "detector_config": os.getenv("SLOPECOACH_DETECTOR_CONFIG"),
        "detector_checkpoint": os.getenv("SLOPECOACH_DETECTOR_CHECKPOINT"),
        "pose_config": os.getenv("SLOPECOACH_POSE_CONFIG"),
        "pose_checkpoint": os.getenv("SLOPECOACH_POSE_CHECKPOINT"),
    }
    missing = [name for name, value in required.items() if not value or not Path(value).is_file()]
    if missing:
        raise RuntimeError(f"MODEL_CHECKPOINT_MISSING: {', '.join(missing)}")
    detector = MMDetPersonDetectorProvider(
        OpenMMLabMMDetBackend(required["detector_config"], required["detector_checkpoint"])
    )
    pose = MMPoseRTMWPoseProvider(
        OpenMMLabMMPoseBackend(required["pose_config"], required["pose_checkpoint"])
    )
    return detector, pose


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pose-doctor":
        report = openmmlab_preflight()
        _write_json(report, None)
        return 0 if report["OPENMMLAB_PREFLIGHT"]["status"] == "READY" else 3
    if args.command == "pose-image":
        if not args.input_non_mirrored:
            raise RuntimeError(
                "MIRROR_STATE_UNRESOLVED: pass --input-non-mirrored to attest input state"
            )
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OPENMMLAB_DEPENDENCY_MISSING: opencv-python") from error
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError("IMAGE_DECODE_FAILED")
        detector, pose_provider = _real_providers()
        height, width = image.shape[:2]
        from slopecoach_ml.pose import FrameGeometry

        geometry = FrameGeometry(width, height)
        started = time.perf_counter()
        stage = time.perf_counter()
        detections = detector.detect(image, geometry)
        detector_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        frame = pose_provider.estimate_detections(
            image, detections, geometry, timestamp_us=0, frame_index=0
        )
        pose_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        result = analyze_pose_frame(
            frame,
            context=ReferenceAnalysisContext(
                "real-pose-image", pose_provider.name, "rtmw-l-cocktail14-256x192", "20231122"
            ),
            config=ReferenceAnalysisConfig(),
        )
        biomechanics_seconds = time.perf_counter() - stage
        payload = {
            "input": args.image,
            "provider": pose_provider.name,
            "detector": detector.name,
            "pose_model": "rtmw-l-cocktail14-256x192@20231122",
            "device": "cpu",
            "frame_geometry": geometry.to_dict(),
            "person_count": len(frame.persons),
            "persons": [person.to_dict() for person in frame.persons],
            "canonical_joint_schema": frame.joint_schema,
            "reference_analysis": result.to_dict(),
            "warnings": list(result.warnings),
            "limitations": list(result.limitations),
            "timing": {
                "total_processing_seconds": time.perf_counter() - started,
                "detector_seconds": detector_seconds,
                "pose_seconds": pose_seconds,
                "biomechanics_seconds": biomechanics_seconds,
            },
        }
        if args.overlay:
            render_debug_overlay(image, frame, args.overlay, provider_name=pose_provider.name)
        _write_json(payload, args.output)
        return 0
    if args.command == "benchmark-real-pose":
        if not args.input_non_mirrored:
            raise RuntimeError(
                "MIRROR_STATE_UNRESOLVED: pass --input-non-mirrored to attest input state"
            )
        detector, pose_provider = _real_providers()
        registry = load_model_registry(_root() / "models/registry.json")
        report = benchmark_real_pose_frames(
            input_path=args.video,
            input_kind="REAL_VIDEO",
            frames=OpenCVVideoSampler(args.video, sample_fps=args.sample_fps),
            detector=detector,
            pose_provider=pose_provider,
            detector_model=registry["rtmdet-m-640-coco-obj365-person"].to_dict(),
            pose_model=registry["rtmw-l-cocktail14-256x192"].to_dict(),
        )
        _write_json(report, args.output)
        return 0
    if args.command == "golden":
        frame, expected = load_golden_fixture(args.fixture)
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
        actual = result.features["left_knee_angle_2d_degrees"]
        passed = (
            actual is not None
            and abs(actual - expected["left_knee_angle_2d_degrees"])
            <= expected["absolute_tolerance"]
        )
        _write_json(
            {"golden_passed": passed, "reference_analysis_result": result.to_dict()}, args.output
        )
        return 0 if passed else 1
    if args.command == "inspect-video":
        metadata = inspect_video(args.video)
        quality = VideoQualityGate().evaluate(metadata)
        _write_json(
            {"video_metadata": metadata.to_dict(), "video_quality": quality.to_dict()}, args.output
        )
        return 0 if quality.status is not VideoQualityStatus.NOT_ANALYZABLE else 2
    input_path = Path(args.input)
    if input_path.suffix.lower() == ".json":
        report, result = benchmark_golden(input_path)
        _write_json({"benchmark": report, "reference_analysis_result": result}, args.output)
        return 0 if report["golden_passed"] else 1
    report = benchmark_video(input_path)
    _write_json(report, args.output)
    return 0 if report["video_metadata"]["readable"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
