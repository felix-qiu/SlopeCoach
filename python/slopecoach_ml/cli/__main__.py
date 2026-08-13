from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from slopecoach_ml.benchmark import (
    RealPoseDebugCollector,
    TargetIdentityDebugCollector,
    TemporalTurnCollector,
    benchmark_golden,
    benchmark_real_pose_frames,
    benchmark_target_identity_frames,
    benchmark_temporal_turns_frames,
    benchmark_video,
    write_ground_truth_comparison,
    write_temporal_debug_artifacts,
)
from slopecoach_ml.detection.mmdet_provider import (
    MMDetPersonDetectorProvider,
    OpenMMLabMMDetBackend,
)
from slopecoach_ml.identity import load_target_ground_truth, prepare_target_gt_template
from slopecoach_ml.models import load_model_registry
from slopecoach_ml.openmmlab import configured_device, openmmlab_preflight
from slopecoach_ml.pose import render_debug_overlay
from slopecoach_ml.pose.mmpose_provider import MMPoseRTMWPoseProvider, OpenMMLabMMPoseBackend
from slopecoach_ml.quality import VideoQualityGate, VideoQualityStatus
from slopecoach_ml.reference import (
    ReferenceAnalysisConfig,
    ReferenceAnalysisContext,
    analyze_pose_frame,
    load_golden_fixture,
)
from slopecoach_ml.temporal import run_temporal_golden, run_turn_golden
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
    real_video.add_argument("--debug-dir")
    real_video.add_argument("--max-debug-frames", type=int, default=10)
    real_video.add_argument("--input-non-mirrored", action="store_true")
    target_identity = subparsers.add_parser(
        "benchmark-target-identity", help="run A3 target-identity video benchmark"
    )
    target_identity.add_argument("video")
    target_identity.add_argument("--sample-fps", type=float, default=2.0)
    target_identity.add_argument("--output")
    target_identity.add_argument("--debug-dir")
    target_identity.add_argument("--max-debug-frames", type=int, default=12)
    target_identity.add_argument("--input-non-mirrored", action="store_true")
    target_identity.add_argument("--target-gt")
    prepare_gt = subparsers.add_parser(
        "prepare-target-gt", help="create an unlabeled manual target-GT review template"
    )
    prepare_gt.add_argument("video")
    prepare_gt.add_argument("--sample-fps", type=float, default=5.0)
    prepare_gt.add_argument("--output", required=True)
    prepare_gt.add_argument("--review-dir")
    temporal_golden = subparsers.add_parser(
        "temporal-golden", help="run deterministic A4 temporal pose Golden"
    )
    temporal_golden.add_argument(
        "--fixture", default=str(_root() / "fixtures/golden_temporal_pose_001.json")
    )
    temporal_golden.add_argument("--output")
    turn_golden = subparsers.add_parser(
        "turn-golden", help="run deterministic A4 turn signal Golden"
    )
    turn_golden.add_argument(
        "--fixture", default=str(_root() / "fixtures/golden_turn_signal_001.json")
    )
    turn_golden.add_argument("--output")
    temporal_turns = subparsers.add_parser(
        "benchmark-temporal-turns", help="run A4 temporal pose and turn benchmark"
    )
    temporal_turns.add_argument("video")
    temporal_turns.add_argument("--sample-fps", type=float, default=5.0)
    temporal_turns.add_argument("--output")
    temporal_turns.add_argument("--debug-dir")
    temporal_turns.add_argument("--max-debug-frames", type=int, default=12)
    temporal_turns.add_argument("--input-non-mirrored", action="store_true")
    return parser


def _real_providers():
    device = configured_device()
    required = {
        "detector_config": os.getenv("SLOPECOACH_DETECTOR_CONFIG"),
        "detector_checkpoint": os.getenv("SLOPECOACH_DETECTOR_CHECKPOINT"),
        "pose_config": os.getenv("SLOPECOACH_POSE_CONFIG"),
        "pose_checkpoint": os.getenv("SLOPECOACH_POSE_CHECKPOINT"),
    }
    missing_configs = [
        name
        for name in ("detector_config", "pose_config")
        if not required[name] or not Path(required[name]).is_file()
    ]
    missing_checkpoints = [
        name
        for name in ("detector_checkpoint", "pose_checkpoint")
        if not required[name] or not Path(required[name]).is_file()
    ]
    if missing_configs:
        raise RuntimeError(f"MODEL_CONFIG_MISSING: {', '.join(missing_configs)}")
    if missing_checkpoints:
        raise RuntimeError(f"MODEL_CHECKPOINT_MISSING: {', '.join(missing_checkpoints)}")
    started = time.perf_counter()
    detector = MMDetPersonDetectorProvider(
        OpenMMLabMMDetBackend(
            required["detector_config"], required["detector_checkpoint"], device=device
        )
    )
    detector_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pose = MMPoseRTMWPoseProvider(
        OpenMMLabMMPoseBackend(required["pose_config"], required["pose_checkpoint"], device=device)
    )
    pose_load_seconds = time.perf_counter() - started
    return (
        detector,
        pose,
        device,
        {
            "detector_seconds": detector_load_seconds,
            "pose_seconds": pose_load_seconds,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "temporal-golden":
        result = run_temporal_golden(args.fixture)
        _write_json(result, args.output)
        return 0 if result["golden_passed"] else 1
    if args.command == "turn-golden":
        result = run_turn_golden(args.fixture)
        _write_json(result, args.output)
        return 0 if result["golden_passed"] else 1
    if args.command == "prepare-target-gt":
        result = prepare_target_gt_template(
            args.video,
            sample_fps=args.sample_fps,
            output_path=args.output,
            review_dir=args.review_dir,
        )
        _write_json(result, None)
        return 0
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
        detector, pose_provider, device, model_load = _real_providers()
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
        overlay_seconds = None
        if args.overlay:
            stage = time.perf_counter()
            render_debug_overlay(image, frame, args.overlay, provider_name=pose_provider.name)
            overlay_seconds = time.perf_counter() - stage
        registry = load_model_registry(_root() / "models/registry.json")
        detector_model = registry["rtmdet-m-640-coco-obj365-person"]
        pose_model = registry["rtmw-l-cocktail14-256x192"]
        payload = {
            "input": args.image,
            "provider": pose_provider.name,
            "detector": detector.name,
            "pose_model": "rtmw-l-cocktail14-256x192@20231122",
            "detector_checkpoint_sha256": detector_model.checkpoint_sha256,
            "pose_checkpoint_sha256": pose_model.checkpoint_sha256,
            "device": device,
            "model_load": model_load,
            "frame_geometry": geometry.to_dict(),
            "person_count": len(frame.persons),
            "raw_output_joint_count": pose_provider.last_raw_joint_count,
            "canonical_output_joint_count": len(frame.persons[0].keypoints)
            if frame.persons
            else None,
            "persons": [person.to_dict() for person in frame.persons],
            "canonical_joint_schema": frame.joint_schema,
            "reference_analysis": result.to_dict(),
            "warnings": list(result.warnings),
            "limitations": list(result.limitations),
            "timing": {
                "total_processing_seconds": time.perf_counter() - started,
                "detector_seconds": detector_seconds,
                "pose_seconds": pose_provider.last_backend_seconds,
                "canonical_adapter_seconds": pose_provider.last_adapter_seconds,
                "biomechanics_seconds": biomechanics_seconds,
                "overlay_seconds": overlay_seconds,
                "provider_total_pose_stage_seconds": pose_seconds,
                "fps": None,
            },
        }
        _write_json(payload, args.output)
        return 0
    if args.command in {
        "benchmark-real-pose",
        "benchmark-target-identity",
        "benchmark-temporal-turns",
    }:
        if not args.input_non_mirrored:
            raise RuntimeError(
                "MIRROR_STATE_UNRESOLVED: pass --input-non-mirrored to attest input state"
            )
        maximum_debug = 10 if args.command == "benchmark-real-pose" else 12
        if args.max_debug_frames < 0 or args.max_debug_frames > maximum_debug:
            raise ValueError(f"max-debug-frames must be in [0, {maximum_debug}]")
        detector, pose_provider, device, model_load = _real_providers()
        registry = load_model_registry(_root() / "models/registry.json")
        sampler = OpenCVVideoSampler(args.video, sample_fps=args.sample_fps)
        warmup_frames = 0
        warmup_iterator = iter(sampler)
        try:
            warmup = next(warmup_iterator)
        except StopIteration:
            warmup = None
        finally:
            close = getattr(warmup_iterator, "close", None)
            if close:
                close()
        if warmup is not None:
            warmup_detections = detector.detect(warmup.image, warmup.geometry)
            pose_provider.estimate_detections(
                warmup.image,
                warmup_detections,
                warmup.geometry,
                timestamp_us=warmup.timestamp_us,
                frame_index=warmup.frame_index,
            )
            warmup_frames = 1
        if args.command == "benchmark-target-identity":
            target_ground_truth = (
                load_target_ground_truth(args.target_gt, args.video) if args.target_gt else None
            )
            ground_truth_status = "NOT_AVAILABLE"
            if target_ground_truth:
                labeled = any(
                    frame.target_state.value in {"PRESENT", "ABSENT"}
                    for frame in target_ground_truth.frames
                )
                ground_truth_status = (
                    "AVAILABLE" if labeled else "TEMPLATE_CREATED_REQUIRES_HUMAN_LABELING"
                )
            debug_collector = TargetIdentityDebugCollector() if args.debug_dir else None
            report = benchmark_target_identity_frames(
                input_path=args.video,
                frames=OpenCVVideoSampler(args.video, sample_fps=args.sample_fps),
                detector=detector,
                pose_provider=pose_provider,
                detector_model=registry["rtmdet-m-640-coco-obj365-person"].to_dict(),
                pose_model=registry["rtmw-l-cocktail14-256x192"].to_dict(),
                device=device,
                sample_fps=args.sample_fps,
                model_load=model_load,
                warmup_frames=warmup_frames,
                frame_observer=debug_collector.observe if debug_collector else None,
                target_ground_truth=target_ground_truth,
                ground_truth_status=ground_truth_status,
            )
            report["debug_artifacts"] = (
                debug_collector.write(
                    args.debug_dir,
                    report["frame_observations"],
                    max_frames=args.max_debug_frames,
                )
                if debug_collector
                else {"selected_frame_indices": [], "overlay_paths": [], "contact_sheet": None}
            )
            if (
                target_ground_truth
                and ground_truth_status == "AVAILABLE"
                and debug_collector
                and args.debug_dir
            ):
                report["debug_artifacts"].update(
                    write_ground_truth_comparison(
                        args.debug_dir,
                        report["debug_artifacts"],
                        report["frame_observations"],
                        target_ground_truth,
                        report.get("frame_gt_classifications", []),
                    )
                )
            _write_json(report, args.output)
            return 0
        if args.command == "benchmark-temporal-turns":
            collector = TemporalTurnCollector(keep_images=bool(args.debug_dir))
            identity_template = (
                _root()
                / "benchmarks/ski_bench/annotations"
                / f"{Path(args.video).stem}.target.json"
            )
            identity_gt_status = (
                "TEMPLATE_CREATED_REQUIRES_HUMAN_LABELING"
                if identity_template.is_file()
                else "NOT_AVAILABLE"
            )
            report = benchmark_temporal_turns_frames(
                input_path=args.video,
                frames=OpenCVVideoSampler(args.video, sample_fps=args.sample_fps),
                detector=detector,
                pose_provider=pose_provider,
                detector_model=registry["rtmdet-m-640-coco-obj365-person"].to_dict(),
                pose_model=registry["rtmw-l-cocktail14-256x192"].to_dict(),
                device=device,
                sample_fps=args.sample_fps,
                model_load=model_load,
                warmup_frames=warmup_frames,
                collector=collector,
                target_identity_gt_status=identity_gt_status,
            )
            report["debug_artifacts"] = (
                write_temporal_debug_artifacts(
                    args.debug_dir, report, collector, max_frames=args.max_debug_frames
                )
                if args.debug_dir
                else {
                    "selected_frame_indices": [],
                    "overlay_paths": [],
                    "contact_sheet": None,
                    "temporal_trace": None,
                    "turn_signal": None,
                    "turn_events": None,
                }
            )
            _write_json(report, args.output)
            return 0
        debug_collector = RealPoseDebugCollector() if args.debug_dir else None
        report = benchmark_real_pose_frames(
            input_path=args.video,
            input_kind="REAL_VIDEO",
            frames=OpenCVVideoSampler(args.video, sample_fps=args.sample_fps),
            detector=detector,
            pose_provider=pose_provider,
            detector_model=registry["rtmdet-m-640-coco-obj365-person"].to_dict(),
            pose_model=registry["rtmw-l-cocktail14-256x192"].to_dict(),
            device=device,
            sample_fps=args.sample_fps,
            model_load=model_load,
            warmup_frames=warmup_frames,
            frame_observer=debug_collector.observe if debug_collector else None,
        )
        if debug_collector:
            started = time.perf_counter()
            report["debug_artifacts"] = debug_collector.write(
                args.debug_dir,
                report["frame_observations"],
                provider_name=pose_provider.name,
                max_frames=args.max_debug_frames,
            )
            report["performance"]["overlay_seconds"] = time.perf_counter() - started
        else:
            report["debug_artifacts"] = {
                "selected_frame_indices": [],
                "overlay_paths": [],
                "contact_sheet": None,
            }
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
