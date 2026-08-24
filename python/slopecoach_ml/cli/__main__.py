from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from slopecoach_ml.analysis_result import run_analysis_result_golden
from slopecoach_ml.benchmark import (
    RealPoseDebugCollector,
    SportTypeBenchmarkCollector,
    TargetIdentityDebugCollector,
    TemporalTurnCollector,
    benchmark_analysis_result_artifact,
    benchmark_biomechanics_frames,
    benchmark_diagnosis_artifact,
    benchmark_golden,
    benchmark_real_pose_frames,
    benchmark_scoring_coach_artifact,
    benchmark_sport_type_frames,
    benchmark_target_identity_frames,
    benchmark_temporal_turns_frames,
    benchmark_video,
    execute_biomechanics_dataset,
    load_real_dataset_manifest,
    prepare_real_dataset_manifest,
    write_biomechanics_debug_artifacts,
    write_biomechanics_overlay_video,
    write_ground_truth_comparison,
    write_sport_type_debug_artifacts,
    write_temporal_debug_artifacts,
)
from slopecoach_ml.biomechanics.golden import run_biomechanics_golden
from slopecoach_ml.coach import run_coach_golden
from slopecoach_ml.detection.mmdet_provider import (
    MMDetPersonDetectorProvider,
    OpenMMLabMMDetBackend,
)
from slopecoach_ml.diagnosis import run_diagnosis_golden
from slopecoach_ml.identity import (
    ManualTargetSeed,
    load_target_ground_truth,
    prepare_target_gt_template,
)
from slopecoach_ml.models import load_model_registry
from slopecoach_ml.openmmlab import configured_device, openmmlab_preflight
from slopecoach_ml.pose import render_debug_overlay
from slopecoach_ml.pose.mmpose_provider import MMPoseRTMWPoseProvider, OpenMMLabMMPoseBackend
from slopecoach_ml.product import build_mvp_analysis_payload, select_user_sport_type
from slopecoach_ml.quality import VideoQualityGate, VideoQualityStatus
from slopecoach_ml.reference import (
    ReferenceAnalysisConfig,
    ReferenceAnalysisContext,
    analyze_pose_frame,
    load_golden_fixture,
)
from slopecoach_ml.scoring import run_a8_provenance_golden, run_scorecard_golden
from slopecoach_ml.sport_type import (
    ClipVisualSportEvidenceProvider,
    FailedSportEvidenceProvider,
    MMDetEquipmentSportEvidenceProvider,
    NotConfiguredEquipmentSportEvidenceProvider,
    NotConfiguredVisualSportEvidenceProvider,
    OpenAIClipVisualSportBackend,
    OpenMMLabEquipmentBackend,
    SportEvidenceKind,
    SportType,
    equipment_provider_doctor,
    prepare_visual_sport_model,
    run_sport_type_golden,
    sha256_file,
    visual_provider_doctor,
)
from slopecoach_ml.sport_type.calibration import (
    apply_calibrated_fusion,
    build_calibration_dataset,
    fit_calibration_artifact,
    prepare_sport_type_gt,
    run_calibration_golden,
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


def _nonnegative_finite_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("target seed time must be numeric") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("target seed time must be finite and >= 0")
    return parsed


def _source_pixel_point(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("target seed point must be exactly X,Y")
    try:
        x_px, y_px = (float(item) for item in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("target seed point X,Y must be numeric") from error
    if not math.isfinite(x_px) or not math.isfinite(y_px):
        raise argparse.ArgumentTypeError("target seed point X,Y must be finite")
    return x_px, y_px


class _SlopeCoachArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.command in {"benchmark-biomechanics", "analyze-video"}:
            has_time = parsed.target_seed_time is not None
            has_point = parsed.target_seed_point is not None
            if has_time != has_point:
                self.error("--target-seed-time and --target-seed-point must be supplied together")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = _SlopeCoachArgumentParser(description="SlopeCoach research/reference CLI")
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
    equipment_doctor = subparsers.add_parser(
        "sport-equipment-doctor", help="validate the optional full-COCO equipment provider"
    )
    equipment_doctor.add_argument("--equipment-config")
    equipment_doctor.add_argument("--equipment-checkpoint")
    visual_doctor = subparsers.add_parser(
        "sport-visual-doctor", help="validate the optional OpenAI CLIP visual provider"
    )
    visual_doctor.add_argument("--visual-checkpoint")
    visual_doctor.add_argument("--visual-model-name", choices=("ViT-B/32",), default="ViT-B/32")
    prepare_visual = subparsers.add_parser(
        "prepare-visual-sport-model", help="explicitly download the official CLIP ViT-B/32 weight"
    )
    prepare_visual.add_argument(
        "--destination", default=str(_root() / "artifacts/models/a6_2/openai_clip")
    )
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
    biomechanics_golden = subparsers.add_parser(
        "biomechanics-golden", help="run deterministic A5 temporal biomechanics Golden"
    )
    biomechanics_golden.add_argument(
        "--fixture",
        default=str(_root() / "fixtures/golden_temporal_biomechanics_001.json"),
    )
    biomechanics_golden.add_argument("--output")
    sport_type_golden = subparsers.add_parser(
        "sport-type-golden", help="run deterministic A6 SportType fusion Golden"
    )
    sport_type_golden.add_argument(
        "--fixture", default=str(_root() / "fixtures/golden_sport_type_001.json")
    )
    sport_type_golden.add_argument("--output")
    calibration_golden = subparsers.add_parser(
        "sport-calibration-golden", help="run deterministic A6.3 calibrated fusion Golden"
    )
    calibration_golden.add_argument(
        "--fixture", default=str(_root() / "fixtures/golden_sport_calibration_001.json")
    )
    calibration_golden.add_argument("--output")
    diagnosis_golden = subparsers.add_parser(
        "diagnosis-golden", help="run deterministic A7 provisional diagnosis Golden"
    )
    diagnosis_golden.add_argument(
        "--fixture", default=str(_root() / "fixtures/golden_diagnosis_001.json")
    )
    diagnosis_golden.add_argument("--output")
    diagnosis_benchmark = subparsers.add_parser(
        "benchmark-diagnosis", help="run artifact-only A7 diagnosis benchmark"
    )
    diagnosis_benchmark.add_argument("artifact")
    diagnosis_benchmark.add_argument(
        "--sport-type", choices=("auto", "ski", "snowboard"), default="auto"
    )
    diagnosis_benchmark.add_argument("--output")
    scorecard_golden = subparsers.add_parser(
        "scorecard-golden", help="run deterministic A8 structure-only ScoreCard Golden"
    )
    scorecard_golden.add_argument(
        "--fixture", default=str(_root() / "fixtures/golden_scorecard_001.json")
    )
    scorecard_golden.add_argument("--output")
    coach_golden = subparsers.add_parser(
        "coach-golden", help="run deterministic A8 controlled zh-CN coach Golden"
    )
    coach_golden.add_argument("--fixture", default=str(_root() / "fixtures/golden_coach_001.json"))
    coach_golden.add_argument("--output")
    provenance_golden = subparsers.add_parser(
        "a8-provenance-golden", help="run deterministic A8.1 provenance Golden"
    )
    provenance_golden.add_argument(
        "--fixture",
        default=str(_root() / "fixtures/golden_a8_1_provenance_001.json"),
    )
    provenance_golden.add_argument("--output")
    scoring_coach = subparsers.add_parser(
        "benchmark-scoring-coach", help="run artifact-only A8 scorecard/coach benchmark"
    )
    scoring_coach.add_argument("artifact")
    scoring_coach.add_argument("--output")
    analysis_result_golden = subparsers.add_parser(
        "analysis-result-golden", help="run deterministic A9 result/report Golden"
    )
    analysis_result_golden.add_argument(
        "--fixture", default=str(_root() / "fixtures/golden_analysis_result_001.json")
    )
    analysis_result_golden.add_argument("--output")
    analysis_result_benchmark = subparsers.add_parser(
        "benchmark-analysis-result", help="assemble A9 contracts from an A7 artifact"
    )
    analysis_result_benchmark.add_argument("artifact")
    analysis_result_benchmark.add_argument("--output")
    prepare_sport_gt = subparsers.add_parser(
        "prepare-sport-type-gt", help="create UNLABELED manual SportType GT templates"
    )
    prepare_sport_gt.add_argument("--manifest", required=True)
    prepare_sport_gt.add_argument("--output-dir", required=True)
    build_calibration = subparsers.add_parser(
        "build-sport-calibration-dataset", help="extract source-level samples from A6 artifacts"
    )
    build_calibration.add_argument("artifacts", nargs="+")
    build_calibration.add_argument("--annotations-dir")
    build_calibration.add_argument("--output", required=True)
    fit_calibration = subparsers.add_parser(
        "fit-sport-evidence-calibration", help="fit research-only provider calibrators"
    )
    fit_calibration.add_argument("dataset")
    fit_calibration.add_argument("--annotations-dir")
    fit_calibration.add_argument("--output", required=True)
    apply_calibration = subparsers.add_parser(
        "apply-sport-evidence-calibration", help="apply calibration to an existing A6 artifact"
    )
    apply_calibration.add_argument("artifact")
    apply_calibration.add_argument("--calibration", required=True)
    apply_calibration.add_argument("--output")
    temporal_turns = subparsers.add_parser(
        "benchmark-temporal-turns", help="run A4 temporal pose and turn benchmark"
    )
    temporal_turns.add_argument("video")
    temporal_turns.add_argument("--sample-fps", type=float, default=5.0)
    temporal_turns.add_argument("--output")
    temporal_turns.add_argument("--debug-dir")
    temporal_turns.add_argument("--max-debug-frames", type=int, default=12)
    temporal_turns.add_argument("--input-non-mirrored", action="store_true")
    biomechanics = subparsers.add_parser(
        "benchmark-biomechanics", help="run A5 temporal biomechanics benchmark"
    )
    biomechanics.add_argument("video")
    biomechanics.add_argument("--sample-fps", type=float, default=5.0)
    biomechanics.add_argument("--output")
    biomechanics.add_argument("--debug-dir")
    biomechanics.add_argument(
        "--overlay-video", help="write a sampled pose/biomechanics debug MP4 without model reruns"
    )
    biomechanics.add_argument(
        "--target-seed-time",
        type=_nonnegative_finite_seconds,
        help="manual identity initialization time on the source-video timeline, in seconds",
    )
    biomechanics.add_argument(
        "--target-seed-point",
        type=_source_pixel_point,
        metavar="X,Y",
        help="manual identity initialization point in upright SourcePixel2D coordinates",
    )
    biomechanics.add_argument("--max-debug-frames", type=int, default=12)
    biomechanics.add_argument("--input-non-mirrored", action="store_true")
    analyze_video = subparsers.add_parser(
        "analyze-video",
        help="run the MVP video path with an explicit user-selected SportType",
    )
    analyze_video.add_argument("video")
    analyze_video.add_argument(
        "--sport-type",
        type=str.upper,
        choices=("SKI", "SNOWBOARD"),
        required=True,
    )
    analyze_video.add_argument("--sample-fps", type=float, default=5.0)
    analyze_video.add_argument("--output")
    analyze_video.add_argument("--debug-dir")
    analyze_video.add_argument(
        "--overlay-video", help="write a sampled pose/biomechanics debug MP4 without model reruns"
    )
    analyze_video.add_argument(
        "--target-seed-time",
        type=_nonnegative_finite_seconds,
        help="manual identity initialization time on the source-video timeline, in seconds",
    )
    analyze_video.add_argument(
        "--target-seed-point",
        type=_source_pixel_point,
        metavar="X,Y",
        help="manual identity initialization point in upright SourcePixel2D coordinates",
    )
    analyze_video.add_argument("--max-debug-frames", type=int, default=12)
    analyze_video.add_argument("--input-non-mirrored", action="store_true")
    sport_type = subparsers.add_parser(
        "benchmark-sport-type", help="run A6 auto/user SportType foundation benchmark"
    )
    sport_type.add_argument("video")
    sport_type.add_argument("--sample-fps", type=float, default=5.0)
    sport_type.add_argument("--sport-type", choices=("auto", "ski", "snowboard"), default="auto")
    sport_type.add_argument("--equipment-provider", choices=("none", "rtmdet-coco"), default="none")
    sport_type.add_argument("--equipment-config")
    sport_type.add_argument("--equipment-checkpoint")
    sport_type.add_argument("--visual-provider", choices=("none", "openai-clip"), default="none")
    sport_type.add_argument("--visual-checkpoint")
    sport_type.add_argument("--visual-model-name", choices=("ViT-B/32",), default="ViT-B/32")
    sport_type.add_argument("--calibration-artifact")
    sport_type.add_argument("--source-video-id")
    sport_type.add_argument("--output")
    sport_type.add_argument("--debug-dir")
    sport_type.add_argument("--max-debug-frames", type=int, default=12)
    sport_type.add_argument("--input-non-mirrored", action="store_true")
    prepare_dataset = subparsers.add_parser(
        "prepare-biomechanics-dataset", help="prepare a local-only A5.2 real-video manifest"
    )
    prepare_dataset.add_argument("--video-dir", required=True)
    prepare_dataset.add_argument("--output", required=True)
    dataset = subparsers.add_parser(
        "benchmark-biomechanics-dataset", help="run sequential A5.2 dataset robustness benchmark"
    )
    dataset.add_argument("manifest")
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--per-clip-output-dir", required=True)
    dataset.add_argument("--debug-dir")
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


def _keep_temporal_images(args: argparse.Namespace) -> bool:
    return bool(args.debug_dir or getattr(args, "overlay_video", None))


def _temporal_collector(args: argparse.Namespace):
    collector_type = (
        SportTypeBenchmarkCollector
        if args.command == "benchmark-sport-type"
        else TemporalTurnCollector
    )
    return collector_type(keep_images=_keep_temporal_images(args))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-sport-type-gt":
        _write_json(prepare_sport_type_gt(args.manifest, args.output_dir), None)
        return 0
    if args.command == "benchmark-diagnosis":
        result = benchmark_diagnosis_artifact(args.artifact, sport_type=args.sport_type)
        _write_json(result, args.output)
        return 0
    if args.command == "benchmark-scoring-coach":
        result = benchmark_scoring_coach_artifact(args.artifact)
        _write_json(result, args.output)
        return 0
    if args.command == "benchmark-analysis-result":
        result = benchmark_analysis_result_artifact(args.artifact)
        _write_json(result, args.output)
        return 0
    if args.command == "build-sport-calibration-dataset":
        result = build_calibration_dataset(args.artifacts, args.annotations_dir)
        _write_json(result, args.output)
        return 0
    if args.command == "fit-sport-evidence-calibration":
        dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
        if args.annotations_dir:
            source_artifacts = [item["source_artifact"] for item in dataset.get("clips", [])]
            dataset = build_calibration_dataset(source_artifacts, args.annotations_dir)
        result = fit_calibration_artifact(dataset)
        _write_json(result, args.output)
        return 0 if result["status"] == "RESEARCH_CALIBRATION_AVAILABLE" else 3
    if args.command == "apply-sport-evidence-calibration":
        dataset = build_calibration_dataset([args.artifact])
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        result = {
            "raw_auto_decision": dataset["raw_auto_decisions"][0],
            "raw_provider_summaries": dataset["source_samples"],
            "calibration_artifact_sha256": calibration.get("calibration_artifact_sha256"),
            "calibrated_fusion_result": apply_calibrated_fusion(
                dataset["source_samples"], calibration
            ),
            "CALIBRATED_FUSION_CONTROLS_ROUTING": False,
        }
        _write_json(result, args.output)
        return 0
    if args.command == "prepare-biomechanics-dataset":
        result = prepare_real_dataset_manifest(args.video_dir, args.output)
        _write_json(result, None)
        return 0
    if args.command == "benchmark-biomechanics-dataset":
        manifest = load_real_dataset_manifest(args.manifest)
        registry = load_model_registry(_root() / "models/registry.json")
        providers = {}

        def benchmark_clip(clip, path, debug_path):
            if not providers:
                detector, pose_provider, device, model_load = _real_providers()
                providers.update(
                    detector=detector,
                    pose_provider=pose_provider,
                    device=device,
                    model_load=model_load,
                )
            detector = providers["detector"]
            pose_provider = providers["pose_provider"]
            device = providers["device"]
            model_load = providers["model_load"]
            collector = TemporalTurnCollector(keep_images=bool(debug_path))
            sampler = OpenCVVideoSampler(path, sample_fps=clip.sample_fps)
            warmup_iterator = iter(sampler)
            try:
                warmup = next(warmup_iterator)
            except StopIteration:
                warmup = None
            finally:
                close = getattr(warmup_iterator, "close", None)
                if close:
                    close()
            warmup_frames = 0
            if warmup is not None:
                detections = detector.detect(warmup.image, warmup.geometry)
                pose_provider.estimate_detections(
                    warmup.image,
                    detections,
                    warmup.geometry,
                    timestamp_us=warmup.timestamp_us,
                    frame_index=warmup.frame_index,
                )
                warmup_frames = 1
            identity_template = (
                _root() / "benchmarks/ski_bench/annotations" / f"{path.stem}.target.json"
            )
            report = benchmark_biomechanics_frames(
                input_path=path,
                frames=OpenCVVideoSampler(path, sample_fps=clip.sample_fps),
                detector=detector,
                pose_provider=pose_provider,
                detector_model=registry["rtmdet-m-640-coco-obj365-person"].to_dict(),
                pose_model=registry["rtmw-l-cocktail14-256x192"].to_dict(),
                device=device,
                sample_fps=clip.sample_fps,
                model_load=model_load,
                warmup_frames=warmup_frames,
                collector=collector,
                target_identity_gt_status=(
                    "TEMPLATE_CREATED_REQUIRES_HUMAN_LABELING"
                    if identity_template.is_file()
                    else "NOT_AVAILABLE"
                ),
            )
            report["debug_artifacts"] = (
                write_biomechanics_debug_artifacts(debug_path, report, collector)
                if debug_path
                else {}
            )
            report.pop("_upstream_debug_report", None)
            return report

        result = execute_biomechanics_dataset(
            manifest,
            benchmark_clip,
            per_clip_output_dir=args.per_clip_output_dir,
            debug_dir=args.debug_dir,
        )
        _write_json(result, args.output)
        return 0
    if args.command == "temporal-golden":
        result = run_temporal_golden(args.fixture)
        _write_json(result, args.output)
        return 0 if result["golden_passed"] else 1
    if args.command == "turn-golden":
        result = run_turn_golden(args.fixture)
        _write_json(result, args.output)
        return 0 if result["golden_passed"] else 1
    if args.command == "biomechanics-golden":
        result = run_biomechanics_golden(args.fixture)
        _write_json(result, args.output)
        return 0 if result["golden_passed"] else 1
    if args.command == "sport-type-golden":
        result = run_sport_type_golden(args.fixture)
        _write_json(result, args.output)
        return 0 if result["golden_passed"] else 1
    if args.command == "sport-calibration-golden":
        result = run_calibration_golden(args.fixture)
        _write_json(result, args.output)
        return 0 if result["golden_passed"] else 1
    if args.command == "diagnosis-golden":
        result = run_diagnosis_golden(args.fixture)
        _write_json(result, args.output)
        return 0 if result["golden_passed"] else 1
    if args.command == "scorecard-golden":
        result = run_scorecard_golden(args.fixture)
        _write_json(result, args.output)
        return 0 if result["golden_passed"] else 1
    if args.command == "coach-golden":
        result = run_coach_golden(args.fixture)
        _write_json(result, args.output)
        return 0 if result["golden_passed"] else 1
    if args.command == "a8-provenance-golden":
        result = run_a8_provenance_golden(args.fixture)
        _write_json(result, args.output)
        return 0 if result["golden_passed"] else 1
    if args.command == "analysis-result-golden":
        result = run_analysis_result_golden(args.fixture)
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
    if args.command == "sport-equipment-doctor":
        config_path = args.equipment_config or os.getenv("SLOPECOACH_EQUIPMENT_DETECTOR_CONFIG")
        checkpoint_path = args.equipment_checkpoint or os.getenv(
            "SLOPECOACH_EQUIPMENT_DETECTOR_CHECKPOINT"
        )
        expected_sha = _equipment_registry_sha()
        report = equipment_provider_doctor(
            config_path,
            checkpoint_path,
            device=configured_device(),
            expected_checkpoint_sha256=expected_sha,
        )
        _write_json(report, None)
        return 0 if report["EQUIPMENT_SPORT_PROVIDER_READINESS"].startswith("READY") else 3
    if args.command == "prepare-visual-sport-model":
        report = prepare_visual_sport_model(args.destination)
        _write_json(report, None)
        return 0
    if args.command == "sport-visual-doctor":
        registry_model = _visual_registry_model()
        checkpoint_path = args.visual_checkpoint or os.getenv("SLOPECOACH_VISUAL_SPORT_CHECKPOINT")
        report = visual_provider_doctor(
            checkpoint_path,
            device=configured_device(),
            expected_checkpoint_sha256=(
                registry_model.checkpoint_sha256 if registry_model else None
            ),
            implementation_commit=registry_model.model_version if registry_model else None,
        )
        _write_json(report, None)
        return 0 if report["VISUAL_SPORT_PROVIDER_READINESS"].startswith("READY") else 3
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
        "benchmark-biomechanics",
        "benchmark-sport-type",
        "analyze-video",
    }:
        product_sport_type = (
            select_user_sport_type(args.sport_type) if args.command == "analyze-video" else None
        )
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
        if args.command in {
            "benchmark-temporal-turns",
            "benchmark-biomechanics",
            "benchmark-sport-type",
            "analyze-video",
        }:
            collector = _temporal_collector(args)
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
            benchmark_runner = {
                "benchmark-temporal-turns": benchmark_temporal_turns_frames,
                "benchmark-biomechanics": benchmark_biomechanics_frames,
                "benchmark-sport-type": benchmark_sport_type_frames,
                "analyze-video": benchmark_biomechanics_frames,
            }[args.command]
            runner_kwargs = dict(
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
            if (
                args.command in {"benchmark-biomechanics", "analyze-video"}
                and args.target_seed_time is not None
            ):
                runner_kwargs["manual_target_seed"] = ManualTargetSeed(
                    args.target_seed_time,
                    *args.target_seed_point,
                )
            if args.command == "benchmark-sport-type":
                runner_kwargs["user_selection"] = {
                    "auto": None,
                    "ski": SportType.SKI,
                    "snowboard": SportType.SNOWBOARD,
                }[args.sport_type]
                runner_kwargs["evidence_providers"] = (
                    _provider_or_failure(
                        lambda: _equipment_provider(args),
                        "openmmlab-rtmdet-tiny-coco-equipment",
                        SportEvidenceKind.EQUIPMENT,
                    )
                    if args.equipment_provider == "rtmdet-coco"
                    else NotConfiguredEquipmentSportEvidenceProvider(),
                    _provider_or_failure(
                        lambda: _visual_provider(args),
                        "openai-clip-vit-b32-visual-sport",
                        SportEvidenceKind.VISUAL_CLASSIFIER,
                    )
                    if args.visual_provider == "openai-clip"
                    else NotConfiguredVisualSportEvidenceProvider(),
                )
                runner_kwargs["calibration_artifact"] = (
                    json.loads(Path(args.calibration_artifact).read_text(encoding="utf-8"))
                    if args.calibration_artifact
                    else None
                )
                runner_kwargs["source_video_id"] = args.source_video_id
            report = benchmark_runner(**runner_kwargs)
            report["debug_artifacts"] = (
                (
                    write_sport_type_debug_artifacts
                    if args.command == "benchmark-sport-type"
                    else write_biomechanics_debug_artifacts
                    if args.command == "benchmark-biomechanics"
                    else write_temporal_debug_artifacts
                )(args.debug_dir, report, collector, max_frames=args.max_debug_frames)
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
            if args.command in {"benchmark-biomechanics", "analyze-video"} and args.overlay_video:
                report["debug_artifacts"]["overlay_video"] = write_biomechanics_overlay_video(
                    args.overlay_video,
                    report,
                    collector,
                    fps=args.sample_fps,
                )
            report.pop("_upstream_biomechanics_report", None)
            report.pop("_equipment_debug_frames", None)
            report.pop("_visual_debug_frames", None)
            report.pop("_upstream_debug_report", None)
            if args.command == "analyze-video":
                assert product_sport_type is not None
                report = build_mvp_analysis_payload(
                    video=args.video,
                    sport_type=product_sport_type,
                    biomechanics_report=report,
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


def _equipment_registry_sha() -> str | None:
    registry = load_model_registry(_root() / "models/registry.json")
    model = registry.get("rtmdet-tiny-640-coco-equipment")
    return model.checkpoint_sha256 if model else None


def _visual_registry_model():
    return load_model_registry(_root() / "models/registry.json").get(
        "openai-clip-vit-b32-visual-sport"
    )


def _provider_or_failure(factory, name, kind):
    try:
        return factory()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        return FailedSportEvidenceProvider(
            name=name,
            kind=kind,
            error=f"{type(error).__name__}: {error}",
        )


def _equipment_provider(args):
    config_path = args.equipment_config or os.getenv("SLOPECOACH_EQUIPMENT_DETECTOR_CONFIG")
    checkpoint_path = args.equipment_checkpoint or os.getenv(
        "SLOPECOACH_EQUIPMENT_DETECTOR_CHECKPOINT"
    )
    if not config_path or not Path(config_path).is_file():
        raise RuntimeError("EQUIPMENT_MODEL_CONFIG_MISSING")
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise RuntimeError("EQUIPMENT_MODEL_CHECKPOINT_MISSING")
    actual_sha = sha256_file(checkpoint_path)
    registry_model = load_model_registry(_root() / "models/registry.json").get(
        "rtmdet-tiny-640-coco-equipment"
    )
    expected_sha = registry_model.checkpoint_sha256 if registry_model else None
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError("EQUIPMENT_CHECKPOINT_SHA256_MISMATCH")
    device = configured_device()
    backend = OpenMMLabEquipmentBackend(config_path, checkpoint_path, device=device)
    return MMDetEquipmentSportEvidenceProvider(
        backend,
        device=device,
        config_path=config_path,
        config_source=registry_model.config_source if registry_model else None,
        checkpoint_path=checkpoint_path,
        checkpoint_source=registry_model.checkpoint_source if registry_model else None,
        checkpoint_sha256=actual_sha,
        model_load_seconds=backend.model_load_seconds,
    )


def _visual_provider(args):
    checkpoint_path = args.visual_checkpoint or os.getenv("SLOPECOACH_VISUAL_SPORT_CHECKPOINT")
    if args.visual_model_name != "ViT-B/32":
        raise RuntimeError("VISUAL_MODEL_UNSUPPORTED")
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise RuntimeError("VISUAL_MODEL_CHECKPOINT_MISSING")
    actual_sha = sha256_file(checkpoint_path)
    registry_model = _visual_registry_model()
    expected_sha = registry_model.checkpoint_sha256 if registry_model else None
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError("VISUAL_CHECKPOINT_SHA256_MISMATCH")
    device = configured_device()
    backend = OpenAIClipVisualSportBackend(checkpoint_path, device=device)
    return ClipVisualSportEvidenceProvider(
        backend,
        device=device,
        implementation_commit=registry_model.model_version if registry_model else None,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=actual_sha,
        model_load_seconds=backend.model_load_seconds,
        text_prototype_seconds=backend.text_prototype_seconds,
        input_resolution=backend.input_resolution,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
