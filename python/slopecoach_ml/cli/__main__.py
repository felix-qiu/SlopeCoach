from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slopecoach_ml.benchmark import benchmark_golden, benchmark_video
from slopecoach_ml.quality import VideoQualityGate, VideoQualityStatus
from slopecoach_ml.reference import analyze_pose_frame, load_golden_fixture
from slopecoach_ml.video import inspect_video


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "golden":
        frame, expected = load_golden_fixture(args.fixture)
        result = analyze_pose_frame(frame)
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
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
