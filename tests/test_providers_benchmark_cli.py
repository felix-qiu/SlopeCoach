from __future__ import annotations

import json
from pathlib import Path

from slopecoach_ml.benchmark import benchmark_golden, benchmark_video
from slopecoach_ml.cli import main
from slopecoach_ml.pose.providers import MockPoseProvider
from slopecoach_ml.reference import load_golden_fixture


FIXTURE = Path(__file__).parents[1] / "fixtures/golden_pose_001.json"


def test_mock_pose_provider() -> None:
    frame, _ = load_golden_fixture(FIXTURE)
    assert MockPoseProvider(frame).estimate(object(), frame.geometry) is frame


def test_golden_benchmark_runs_without_fake_ground_truth() -> None:
    report, result = benchmark_golden(FIXTURE)
    assert report["golden_passed"] is True
    assert report["REAL_GT_STATUS"] == "NOT_AVAILABLE"
    assert all(value is None for value in report["ground_truth_metrics"].values())
    assert result["features"]["left_knee_angle_2d_degrees"] == 90.0


def test_real_video_benchmark_reports_provider_not_configured(tmp_path) -> None:
    report = benchmark_video(tmp_path / "missing.mp4")
    assert report["pose_provider"] == "NOT_CONFIGURED"
    assert report["pose"]["coverage"] is None
    assert report["biomechanics"]["knee_angle_2d_coverage"] is None


def test_cli_golden_writes_json(tmp_path) -> None:
    output = tmp_path / "golden.json"
    assert main(["golden", "--fixture", str(FIXTURE), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["golden_passed"] is True


def test_cli_missing_video_returns_nonzero(tmp_path) -> None:
    assert main(["inspect-video", str(tmp_path / "missing.mp4")]) != 0
