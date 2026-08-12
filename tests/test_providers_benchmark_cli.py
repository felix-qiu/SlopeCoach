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
    assert report["input_kind"] == "REAL_VIDEO"
    assert report["pose_provider"] == "NOT_CONFIGURED"
    assert report["pose"]["coverage"] is None
    assert report["biomechanics"]["knee_angle_2d_coverage"] is None


def test_synthetic_metadata_smoke_is_not_real_video(tmp_path) -> None:
    report = benchmark_video(
        tmp_path / "missing.mp4", input_kind="SYNTHETIC_METADATA_SMOKE"
    )
    assert report["input_kind"] == "SYNTHETIC_METADATA_SMOKE"


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def test_golden_benchmark_uses_one_deterministic_clock_domain() -> None:
    report, _ = benchmark_golden(
        FIXTURE, clock=FakeClock([10.0, 10.0, 11.0, 11.0, 13.5, 14.0])
    )
    performance = report["performance"]
    assert performance["total_processing_seconds"] == 4.0
    assert performance["per_stage_seconds"] == {"parse": 1.0, "analysis": 2.5}
    assert performance["mean_latency_seconds"] == 2.5


def test_video_benchmark_uses_one_deterministic_clock_domain(tmp_path) -> None:
    clock = FakeClock([20.0, 20.0, 21.5, 21.5, 22.0, 23.0])
    report = benchmark_video(tmp_path / "missing.mp4", clock=clock)
    performance = report["performance"]
    assert performance["total_processing_seconds"] == 3.0
    assert performance["per_stage_seconds"] == {
        "video_inspection": 1.5,
        "quality_gate": 0.5,
    }
    assert performance["mean_latency_seconds"] == 1.5


def test_cli_golden_writes_json(tmp_path) -> None:
    output = tmp_path / "golden.json"
    assert main(["golden", "--fixture", str(FIXTURE), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["golden_passed"] is True


def test_cli_missing_video_returns_nonzero(tmp_path) -> None:
    assert main(["inspect-video", str(tmp_path / "missing.mp4")]) != 0
