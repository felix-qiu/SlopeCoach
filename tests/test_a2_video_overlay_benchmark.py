from __future__ import annotations

from dataclasses import dataclass

import pytest

from slopecoach_ml.benchmark import benchmark_real_pose_frames
from slopecoach_ml.detection import Detection
from slopecoach_ml.pose import (
    BoundingBox2D,
    FrameGeometry,
    MMPoseRTMWPoseProvider,
    overlay_primitives,
)
from slopecoach_ml.video import SampledFrame
from slopecoach_ml.video.frames import probe_rotation


class Detector:
    name = "fake-detector"

    def detect(self, image, geometry):
        return (Detection(0, BoundingBox2D(0, 0, 50, 50), 0.9),)


class PoseBackend:
    def infer(self, image, boxes):
        coordinates = [[10.0, 10.0] for _ in range(133)]
        coordinates[11] = [10.0, 10.0]
        coordinates[13] = [10.0, 20.0]
        coordinates[15] = [20.0, 20.0]
        return [(coordinates, [0.9] * 133)]


@dataclass
class Clock:
    current: float = 0.0

    def __call__(self):
        self.current += 0.1
        return self.current


def frames(count: int):
    geometry = FrameGeometry(100, 100)
    return [
        SampledFrame(index, index * 500_000, geometry, object())
        for index in range(count)
    ]


def test_overlay_primitives_use_canonical_pose_coordinates() -> None:
    provider = MMPoseRTMWPoseProvider(PoseBackend())
    pose = provider.estimate_detections(
        object(),
        Detector().detect(object(), FrameGeometry(100, 100)),
        FrameGeometry(100, 100),
        timestamp_us=7,
        frame_index=0,
    )
    primitives = overlay_primitives(pose)
    assert (10.0, 10.0, "left_hip", 0.9) in primitives["points"]
    assert (10.0, 10.0, 10.0, 20.0) in primitives["lines"]


@pytest.mark.parametrize("kind", ["REAL_VIDEO", "SYNTHETIC_PIPELINE_SMOKE"])
def test_benchmark_classification_aggregation_and_p95(kind, tmp_path) -> None:
    report = benchmark_real_pose_frames(
        input_path=tmp_path / "missing.mp4",
        input_kind=kind,
        frames=frames(20),
        detector=Detector(),
        pose_provider=MMPoseRTMWPoseProvider(PoseBackend()),
        detector_model={"provider": "fake", "model_id": "det", "model_version": "1"},
        pose_model={"provider": "fake", "model_id": "pose", "model_version": "1"},
        device="cpu",
        clock=Clock(),
    )
    assert report["input_kind"] == kind
    assert report["device"] == "cpu"
    assert report["performance"]["sampled_frame_count"] == 20
    assert report["performance"]["p95_frame_latency_seconds"] == pytest.approx(0.7)
    assert report["detector"]["single_person_frame_ratio"] == 1.0
    assert report["pose"]["coverage"] == 1.0
    assert report["biomechanics"]["left_knee_angle_2d_coverage"] == 1.0
    assert all(value is None for value in report["ground_truth_metrics"].values())


def test_timestamp_is_preserved_from_sample() -> None:
    sample = frames(2)[1]
    assert sample.timestamp_us == 500_000


def test_orientation_metadata_policy(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = '{"streams":[{"side_data_list":[{"rotation":-90}]}]}'

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Result())
    assert probe_rotation("video.mov") == 270
