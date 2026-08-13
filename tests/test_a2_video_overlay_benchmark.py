from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from slopecoach_ml.benchmark import (
    benchmark_real_pose_frames,
    select_debug_frame_indices,
)
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

    def __init__(self, counts=None):
        self.counts = iter(counts) if counts is not None else None

    def detect(self, image, geometry):
        count = next(self.counts) if self.counts is not None else 1
        return tuple(
            Detection(index, BoundingBox2D(index * 5, 0, 50, 50), 0.9 - index * 0.1)
            for index in range(count)
        )


class PoseBackend:
    def __init__(self, *, shift=0.0, low_joint=None):
        self.shift = shift
        self.low_joint = low_joint

    def infer(self, image, boxes):
        results = []
        for _ in boxes:
            coordinates = [[10.0 + self.shift, 10.0] for _ in range(133)]
            coordinates[11] = [10.0 + self.shift, 10.0]
            coordinates[12] = [20.0 + self.shift, 10.0]
            coordinates[13] = [10.0 + self.shift, 20.0]
            coordinates[14] = [20.0 + self.shift, 20.0]
            coordinates[15] = [20.0 + self.shift, 20.0]
            coordinates[16] = [30.0 + self.shift, 20.0]
            scores = [0.9] * 133
            if self.low_joint is not None:
                scores[self.low_joint] = 0.1
            results.append((coordinates, scores))
        return results


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


def report(count=20, *, detector=None, pose=None):
    return benchmark_real_pose_frames(
        input_path="missing.mp4",
        input_kind="REAL_VIDEO",
        frames=frames(count),
        detector=detector or Detector(),
        pose_provider=pose or MMPoseRTMWPoseProvider(PoseBackend()),
        detector_model={"provider": "fake", "model_id": "det", "model_version": "1"},
        pose_model={"provider": "fake", "model_id": "pose", "model_version": "1"},
        device="cpu",
        sample_fps=2.0,
        clock=Clock(),
    )


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


def test_benchmark_full_metrics_and_p95_with_twenty_samples() -> None:
    result = report()
    assert result["input_kind"] == "REAL_VIDEO"
    assert result["runtime"]["device"] == "cpu"
    assert result["sampling"]["sampled_frame_count"] == 20
    assert result["performance"]["p95_pipeline_latency_seconds"] == pytest.approx(0.7)
    assert result["performance"]["p95_detector_latency_seconds"] == pytest.approx(0.1)
    assert result["detector"]["single_person_frame_ratio"] == 1.0
    assert result["pose"]["pose_success_ratio"] == 1.0
    assert result["pose"]["raw_joint_counts_observed"] == [133]
    assert result["pose"]["required_left_joint_mean_confidence"] == pytest.approx(0.9)
    assert result["pose"]["required_right_joint_mean_confidence"] == pytest.approx(0.9)
    assert result["pose"]["required_left_joint_coverage"] == 1.0
    assert result["pose"]["required_right_joint_coverage"] == 1.0
    assert result["pose"]["out_of_frame_joint_ratio"] == 0.0
    assert result["analysis"]["left_knee_angle_coverage"] == 1.0
    assert result["temporal_observation"]["consecutive_single_person_pairs"] == 19
    assert result["REAL_GT_STATUS"] == "NOT_AVAILABLE"
    assert all(value is None for value in result["ground_truth_metrics"].values())


def test_p95_is_null_below_twenty_samples() -> None:
    performance = report(19)["performance"]
    assert performance["p95_detector_latency_seconds"] is None
    assert performance["p95_pose_latency_seconds"] is None
    assert performance["p95_pipeline_latency_seconds"] is None


def test_zero_single_and_multi_person_status_and_temporal_boundaries() -> None:
    result = report(5, detector=Detector([0, 1, 1, 2, 1]))
    assert result["analysis"]["frame_status_counts"] == {
        "MULTIPLE_PERSONS_TARGET_UNRESOLVED": 1,
        "NO_PERSON_DETECTED": 1,
        "POSE_FRAME_OK": 3,
    }
    assert result["failure_reasons"]["NO_PERSON"] == 1
    assert result["failure_reasons"]["MULTIPLE_PERSONS"] == 1
    assert result["temporal_observation"]["consecutive_single_person_pairs"] == 1
    assert result["analysis"]["multiple_person_unresolved_ratio"] == pytest.approx(0.2)


def test_low_confidence_reason_and_required_joint_coverage() -> None:
    result = report(1, pose=MMPoseRTMWPoseProvider(PoseBackend(low_joint=13)))
    assert result["analysis"]["frame_status_counts"] == {
        "REQUIRED_JOINTS_LOW_CONFIDENCE": 1
    }
    assert result["failure_reasons"]["LOW_CONFIDENCE_LEFT_KNEE"] == 1
    assert result["pose"]["required_left_joint_coverage"] == pytest.approx(2 / 3)
    assert result["analysis"]["left_knee_angle_coverage"] == 0.0


def test_out_of_frame_status_ratio_and_failure_reasons() -> None:
    result = report(1, pose=MMPoseRTMWPoseProvider(PoseBackend(shift=100)))
    assert result["analysis"]["frame_status_counts"] == {
        "REQUIRED_JOINTS_OUT_OF_FRAME": 1
    }
    assert result["pose"]["out_of_frame_joint_ratio"] > 0
    assert result["pose"]["required_left_joint_coverage"] == 0.0
    assert result["failure_reasons"]["LEFT_HIP_OUT_OF_FRAME"] == 1
    assert result["failure_reasons"]["LEFT_KNEE_OUT_OF_FRAME"] == 1
    assert result["failure_reasons"]["LEFT_ANKLE_OUT_OF_FRAME"] == 1


def test_pose_provider_failure_is_distinct_and_serializable() -> None:
    class FailingPose(PoseBackend):
        def infer(self, image, boxes):
            raise RuntimeError("backend failed")

    result = report(1, pose=MMPoseRTMWPoseProvider(FailingPose()))
    assert result["analysis"]["frame_status_counts"] == {"POSE_PROVIDER_FAILED": 1}
    assert result["failure_reasons"]["POSE_INFERENCE_ERROR"] == 1
    assert result["failure_reasons"]["NO_PERSON"] == 0
    assert result["pose"]["pose_success_ratio"] == 0.0
    assert json.loads(json.dumps(result, allow_nan=False))["input_kind"] == "REAL_VIDEO"


def test_temporal_joint_displacement_is_bbox_normalized() -> None:
    class MovingPose(PoseBackend):
        def infer(self, image, boxes):
            self.shift += 5
            return super().infer(image, boxes)

    result = report(2, pose=MMPoseRTMWPoseProvider(MovingPose()))
    assert result["temporal_observation"]["consecutive_single_person_pairs"] == 1
    assert result["temporal_observation"][
        "median_normalized_joint_displacement"
    ] == pytest.approx(5 / (50 * 2**0.5))
    assert result["temporal_observation"][
        "median_left_knee_angle_delta_degrees"
    ] == pytest.approx(0)


def test_debug_selection_is_bounded_and_representative() -> None:
    observations = [
        {
            "frame_index": index,
            "status": "POSE_FRAME_OK",
            "mean_joint_confidence": 0.9 - index * 0.1,
            "normalized_joint_displacement": float(index) if index else None,
        }
        for index in range(5)
    ]
    selected = select_debug_frame_indices(observations, max_frames=4)
    assert selected == [0, 2, 4]


def test_timestamp_is_preserved_from_sample() -> None:
    sample = frames(2)[1]
    assert sample.timestamp_us == 500_000


def test_orientation_metadata_policy(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = '{"streams":[{"side_data_list":[{"rotation":-90}]}]}'

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Result())
    assert probe_rotation("video.mov") == 270
