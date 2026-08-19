from __future__ import annotations

import json
from dataclasses import dataclass

from slopecoach_ml.benchmark import benchmark_target_identity_frames
from slopecoach_ml.detection import Detection
from slopecoach_ml.identity import InitialTargetSelectorConfig, ManualTargetSeed
from slopecoach_ml.pose import BoundingBox2D, FrameGeometry, MMPoseRTMWPoseProvider
from slopecoach_ml.video import SampledFrame


class Detector:
    name = "fake-detector"

    def detect(self, image, geometry):
        x = 250 + image * 10
        return (
            Detection(0, BoundingBox2D(x, 100, 120, 240), 0.95),
            Detection(1, BoundingBox2D(10, 10, 4, 8), 0.2),
        )


class CrossingDetector:
    name = "synthetic-crossing-detector"
    sequence = (
        ((100, 100), (200, 260)),
        ((101, 140), (201, 240)),
        ((202, 220),),
        ((203, 200),),
        ((104, 260), (204, 180)),
    )

    def detect(self, image, geometry):
        return tuple(
            Detection(detection_id, BoundingBox2D(x, 100, 100, 200), 0.9)
            for detection_id, x in self.sequence[image]
        )


class PoseBackend:
    def infer(self, image, boxes):
        result = []
        for left, top, right, bottom in boxes:
            coordinates = [[left, top] for _ in range(133)]
            coordinates[11] = [left + 40, top + 80]
            coordinates[13] = [left + 40, top + 140]
            coordinates[15] = [left + 90, top + 140]
            coordinates[12] = [left + 70, top + 80]
            coordinates[14] = [left + 70, top + 140]
            coordinates[16] = [left + 100, top + 190]
            result.append((coordinates, [0.9] * 133))
        return result


class Appearance:
    def encode(self, image, bbox):
        return (1.0, 0.0)


@dataclass
class Clock:
    value: float = 0.0

    def __call__(self):
        self.value += 0.001
        return self.value


def test_target_benchmark_contract_pose_reduction_and_null_gt(monkeypatch) -> None:
    monkeypatch.setattr(
        "slopecoach_ml.benchmark.target_identity.inspect_video",
        lambda path: type(
            "Metadata", (), {"to_dict": lambda self: {"path": str(path)}}
        )(),
    )
    geometry = FrameGeometry(640, 480)
    frames = [SampledFrame(i, i * 500_000, geometry, i) for i in range(6)]
    report = benchmark_target_identity_frames(
        input_path="missing.mp4",
        frames=frames,
        detector=Detector(),
        pose_provider=MMPoseRTMWPoseProvider(PoseBackend()),
        detector_model={"model_id": "det"},
        pose_model={"model_id": "pose"},
        selector_config=InitialTargetSelectorConfig(
            initialization_window_us=1_000_000,
            minimum_track_observations=3,
            minimum_lock_score=0.4,
            minimum_winner_margin=0.05,
        ),
        appearance_encoder=Appearance(),
        clock=Clock(),
    )
    assert report["benchmark_contract_version"] == "ski-bench-target-identity-v2"
    assert report["input_kind"] == "REAL_VIDEO"
    assert report["detections"]["raw_detection_person_count"] == 12
    assert report["candidates"]["viable_candidate_count"] == 6
    assert report["pose_efficiency"]["pose_person_inference_count"] <= 6
    assert report["pose_efficiency"]["pose_inference_reduction_ratio"] >= 0.5
    assert report["TARGET_IDENTITY_GT_STATUS"] == "NOT_AVAILABLE"
    assert report["identity_accuracy"]["wrong_target_rate"] is None
    assert report["identity_accuracy"]["target_frame_accuracy"] is None
    assert report["tracking"]["terminated_track_count"] >= 0
    assert report["tracking"]["track_fragmentation_count"] is None
    assert "track_fragments" not in report["tracking"]
    assert report["DEEP_REID_STATUS"] == "NOT_CONFIGURED"
    scores = [
        item["latest_identity_match_score"] for item in report["frame_observations"]
    ]
    assert any(score is None for score in scores)
    assert any(score is not None for score in scores)
    assert all("last_observed_age_us" in item for item in report["frame_observations"])
    json.dumps(report, allow_nan=False)


def test_benchmark_wrong_person_never_enters_trusted_target_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "slopecoach_ml.benchmark.target_identity.inspect_video",
        lambda path: type(
            "Metadata", (), {"to_dict": lambda self: {"path": str(path)}}
        )(),
    )
    geometry = FrameGeometry(1000, 600)
    frames = [SampledFrame(i, i * 200_000, geometry, i) for i in range(5)]
    observed = []
    report = benchmark_target_identity_frames(
        input_path="missing.mp4",
        frames=frames,
        detector=CrossingDetector(),
        pose_provider=MMPoseRTMWPoseProvider(PoseBackend()),
        detector_model={"model_id": "det"},
        pose_model={"model_id": "pose"},
        sample_fps=5,
        manual_target_seed=ManualTargetSeed(0.0, 150, 150),
        appearance_encoder=Appearance(),
        frame_observer=lambda frame, observation, pose: observed.append(
            (observation, pose)
        ),
        clock=Clock(),
    )

    assert report["manual_target_seed"]["selected_detection_id"] == 100
    bystander_ids = {200, 201, 202, 203, 204}
    for observation, _pose_frame in observed:
        active_track_id = observation["active_track_id"]
        active_detection_id = next(
            (
                item["detection_id"]
                for item in observation["tracks"]
                if item["track_id"] == active_track_id
            ),
            None,
        )
        if observation["target_pose_available"]:
            assert active_detection_id not in bystander_ids

    occluded = [observed[index][0] for index in (2, 3)]
    assert all(item["identity_state"] == "SUSPECT" for item in occluded)
    assert all(not item["target_pose_available"] for item in occluded)
    assert all(item["left_knee_angle_2d_degrees"] is None for item in occluded)
    assert all("TARGET_IDENTITY_UNCERTAIN" in item["limitations"] for item in occluded)
    # B may be pose-probed for identity recovery, but is never emitted as A's target pose.
    assert [
        [person.detection_id for person in observed[index][1].persons]
        for index in (2, 3)
    ] == [[202], [203]]
    assert report["tracking"]["preferred_association_conflict_count"] >= 1
    assert report["tracking"]["preferred_association_override_count"] == 0
    assert report["tracking"]["preferred_association_rejected_non_duplicate_count"] >= 1
    assert report["identity_accuracy"]["wrong_target_rate"] is None
    json.dumps(report, allow_nan=False)


def test_a2_real_pose_baseline_api_remains_importable() -> None:
    from slopecoach_ml.benchmark import benchmark_real_pose_frames

    assert callable(benchmark_real_pose_frames)
