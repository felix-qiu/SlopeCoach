from __future__ import annotations

import json

from slopecoach_ml.benchmark import benchmark_temporal_turns_frames
from slopecoach_ml.detection import Detection
from slopecoach_ml.pose import BoundingBox2D, FrameGeometry, MMPoseRTMWPoseProvider
from slopecoach_ml.video import SampledFrame


class Detector:
    name = "fake-detector"

    def detect(self, image, geometry):
        return (Detection(0, BoundingBox2D(200 + image, 50, 100, 300), 0.95),)


class PoseBackend:
    def infer(self, image, boxes):
        results = []
        for left, top, right, bottom in boxes:
            coordinates = [[left + 50, top + 20] for _ in range(133)]
            coordinates[5] = [left + 30, top + 30]
            coordinates[6] = [left + 70, top + 30]
            coordinates[11] = [left + 35, top + 110]
            coordinates[12] = [left + 65, top + 110]
            coordinates[13] = [left + 40 + image, top + 190]
            coordinates[14] = [left + 60 + image, top + 190]
            coordinates[15] = [left + 42 + image, top + 270]
            coordinates[16] = [left + 58 + image, top + 270]
            results.append((coordinates, [0.9] * 133))
        return results


class Appearance:
    def encode(self, image, bbox):
        return (1.0, 0.0)


def test_temporal_benchmark_contract_and_null_turn_gt(monkeypatch) -> None:
    monkeypatch.setattr(
        "slopecoach_ml.benchmark.target_identity.inspect_video",
        lambda path: type(
            "Metadata", (), {"to_dict": lambda self: {"path": str(path)}}
        )(),
    )
    geometry = FrameGeometry(640, 480)
    frames = [
        SampledFrame(index, index * 200_000, geometry, index) for index in range(10)
    ]
    report = benchmark_temporal_turns_frames(
        input_path="missing.mp4",
        frames=frames,
        detector=Detector(),
        pose_provider=MMPoseRTMWPoseProvider(PoseBackend()),
        detector_model={"model_id": "det"},
        pose_model={"model_id": "pose"},
        sample_fps=5,
        appearance_encoder=Appearance(),
    )
    assert report["benchmark_contract_version"] == "ski-bench-temporal-turns-v1"
    assert (
        report["identity_input"]["target_identity_gt_annotation_status"] == "DEFERRED"
    )
    assert report["identity_input"]["target_identity_gt_status"] == "NOT_AVAILABLE"
    assert report["identity_input"]["target_identity_accuracy_status"] == "UNKNOWN"
    assert report["turn_segmentation"]["TURN_SEGMENTATION_GT_STATUS"] == "NOT_AVAILABLE"
    assert report["turn_segmentation"]["turn_precision"] is None
    assert report["turn_segmentation"]["turn_recall"] is None
    assert report["turn_segmentation"]["turn_f1"] is None
    assert report["validation"]["A4_PRODUCT_VALIDATION"] == "BLOCKED_BY_IDENTITY_GT"
    json.dumps(report, allow_nan=False)
