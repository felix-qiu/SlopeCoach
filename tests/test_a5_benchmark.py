from __future__ import annotations

import json

from slopecoach_ml.benchmark import benchmark_biomechanics_frames
from slopecoach_ml.detection import Detection
from slopecoach_ml.pose import BoundingBox2D, FrameGeometry, MMPoseRTMWPoseProvider
from slopecoach_ml.video import SampledFrame


class Detector:
    name = "fake-detector"

    def detect(self, image, geometry):
        return (Detection(0, BoundingBox2D(200 + image, 50, 100, 300), 0.95),)


class PoseBackend:
    def infer(self, image, boxes):
        output = []
        for left, top, right, bottom in boxes:
            points = [[left + 50, top + 20] for _ in range(133)]
            for index, point in {
                5: (left + 30, top + 30),
                6: (left + 70, top + 30),
                11: (left + 35, top + 110),
                12: (left + 65, top + 110),
                13: (left + 40, top + 190),
                14: (left + 60, top + 190),
                15: (left + 42, top + 270),
                16: (left + 58, top + 270),
            }.items():
                points[index] = point
            output.append((points, [0.9] * 133))
        return output


class Appearance:
    def encode(self, image, bbox):
        return (1.0, 0.0)


def test_a5_benchmark_contract_status_and_gt_nulls(monkeypatch):
    monkeypatch.setattr(
        "slopecoach_ml.benchmark.target_identity.inspect_video",
        lambda path: type(
            "Metadata", (), {"to_dict": lambda self: {"path": str(path)}}
        )(),
    )
    frames = [
        SampledFrame(index, index * 200_000, FrameGeometry(640, 480), index)
        for index in range(10)
    ]
    report = benchmark_biomechanics_frames(
        input_path="missing.mp4",
        frames=frames,
        detector=Detector(),
        pose_provider=MMPoseRTMWPoseProvider(PoseBackend()),
        detector_model={"model_id": "det"},
        pose_model={"model_id": "pose"},
        appearance_encoder=Appearance(),
    )
    assert report["benchmark_contract_version"] == "ski-bench-biomechanics-v1"
    assert report["config"]["FIXED_ML_FEATURE_VECTOR_STATUS"] == "NOT_FROZEN"
    assert len(report["frame_biomechanics"]["feature_coverage"]) == 14
    assert len(report["feature_registry"]) == 30
    assert report["ground_truth"]["feature_accuracy"] is None
    assert report["ground_truth"]["biomechanics_mae"] is None
    assert report["turn_biomechanics"] == []
    assert report["validation"]["A5_PRODUCT_VALIDATION"] == "BLOCKED_BY_GT"
    json.dumps(report, allow_nan=False)
