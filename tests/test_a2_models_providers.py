from __future__ import annotations

from pathlib import Path

import pytest

from slopecoach_ml.detection import MMDetPersonDetectorProvider
from slopecoach_ml.models import ModelMetadata, load_model_registry
from slopecoach_ml.openmmlab import openmmlab_preflight
from slopecoach_ml.cli import main
from slopecoach_ml.pose import (
    FrameGeometry,
    Joint,
    MMPoseRTMWPoseProvider,
    RTMW_COCO_WHOLEBODY_INDEX_BY_JOINT,
)
from slopecoach_ml.pose.mmpose_provider import OpenMMLabMMPoseBackend


REGISTRY = Path(__file__).parents[1] / "models/registry.json"


class FakeDetectorBackend:
    def infer(self, image):
        return [(-5.0, 10.0, 105.0, 210.0, 0.9, 0), (0, 0, 10, 10, 0.99, 2)]


class FakePoseBackend:
    def infer(self, image, bboxes_xyxy):
        coordinates = [[float(i), float(i + 100)] for i in range(133)]
        scores = [0.5 + i / 1000 for i in range(133)]
        return [(coordinates, scores) for _ in bboxes_xyxy]


def test_model_registry_parsing_and_provenance_serialization() -> None:
    registry = load_model_registry(REGISTRY)
    assert set(registry) == {
        "rtmdet-m-640-coco-obj365-person",
        "rtmw-l-cocktail14-256x192",
    }
    pose = registry["rtmw-l-cocktail14-256x192"]
    assert pose.framework_version == "1.3.2"
    assert pose.input_size == (256, 192)
    assert ModelMetadata.from_dict(pose.to_dict()) == pose
    assert pose.checkpoint_license is None


def test_dependency_missing_is_clear(monkeypatch) -> None:
    import builtins

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.startswith("mmpose"):
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match="OPENMMLAB_DEPENDENCY_MISSING: mmpose"):
        OpenMMLabMMPoseBackend("config.py", "weights.pth")


def test_pose_doctor_is_blocked_without_optional_stack_or_checkpoints() -> None:
    report = openmmlab_preflight()["OPENMMLAB_PREFLIGHT"]
    assert report["status"] == "BLOCKED"
    assert report["configured_pose_model"] == "rtmw-l-cocktail14-256x192"


def test_pose_image_requires_explicit_non_mirrored_attestation(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="MIRROR_STATE_UNRESOLVED"):
        main(["pose-image", str(tmp_path / "image.jpg")])


def test_rtmw_mapping_preserves_explicit_left_right_identity() -> None:
    expected = {
        Joint.LEFT_SHOULDER: 5,
        Joint.RIGHT_SHOULDER: 6,
        Joint.LEFT_HIP: 11,
        Joint.RIGHT_HIP: 12,
        Joint.LEFT_KNEE: 13,
        Joint.RIGHT_KNEE: 14,
        Joint.LEFT_ANKLE: 15,
        Joint.RIGHT_ANKLE: 16,
    }
    for joint, index in expected.items():
        assert RTMW_COCO_WHOLEBODY_INDEX_BY_JOINT[joint] == index


def test_detector_and_pose_conversion_preserve_source_coordinates_without_double_inverse() -> (
    None
):
    geometry = FrameGeometry(640, 480)
    detector = MMDetPersonDetectorProvider(FakeDetectorBackend())
    detections = detector.detect(object(), geometry)
    assert len(detections) == 1
    assert detections[0].bbox.x_px == -5.0
    assert detections[0].bbox.width_px == 110.0
    frame = MMPoseRTMWPoseProvider(FakePoseBackend()).estimate_detections(
        object(), detections, geometry, timestamp_us=1234, frame_index=7
    )
    assert frame.timestamp_us == 1234
    assert frame.geometry.coordinate_space.value == "SourcePixel2D"
    assert frame.persons[0].joint(Joint.LEFT_HIP).x_px == 11.0
    assert frame.persons[0].joint(Joint.RIGHT_HIP).x_px == 12.0
    assert frame.persons[0].bbox.x_px == -5.0


def test_zero_detection_does_not_call_pose_backend() -> None:
    class FailingBackend:
        def infer(self, image, boxes):
            raise AssertionError("must not run")

    frame = MMPoseRTMWPoseProvider(FailingBackend()).estimate_detections(
        object(), (), FrameGeometry(10, 10), timestamp_us=0, frame_index=0
    )
    assert frame.persons == ()
