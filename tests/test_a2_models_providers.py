from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

from slopecoach_ml.detection import MMDetPersonDetectorProvider
from slopecoach_ml.models import ModelMetadata, load_model_registry
from slopecoach_ml.openmmlab import configured_device, openmmlab_preflight
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
        "rtmdet-tiny-640-coco-equipment",
        "rtmw-l-cocktail14-256x192",
        "openai-clip-vit-b32-visual-sport",
    }
    pose = registry["rtmw-l-cocktail14-256x192"]
    assert pose.framework_version == "1.3.2"
    assert pose.input_size == (256, 192)
    assert ModelMetadata.from_dict(pose.to_dict()) == pose
    assert pose.checkpoint_license is None
    assert len(pose.checkpoint_sha256 or "") == 64
    visual = registry["openai-clip-vit-b32-visual-sport"]
    assert visual.model_family == "CLIP"
    assert visual.checkpoint_license is None
    assert visual.model_version == "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"


def test_registry_rejects_malformed_checkpoint_hash(tmp_path) -> None:
    payload = REGISTRY.read_text(encoding="utf-8").replace(
        "235e820939cb2ff33c505441e71f7e9532958c281636a963c6829d100867aed9", "BAD"
    )
    registry = tmp_path / "registry.json"
    registry.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        load_model_registry(registry)


def test_device_defaults_to_cpu_and_rejects_invalid(monkeypatch) -> None:
    monkeypatch.delenv("SLOPECOACH_DEVICE", raising=False)
    assert configured_device() == "cpu"
    monkeypatch.setenv("SLOPECOACH_DEVICE", "mps")
    assert configured_device() == "mps"
    monkeypatch.setenv("SLOPECOACH_DEVICE", "cuda")
    with pytest.raises(ValueError, match="INVALID_DEVICE"):
        configured_device()


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
    assert report["configured_device"] == "cpu"
    assert "mmcv_extension_present" in report
    assert "mmcv_ops_nms_importable" in report


def test_pose_doctor_classifies_installed_mmcv_without_extension(monkeypatch) -> None:
    import slopecoach_ml.openmmlab as doctor

    fake_torch = ModuleType("torch")
    fake_torch.backends = type(
        "Backends",
        (),
        {
            "mps": type(
                "MPS",
                (),
                {
                    "is_built": staticmethod(lambda: True),
                    "is_available": staticmethod(lambda: True),
                },
            )()
        },
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(doctor, "_version", lambda distribution: "test-version")
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: None)

    def imports(statement, failure_code):
        if statement == "import mmcv._ext":
            return False, "MMCV_EXTENSION_IMPORT_FAILED: ModuleNotFoundError"
        if statement == "from mmcv.ops import nms":
            return False, "MMCV_OP_IMPORT_FAILED: ModuleNotFoundError"
        return True, None

    monkeypatch.setattr(doctor, "_import_check", imports)
    report = doctor.openmmlab_preflight()["OPENMMLAB_PREFLIGHT"]
    assert report["mmcv_extension_present"] is False
    assert report["status"] == "BLOCKED"
    assert "MMCV_COMPILED_OPS_MISSING" in report["errors"]
    assert any(
        error.startswith("MMCV_EXTENSION_IMPORT_FAILED") for error in report["errors"]
    )
    assert any(error.startswith("MMCV_OP_IMPORT_FAILED") for error in report["errors"])


def test_real_backends_receive_explicit_device(monkeypatch) -> None:
    detector_calls = []
    pose_calls = []

    mmdet_apis = ModuleType("mmdet.apis")
    mmdet_apis.init_detector = (
        lambda config, checkpoint, device: detector_calls.append(device) or object()
    )
    mmdet_apis.inference_detector = lambda model, image: None
    mmpose_apis = ModuleType("mmpose.apis")
    mmpose_apis.init_model = (
        lambda config, checkpoint, device: pose_calls.append(device) or object()
    )
    mmpose_apis.inference_topdown = lambda model, image, **kwargs: []
    monkeypatch.setitem(sys.modules, "mmdet", ModuleType("mmdet"))
    monkeypatch.setitem(sys.modules, "mmdet.apis", mmdet_apis)
    monkeypatch.setitem(sys.modules, "mmpose", ModuleType("mmpose"))
    monkeypatch.setitem(sys.modules, "mmpose.apis", mmpose_apis)
    from slopecoach_ml.detection.mmdet_provider import OpenMMLabMMDetBackend

    OpenMMLabMMDetBackend("det.py", "det.pth", device="mps")
    OpenMMLabMMPoseBackend("pose.py", "pose.pth", device="mps")
    assert detector_calls == ["mps"]
    assert pose_calls == ["mps"]


def test_detector_restores_mmdet_scope_before_inference(monkeypatch) -> None:
    scopes = []
    registry = ModuleType("mmengine.registry")
    registry.init_default_scope = scopes.append
    monkeypatch.setitem(sys.modules, "mmengine", ModuleType("mmengine"))
    monkeypatch.setitem(sys.modules, "mmengine.registry", registry)

    class Instances:
        bboxes = scores = labels = type(
            "Array", (), {"numpy": lambda self: self, "tolist": lambda self: []}
        )()

        def cpu(self):
            return self

    class Result:
        pred_instances = Instances()

    backend = object.__new__(
        __import__(
            "slopecoach_ml.detection.mmdet_provider", fromlist=["OpenMMLabMMDetBackend"]
        ).OpenMMLabMMDetBackend
    )
    backend._model = object()
    backend._infer = lambda model, image: Result()
    assert backend.infer(object()) == []
    assert scopes == ["mmdet"]


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
