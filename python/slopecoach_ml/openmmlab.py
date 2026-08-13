from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Any

PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "mmpose": "mmpose",
    "mmdet": "mmdet",
    "mmcv": "mmcv",
    "mmengine": "mmengine",
}


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def configured_device() -> str:
    device = os.getenv("SLOPECOACH_DEVICE", "cpu").strip().lower()
    if device not in {"cpu", "mps"}:
        raise ValueError("INVALID_DEVICE: SLOPECOACH_DEVICE must be cpu or mps")
    return device


def _import_check(statement: str, failure_code: str) -> tuple[bool, str | None]:
    try:
        exec(statement, {})
    except Exception as error:
        return False, f"{failure_code}: {type(error).__name__}: {error}"
    return True, None


def openmmlab_preflight() -> dict[str, Any]:
    versions = {name: _version(distribution) for name, distribution in PACKAGES.items()}
    device = configured_device()
    mps_built = False
    mps_available = False
    if versions["torch"]:
        import torch

        mps_built = torch.backends.mps.is_built()
        mps_available = torch.backends.mps.is_available()
    detector_config = os.getenv("SLOPECOACH_DETECTOR_CONFIG")
    detector_checkpoint = os.getenv("SLOPECOACH_DETECTOR_CHECKPOINT")
    pose_config = os.getenv("SLOPECOACH_POSE_CONFIG")
    pose_checkpoint = os.getenv("SLOPECOACH_POSE_CHECKPOINT")
    missing = [name for name, version in versions.items() if version is None]
    errors = [f"OPENMMLAB_DEPENDENCY_MISSING: {name}" for name in missing]
    extension_present = False
    extension_importable = False
    nms_importable = False
    mmdet_api_importable = False
    mmpose_api_importable = False
    if versions["mmcv"]:
        try:
            spec = importlib.util.find_spec("mmcv._ext")
            extension_present = bool(spec and spec.origin and Path(spec.origin).is_file())
        except (ImportError, ModuleNotFoundError) as error:
            errors.append(f"MMCV_COMPILED_OPS_MISSING: {type(error).__name__}: {error}")
        if not extension_present:
            errors.append("MMCV_COMPILED_OPS_MISSING")
        extension_importable, error = _import_check(
            "import mmcv._ext", "MMCV_EXTENSION_IMPORT_FAILED"
        )
        if error:
            errors.append(error)
        nms_importable, error = _import_check("from mmcv.ops import nms", "MMCV_OP_IMPORT_FAILED")
        if error:
            errors.append(error)
    if versions["mmdet"]:
        mmdet_api_importable, error = _import_check(
            "from mmdet.apis import init_detector, inference_detector", "MMDET_API_IMPORT_FAILED"
        )
        if error:
            errors.append(error)
    if versions["mmpose"]:
        mmpose_api_importable, error = _import_check(
            "from mmpose.apis import init_model, inference_topdown", "MMPOSE_API_IMPORT_FAILED"
        )
        if error:
            errors.append(error)
    configs = {
        "detector": bool(detector_config and Path(detector_config).is_file()),
        "pose": bool(pose_config and Path(pose_config).is_file()),
    }
    checkpoints = {
        "detector": bool(detector_checkpoint and Path(detector_checkpoint).is_file()),
        "pose": bool(pose_checkpoint and Path(pose_checkpoint).is_file()),
    }
    if not all(configs.values()):
        errors.append("MODEL_CONFIG_MISSING")
    if not all(checkpoints.values()):
        errors.append("MODEL_CHECKPOINT_MISSING")
    runtime_ready = (
        not missing
        and extension_present
        and extension_importable
        and nms_importable
        and mmdet_api_importable
        and mmpose_api_importable
    )
    if device == "mps" and not mps_available:
        errors.append("MPS_UNAVAILABLE")
    device_ready = device == "cpu" or mps_available
    ready = runtime_ready and device_ready and all(configs.values()) and all(checkpoints.values())
    return {
        "OPENMMLAB_PREFLIGHT": {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python": sys.version.split()[0],
            **versions,
            "mmcv_extension_present": extension_present,
            "mmcv_extension_importable": extension_importable,
            "mmcv_ops_nms_importable": nms_importable,
            "mmdet_api_importable": mmdet_api_importable,
            "mmpose_api_importable": mmpose_api_importable,
            "hardware": {"mps_built": mps_built, "mps_available": mps_available},
            "configured_device": device,
            "validated_device": "cpu" if device == "cpu" and runtime_ready else None,
            "installation_strategy": "isolated Python 3.11 provider environment",
            "configured_detector": "rtmdet-m-640-coco-obj365-person",
            "configured_pose_model": "rtmw-l-cocktail14-256x192",
            "config_presence": configs,
            "checkpoint_presence": checkpoints,
            "model_load_status": "NOT_TESTED",
            "status": "READY" if ready else "BLOCKED",
            "errors": list(dict.fromkeys(errors)),
        }
    }
