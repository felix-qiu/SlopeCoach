from __future__ import annotations

import importlib.metadata
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


def openmmlab_preflight() -> dict[str, Any]:
    versions = {name: _version(distribution) for name, distribution in PACKAGES.items()}
    device = "unavailable"
    if versions["torch"]:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    detector_checkpoint = os.getenv("SLOPECOACH_DETECTOR_CHECKPOINT")
    pose_checkpoint = os.getenv("SLOPECOACH_POSE_CHECKPOINT")
    missing = [name for name, version in versions.items() if version is None]
    api_errors: list[str] = []
    if not missing:
        try:
            from mmdet.apis import inference_detector, init_detector  # noqa: F401
        except (ImportError, ModuleNotFoundError) as error:
            api_errors.append(f"OPENMMLAB_API_IMPORT_FAILED: mmdet: {error}")
        try:
            from mmpose.apis import inference_topdown, init_model  # noqa: F401
        except (ImportError, ModuleNotFoundError) as error:
            api_errors.append(f"OPENMMLAB_API_IMPORT_FAILED: mmpose: {error}")
    checkpoints = {
        "detector": bool(detector_checkpoint and Path(detector_checkpoint).is_file()),
        "pose": bool(pose_checkpoint and Path(pose_checkpoint).is_file()),
    }
    ready = not missing and not api_errors and all(checkpoints.values())
    return {
        "OPENMMLAB_PREFLIGHT": {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python": sys.version.split()[0],
            **versions,
            "device_available": device,
            "installation_strategy": "isolated Python 3.11 provider environment",
            "configured_detector": "rtmdet-m-640-coco-obj365-person",
            "configured_pose_model": "rtmw-l-cocktail14-256x192",
            "checkpoint_presence": checkpoints,
            "status": "READY" if ready else "BLOCKED",
            "errors": [f"OPENMMLAB_DEPENDENCY_MISSING: {name}" for name in missing]
            + api_errors
            + ([] if all(checkpoints.values()) else ["MODEL_CHECKPOINT_MISSING"]),
        }
    }
