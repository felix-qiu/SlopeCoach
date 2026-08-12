from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelMetadata:
    provider: str
    framework: str
    framework_version: str
    model_family: str
    model_id: str
    model_version: str
    task: str
    input_size: tuple[int, int]
    joint_schema: str | None
    config_source: str
    checkpoint_source: str
    checkpoint_filename: str
    checkpoint_sha256: str | None
    code_license: str | None
    checkpoint_license: str | None
    training_metadata: str | None
    notes: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        required = {field.name for field in cls.__dataclass_fields__.values()}
        if set(data) != required:
            raise ValueError(f"model metadata fields differ: {sorted(set(data) ^ required)}")
        size = data["input_size"]
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in size
            )
        ):
            raise ValueError("input_size must contain two positive integers")
        values = dict(data)
        values["input_size"] = tuple(size)
        model = cls(**values)
        for name in (
            "provider",
            "framework",
            "framework_version",
            "model_family",
            "model_id",
            "model_version",
            "task",
            "config_source",
            "checkpoint_source",
            "checkpoint_filename",
        ):
            if not isinstance(getattr(model, name), str) or not getattr(model, name):
                raise ValueError(f"{name} must be a non-empty string")
        return model

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input_size"] = list(self.input_size)
        return data


def load_model_registry(path: str | Path) -> dict[str, ModelMetadata]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("registry_version") != "slopecoach-research-model-registry-v1":
        raise ValueError("unsupported model registry version")
    models = [ModelMetadata.from_dict(item) for item in payload["models"]]
    if len({model.model_id for model in models}) != len(models):
        raise ValueError("duplicate model_id in registry")
    return {model.model_id: model for model in models}
