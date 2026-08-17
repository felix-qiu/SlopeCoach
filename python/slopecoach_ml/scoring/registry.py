"""Explicit A7 diagnosis to A8 product-dimension registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .contracts import DIAGNOSIS_DIMENSION_REGISTRY_VERSION, ScoreDimension


@dataclass(frozen=True)
class DiagnosisDimensionMapping:
    diagnosis_code: str
    dimension: ScoreDimension
    research_only: bool
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dimension"] = self.dimension.value
        return payload


DIAGNOSIS_DIMENSION_REGISTRY = (
    DiagnosisDimensionMapping(
        "LIMITED_KNEE_FLEXION_MODULATION_2D",
        ScoreDimension.STANCE,
        True,
        ("IMAGE_SPACE_2D_ONLY", "NOT_PHYSICAL_STIFFNESS"),
    ),
    DiagnosisDimensionMapping(
        "BILATERAL_KNEE_ASYMMETRY_2D",
        ScoreDimension.SYMMETRY,
        True,
        ("IMAGE_SPACE_2D_ONLY", "NOT_PRESSURE_OR_LOAD_ASYMMETRY"),
    ),
    DiagnosisDimensionMapping(
        "KNEE_FLEXION_TIMING_OFFSET_2D",
        ScoreDimension.TIMING,
        True,
        ("IMAGE_SPACE_2D_ONLY", "NOT_EDGE_OR_PRESSURE_TIMING"),
    ),
)


def canonical_dimension_registry_json(registry=DIAGNOSIS_DIMENSION_REGISTRY) -> str:
    payload = {
        "registry_version": DIAGNOSIS_DIMENSION_REGISTRY_VERSION,
        "mappings": [item.to_dict() for item in registry],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


DIAGNOSIS_DIMENSION_REGISTRY_SHA256 = hashlib.sha256(
    canonical_dimension_registry_json().encode()
).hexdigest()
