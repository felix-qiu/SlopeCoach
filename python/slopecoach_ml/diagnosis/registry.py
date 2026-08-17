"""Deterministic A7 research diagnosis rule registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .contracts import (
    DIAGNOSIS_RULE_REGISTRY_VERSION,
    DIAGNOSIS_RULE_SCHEMA_VERSION,
    DiagnosisCode,
    DiagnosisPhase,
    DiagnosisRuleConfig,
)


@dataclass(frozen=True)
class DiagnosisRuleDefinition:
    diagnosis_code: DiagnosisCode
    applicable_sport_types: tuple[str, ...]
    required_features: tuple[str, ...]
    phase: DiagnosisPhase
    operator: str
    threshold_config_field: str
    minimum_evidence_policy: str
    limitations: tuple[str, ...]

    def to_dict(self, config: DiagnosisRuleConfig) -> dict[str, object]:
        payload = asdict(self)
        payload["diagnosis_code"] = self.diagnosis_code.value
        payload["phase"] = self.phase.value
        payload["default_research_threshold"] = getattr(config, self.threshold_config_field)
        return payload


RULE_REGISTRY = (
    DiagnosisRuleDefinition(
        DiagnosisCode.LIMITED_KNEE_FLEXION_MODULATION_2D,
        ("SKI", "SNOWBOARD"),
        ("bilateral_knee_mean_angle_2d_deg",),
        DiagnosisPhase.FULL_TURN,
        "<",
        "limited_knee_flexion_range_deg",
        "MULTI_FRAME_MINIMUM_SAMPLES_AND_COVERAGE",
        ("IMAGE_SPACE_2D_ONLY", "NOT_PHYSICAL_STIFFNESS"),
    ),
    DiagnosisRuleDefinition(
        DiagnosisCode.BILATERAL_KNEE_ASYMMETRY_2D,
        ("SKI", "SNOWBOARD"),
        ("bilateral_knee_abs_difference_2d_deg",),
        DiagnosisPhase.FULL_TURN,
        ">=",
        "knee_asymmetry_median_deg",
        "MULTI_FRAME_MINIMUM_SAMPLES_AND_COVERAGE",
        ("IMAGE_SPACE_2D_ONLY", "NOT_PRESSURE_OR_LOAD_ASYMMETRY"),
    ),
    DiagnosisRuleDefinition(
        DiagnosisCode.KNEE_FLEXION_TIMING_OFFSET_2D,
        ("SKI", "SNOWBOARD"),
        ("minimum_mean_knee_angle_phase_offset", "bilateral_knee_mean_angle_2d_deg"),
        DiagnosisPhase.APEX_RELATIVE_TIMING,
        "abs(value) >",
        "knee_flexion_phase_offset_abs",
        "TURN_FEATURE_PLUS_MULTI_FRAME_MINIMUM_SAMPLES_AND_COVERAGE",
        ("IMAGE_SPACE_2D_ONLY", "NOT_EDGE_OR_PRESSURE_TIMING"),
    ),
)


def canonical_rule_registry_json(
    config: DiagnosisRuleConfig | None = None,
    registry=RULE_REGISTRY,
) -> str:
    settings = config or DiagnosisRuleConfig()
    payload = {
        "schema_version": DIAGNOSIS_RULE_SCHEMA_VERSION,
        "registry_version": DIAGNOSIS_RULE_REGISTRY_VERSION,
        "rules": [item.to_dict(settings) for item in registry],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


DIAGNOSIS_RULE_REGISTRY_SHA256 = hashlib.sha256(canonical_rule_registry_json().encode()).hexdigest()
