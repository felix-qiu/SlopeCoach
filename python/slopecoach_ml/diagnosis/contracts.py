"""Provisional A7 diagnosis contracts for Python research/reference only."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum

DIAGNOSIS_CONTRACT_VERSION = "diagnosis-v1"
DIAGNOSIS_RULE_SCHEMA_VERSION = "diagnosis-rule-schema-v1"
DIAGNOSIS_RULE_REGISTRY_VERSION = "diagnosis-rule-registry-v1"
DIAGNOSIS_BENCHMARK_CONTRACT_VERSION = "ski-bench-diagnosis-v1"
DIAGNOSIS_CONFIG_PROFILE = "RESEARCH_DEFAULTS_A7"


class DiagnosisCode(StrEnum):
    LIMITED_KNEE_FLEXION_MODULATION_2D = "LIMITED_KNEE_FLEXION_MODULATION_2D"
    BILATERAL_KNEE_ASYMMETRY_2D = "BILATERAL_KNEE_ASYMMETRY_2D"
    KNEE_FLEXION_TIMING_OFFSET_2D = "KNEE_FLEXION_TIMING_OFFSET_2D"


class DiagnosisEvaluationStatus(StrEnum):
    TRIGGERED = "TRIGGERED"
    NOT_TRIGGERED = "NOT_TRIGGERED"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class DiagnosisPhase(StrEnum):
    FULL_TURN = "FULL_TURN"
    APEX_RELATIVE_TIMING = "APEX_RELATIVE_TIMING"


class DiagnosisResultStatus(StrEnum):
    NOT_ANALYZABLE_SPORT_TYPE_UNKNOWN = "NOT_ANALYZABLE_SPORT_TYPE_UNKNOWN"
    NOT_ANALYZABLE_NO_QUALIFIED_TURNS = "NOT_ANALYZABLE_NO_QUALIFIED_TURNS"
    NOT_ANALYZABLE_INSUFFICIENT_DIAGNOSIS_EVIDENCE = (
        "NOT_ANALYZABLE_INSUFFICIENT_DIAGNOSIS_EVIDENCE"
    )
    EXECUTED_NO_PROVISIONAL_RULES_TRIGGERED = "EXECUTED_NO_PROVISIONAL_RULES_TRIGGERED"
    EXECUTED_WITH_PROVISIONAL_DIAGNOSES = "EXECUTED_WITH_PROVISIONAL_DIAGNOSES"


@dataclass(frozen=True)
class DiagnosisRuleConfig:
    minimum_turn_feature_samples: int = 5
    minimum_turn_feature_coverage: float = 0.60
    limited_knee_flexion_range_deg: float = 12.0
    knee_asymmetry_median_deg: float = 10.0
    knee_flexion_phase_offset_abs: float = 0.20
    profile: str = DIAGNOSIS_CONFIG_PROFILE

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_turn_feature_samples, bool)
            or not isinstance(self.minimum_turn_feature_samples, int)
            or self.minimum_turn_feature_samples < 1
        ):
            raise ValueError("minimum_turn_feature_samples must be a positive non-bool integer")
        for name in (
            "minimum_turn_feature_coverage",
            "limited_knee_flexion_range_deg",
            "knee_asymmetry_median_deg",
            "knee_flexion_phase_offset_abs",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite non-bool numeric")
        if not 0 < self.minimum_turn_feature_coverage <= 1:
            raise ValueError("minimum_turn_feature_coverage must be in (0, 1]")
        if self.limited_knee_flexion_range_deg < 0 or self.knee_asymmetry_median_deg < 0:
            raise ValueError("angle thresholds must be non-negative")
        if not 0 < self.knee_flexion_phase_offset_abs <= 1:
            raise ValueError("knee_flexion_phase_offset_abs must be in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosisResult:
    status: DiagnosisResultStatus
    sport_type: str
    sport_type_source: str
    rule_evaluations: tuple[dict[str, object], ...]
    diagnoses: tuple[dict[str, object], ...]
    blockers: tuple[str, ...]
    config: DiagnosisRuleConfig
    limitations: tuple[str, ...]
    contract_version: str = DIAGNOSIS_CONTRACT_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "status": self.status.value,
            "sport_type": self.sport_type,
            "sport_type_source": self.sport_type_source,
            "rule_evaluations": list(self.rule_evaluations),
            "diagnoses": list(self.diagnoses),
            "blockers": list(self.blockers),
            "config": self.config.to_dict(),
            "DIAGNOSIS_CONFIDENCE_STATUS": "NOT_CALIBRATED",
            "SEVERITY_STATUS": "NOT_CALIBRATED",
            "DIAGNOSIS_ML_FEATURE_VECTOR_STATUS": "NOT_FROZEN",
            "limitations": list(self.limitations),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False, indent=indent)
