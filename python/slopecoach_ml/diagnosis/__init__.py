"""A7 evidence-backed provisional research diagnosis API."""

from .contracts import (
    DIAGNOSIS_BENCHMARK_CONTRACT_VERSION,
    DIAGNOSIS_CONFIG_PROFILE,
    DIAGNOSIS_CONTRACT_VERSION,
    DIAGNOSIS_RULE_REGISTRY_VERSION,
    DIAGNOSIS_RULE_SCHEMA_VERSION,
    DiagnosisCode,
    DiagnosisEvaluationStatus,
    DiagnosisPhase,
    DiagnosisResult,
    DiagnosisResultStatus,
    DiagnosisRuleConfig,
)
from .evidence import collect_turn_feature_evidence
from .golden import run_diagnosis_golden
from .pipeline import diagnose_biomechanics
from .registry import (
    DIAGNOSIS_RULE_REGISTRY_SHA256,
    RULE_REGISTRY,
    canonical_rule_registry_json,
)

__all__ = [
    "DIAGNOSIS_BENCHMARK_CONTRACT_VERSION",
    "DIAGNOSIS_CONFIG_PROFILE",
    "DIAGNOSIS_CONTRACT_VERSION",
    "DIAGNOSIS_RULE_REGISTRY_SHA256",
    "DIAGNOSIS_RULE_REGISTRY_VERSION",
    "DIAGNOSIS_RULE_SCHEMA_VERSION",
    "DiagnosisCode",
    "DiagnosisEvaluationStatus",
    "DiagnosisPhase",
    "DiagnosisResult",
    "DiagnosisResultStatus",
    "DiagnosisRuleConfig",
    "RULE_REGISTRY",
    "canonical_rule_registry_json",
    "collect_turn_feature_evidence",
    "diagnose_biomechanics",
    "run_diagnosis_golden",
]
