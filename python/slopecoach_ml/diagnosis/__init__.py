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
from .provenance import (
    DIAGNOSIS_SEMANTICS_PROVENANCE_VERSION,
    DiagnosisSemanticsProvenance,
    build_diagnosis_semantics_provenance,
    canonical_diagnosis_config_json,
    canonical_diagnosis_semantics_json,
    diagnosis_config_sha256,
    normalize_diagnosis_config,
    validate_diagnosis_semantics_provenance,
)
from .registry import (
    DIAGNOSIS_RULE_REGISTRY_SHA256,
    RULE_REGISTRY,
    canonical_rule_registry_json,
)
from .truth import validate_diagnosis_truth_consistency

__all__ = [
    "DIAGNOSIS_BENCHMARK_CONTRACT_VERSION",
    "DIAGNOSIS_CONFIG_PROFILE",
    "DIAGNOSIS_CONTRACT_VERSION",
    "DIAGNOSIS_RULE_REGISTRY_SHA256",
    "DIAGNOSIS_RULE_REGISTRY_VERSION",
    "DIAGNOSIS_RULE_SCHEMA_VERSION",
    "DIAGNOSIS_SEMANTICS_PROVENANCE_VERSION",
    "DiagnosisCode",
    "DiagnosisEvaluationStatus",
    "DiagnosisPhase",
    "DiagnosisResult",
    "DiagnosisResultStatus",
    "DiagnosisRuleConfig",
    "DiagnosisSemanticsProvenance",
    "RULE_REGISTRY",
    "canonical_rule_registry_json",
    "build_diagnosis_semantics_provenance",
    "canonical_diagnosis_config_json",
    "canonical_diagnosis_semantics_json",
    "collect_turn_feature_evidence",
    "diagnose_biomechanics",
    "run_diagnosis_golden",
    "diagnosis_config_sha256",
    "normalize_diagnosis_config",
    "validate_diagnosis_semantics_provenance",
    "validate_diagnosis_truth_consistency",
]
