"""Fail-closed compatibility gate for persisted A7 diagnosis artifacts."""

from __future__ import annotations

from slopecoach_ml.diagnosis import (
    DIAGNOSIS_CONTRACT_VERSION,
    DIAGNOSIS_RULE_REGISTRY_SHA256,
    build_diagnosis_semantics_provenance,
    normalize_diagnosis_config,
    validate_diagnosis_semantics_provenance,
    validate_diagnosis_truth_consistency,
)


def validate_diagnosis_artifact_compatibility(
    artifact: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if artifact.get("benchmark_contract_version") != "ski-bench-diagnosis-v1":
        raise ValueError("DIAGNOSIS_BENCHMARK_CONTRACT_INCOMPATIBLE")
    diagnosis = artifact.get("diagnosis_result")
    if not isinstance(diagnosis, dict):
        raise ValueError("ARTIFACT_MISSING_DIAGNOSIS_RESULT")
    if diagnosis.get("contract_version") != DIAGNOSIS_CONTRACT_VERSION:
        raise ValueError("DIAGNOSIS_CONTRACT_INCOMPATIBLE")
    source_registry = artifact.get("diagnosis_rule_registry_sha256")
    top_config = artifact.get("diagnosis_config")
    result_config = diagnosis.get("config")
    if not isinstance(source_registry, str) or not source_registry:
        raise ValueError("DIAGNOSIS_SEMANTIC_PROVENANCE_MISSING")
    if top_config is None or result_config is None:
        raise ValueError("DIAGNOSIS_SEMANTIC_PROVENANCE_MISSING")
    normalized_top = normalize_diagnosis_config(top_config)
    normalized_result = normalize_diagnosis_config(result_config)
    if normalized_top != normalized_result:
        raise ValueError("DIAGNOSIS_CONFIG_PROVENANCE_MISMATCH")

    explicit = artifact.get("diagnosis_semantics_provenance")
    if explicit is None:
        provenance = build_diagnosis_semantics_provenance(
            normalized_top,
            diagnosis_contract_version=DIAGNOSIS_CONTRACT_VERSION,
            diagnosis_rule_registry_sha256=source_registry,
        )
        origin = "LEGACY_EXPLICIT_FIELDS_DERIVED"
    else:
        provenance = validate_diagnosis_semantics_provenance(explicit)
        origin = "EXPLICIT_SEMANTIC_PROVENANCE"
        if provenance.diagnosis_contract_version != diagnosis["contract_version"]:
            raise ValueError("DIAGNOSIS_CONTRACT_INCOMPATIBLE")
        if provenance.diagnosis_rule_registry_sha256 != source_registry:
            raise ValueError("DIAGNOSIS_RULE_REGISTRY_PROVENANCE_MISMATCH")
        if provenance.diagnosis_config != normalized_top:
            raise ValueError("DIAGNOSIS_CONFIG_PROVENANCE_MISMATCH")
    if source_registry != DIAGNOSIS_RULE_REGISTRY_SHA256:
        raise ValueError("DIAGNOSIS_RULE_REGISTRY_INCOMPATIBLE")
    validate_diagnosis_truth_consistency(diagnosis)
    compatibility = {
        "diagnosis_contract_compatible": True,
        "diagnosis_registry_compatible": True,
        "diagnosis_config_consistent": True,
        "diagnosis_semantics_fingerprint_valid": True,
        "diagnosis_provenance_origin": origin,
        "status": "COMPATIBLE",
    }
    return diagnosis, provenance.to_dict(), compatibility
