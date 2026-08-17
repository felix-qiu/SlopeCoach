"""Deterministic provenance for the exact A7 diagnosis semantics used."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .contracts import DIAGNOSIS_CONTRACT_VERSION, DiagnosisRuleConfig
from .registry import DIAGNOSIS_RULE_REGISTRY_SHA256

DIAGNOSIS_SEMANTICS_PROVENANCE_VERSION = "diagnosis-semantics-provenance-v1"
_CONFIG_FIELDS = tuple(DiagnosisRuleConfig().to_dict())


def normalize_diagnosis_config(config) -> dict[str, object]:
    payload = config.to_dict() if isinstance(config, DiagnosisRuleConfig) else config
    if not isinstance(payload, dict) or set(payload) != set(_CONFIG_FIELDS):
        raise ValueError("DIAGNOSIS_SEMANTIC_PROVENANCE_MISSING")
    try:
        normalized = DiagnosisRuleConfig(**payload).to_dict()
    except (TypeError, ValueError) as exc:
        raise ValueError("DIAGNOSIS_CONFIG_PROVENANCE_INVALID") from exc
    return normalized


def canonical_diagnosis_config_json(config) -> str:
    return json.dumps(
        normalize_diagnosis_config(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def diagnosis_config_sha256(config) -> str:
    return hashlib.sha256(canonical_diagnosis_config_json(config).encode()).hexdigest()


@dataclass(frozen=True)
class DiagnosisSemanticsProvenance:
    diagnosis_contract_version: str
    diagnosis_rule_registry_sha256: str
    diagnosis_config: dict[str, object]
    diagnosis_config_sha256: str
    diagnosis_semantics_sha256: str
    version: str = DIAGNOSIS_SEMANTICS_PROVENANCE_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "diagnosis_contract_version": self.diagnosis_contract_version,
            "diagnosis_rule_registry_sha256": self.diagnosis_rule_registry_sha256,
            "diagnosis_config": dict(self.diagnosis_config),
            "diagnosis_config_sha256": self.diagnosis_config_sha256,
            "diagnosis_semantics_sha256": self.diagnosis_semantics_sha256,
        }


def canonical_diagnosis_semantics_json(
    *,
    provenance_version: str,
    diagnosis_contract_version: str,
    diagnosis_rule_registry_sha256: str,
    diagnosis_config_sha256_value: str,
) -> str:
    payload = {
        "version": provenance_version,
        "diagnosis_contract_version": diagnosis_contract_version,
        "diagnosis_rule_registry_sha256": diagnosis_rule_registry_sha256,
        "diagnosis_config_sha256": diagnosis_config_sha256_value,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_diagnosis_semantics_provenance(
    config,
    *,
    diagnosis_contract_version: str = DIAGNOSIS_CONTRACT_VERSION,
    diagnosis_rule_registry_sha256: str = DIAGNOSIS_RULE_REGISTRY_SHA256,
) -> DiagnosisSemanticsProvenance:
    normalized = normalize_diagnosis_config(config)
    config_sha = diagnosis_config_sha256(normalized)
    semantic_json = canonical_diagnosis_semantics_json(
        provenance_version=DIAGNOSIS_SEMANTICS_PROVENANCE_VERSION,
        diagnosis_contract_version=diagnosis_contract_version,
        diagnosis_rule_registry_sha256=diagnosis_rule_registry_sha256,
        diagnosis_config_sha256_value=config_sha,
    )
    return DiagnosisSemanticsProvenance(
        diagnosis_contract_version=diagnosis_contract_version,
        diagnosis_rule_registry_sha256=diagnosis_rule_registry_sha256,
        diagnosis_config=normalized,
        diagnosis_config_sha256=config_sha,
        diagnosis_semantics_sha256=hashlib.sha256(semantic_json.encode()).hexdigest(),
    )


def validate_diagnosis_semantics_provenance(payload) -> DiagnosisSemanticsProvenance:
    if not isinstance(payload, dict):
        raise ValueError("DIAGNOSIS_SEMANTIC_PROVENANCE_MISSING")
    required = {
        "version",
        "diagnosis_contract_version",
        "diagnosis_rule_registry_sha256",
        "diagnosis_config",
        "diagnosis_config_sha256",
        "diagnosis_semantics_sha256",
    }
    if set(payload) != required:
        raise ValueError("DIAGNOSIS_SEMANTIC_PROVENANCE_MISSING")
    if payload["version"] != DIAGNOSIS_SEMANTICS_PROVENANCE_VERSION:
        raise ValueError("DIAGNOSIS_SEMANTIC_PROVENANCE_INCOMPATIBLE")
    expected = build_diagnosis_semantics_provenance(
        payload["diagnosis_config"],
        diagnosis_contract_version=str(payload["diagnosis_contract_version"]),
        diagnosis_rule_registry_sha256=str(payload["diagnosis_rule_registry_sha256"]),
    )
    if payload["diagnosis_config_sha256"] != expected.diagnosis_config_sha256:
        raise ValueError("DIAGNOSIS_CONFIG_SHA256_INVALID")
    if payload["diagnosis_semantics_sha256"] != expected.diagnosis_semantics_sha256:
        raise ValueError("DIAGNOSIS_SEMANTICS_SHA256_INVALID")
    return expected
