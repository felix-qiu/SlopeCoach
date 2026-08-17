"""Agreement-aware calibrated research fusion in estimated LLR space."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

from .artifact import compatible_channel, validate_artifact_fingerprint
from .contracts import (
    CALIBRATED_FUSION_CONTROLS_ROUTING,
    CALIBRATED_FUSION_VERSION,
    SPORT_EVIDENCE_CALIBRATION_CONTRACT_VERSION,
    AgreementState,
    CalibratedFusionStatus,
)
from .platt import logit, stable_sigmoid


def unavailable_fusion(status=CalibratedFusionStatus.NOT_AVAILABLE_NO_CALIBRATION_ARTIFACT):
    return {
        "contract_version": CALIBRATED_FUSION_VERSION,
        "status": status.value,
        "calibrated_ski_probability": None,
        "calibrated_snowboard_probability": None,
        "fused_log_odds": None,
        "available_calibration_channels": [],
        "available_evidence_kinds": [],
        "provider_evidence": [],
        "kind_evidence": [],
        "agreement_state": AgreementState.NO_CALIBRATED_EVIDENCE.value,
        "routing_eligible": False,
        "reason_codes": [status.value],
        "limitations": [
            "CALIBRATED_FUSION_DOES_NOT_CONTROL_ROUTING",
            "PYTHON_RESEARCH_REFERENCE_ONLY",
        ],
        "CALIBRATED_FUSION_CONTROLS_ROUTING": CALIBRATED_FUSION_CONTROLS_ROUTING,
    }


def apply_calibrated_fusion(
    summaries: list[dict[str, object]], artifact: dict[str, object] | None
) -> dict[str, object]:
    started = time.perf_counter()
    if artifact is None:
        result = unavailable_fusion()
        result["performance"] = {"calibrated_fusion_seconds": time.perf_counter() - started}
        return result
    if artifact.get(
        "contract_version"
    ) != SPORT_EVIDENCE_CALIBRATION_CONTRACT_VERSION or not validate_artifact_fingerprint(artifact):
        result = unavailable_fusion(CalibratedFusionStatus.CALIBRATION_ARTIFACT_INCOMPATIBLE)
        result["performance"] = {"calibrated_fusion_seconds": time.perf_counter() - started}
        return result
    if artifact.get("status") != "RESEARCH_CALIBRATION_AVAILABLE":
        result = unavailable_fusion(
            CalibratedFusionStatus.NOT_AVAILABLE_NO_VALID_CALIBRATION_ARTIFACT
        )
        result["performance"] = {"calibrated_fusion_seconds": time.perf_counter() - started}
        return result
    channels = {
        item["calibration_channel_id"]: item
        for item in artifact.get("channels", [])
        if item.get("status") == "ACCEPTED_RESEARCH_CALIBRATION"
    }
    provider_evidence = []
    incompatible = False
    epsilon = artifact["fit_config"]["probability_epsilon"]
    for summary in summaries:
        channel = channels.get(summary["calibration_channel_id"])
        if channel is None:
            continue
        if not compatible_channel(channel, summary):
            incompatible = True
            continue
        p = stable_sigmoid(channel["slope_a"] * summary["raw_direction"] + channel["intercept_b"])
        llr = logit(p, epsilon) - logit(channel["training_snowboard_prior"], epsilon)
        provider_evidence.append(
            {
                "calibration_channel_id": channel["calibration_channel_id"],
                "provider_name": channel["provider_name"],
                "evidence_kind": channel["evidence_kind"],
                "calibrated_snowboard_probability": p,
                "calibrated_ski_probability": 1 - p,
                "estimated_calibrated_log_likelihood_ratio": llr,
            }
        )
    if incompatible:
        result = unavailable_fusion(CalibratedFusionStatus.CALIBRATION_ARTIFACT_INCOMPATIBLE)
        result["performance"] = {"calibrated_fusion_seconds": time.perf_counter() - started}
        return result
    if not provider_evidence:
        result = unavailable_fusion(
            CalibratedFusionStatus.NOT_AVAILABLE_INSUFFICIENT_CALIBRATED_CHANNELS
        )
        result["performance"] = {"calibrated_fusion_seconds": time.perf_counter() - started}
        return result
    grouped = defaultdict(list)
    for item in provider_evidence:
        grouped[item["evidence_kind"]].append(item)
    kind_evidence = []
    for kind in sorted(grouped):
        values = grouped[kind]
        kind_llr = sum(item["estimated_calibrated_log_likelihood_ratio"] for item in values) / len(
            values
        )
        kind_evidence.append(
            {
                "evidence_kind": kind,
                "provider_count": len(values),
                "estimated_calibrated_log_likelihood_ratio": kind_llr,
                "calibrated_snowboard_probability": stable_sigmoid(kind_llr),
            }
        )
    threshold = artifact["fusion"]["calibrated_conflict_probability"]
    probabilities = [item["calibrated_snowboard_probability"] for item in kind_evidence]
    conflict = any(item >= threshold for item in probabilities) and any(
        item <= 1 - threshold for item in probabilities
    )
    llrs = [item["estimated_calibrated_log_likelihood_ratio"] for item in kind_evidence]
    fused = logit(artifact["fusion"]["fusion_prior_snowboard"], epsilon) + sum(llrs)
    snowboard = stable_sigmoid(fused)
    agreement = _agreement(llrs, conflict)
    status = (
        CalibratedFusionStatus.CONFLICTING_CALIBRATED_PRIMARY_EVIDENCE
        if conflict
        else CalibratedFusionStatus.AVAILABLE_SINGLE_PRIMARY_KIND
        if len(kind_evidence) == 1
        else CalibratedFusionStatus.AVAILABLE_MULTIPLE_PRIMARY_KINDS
    )
    return {
        "contract_version": CALIBRATED_FUSION_VERSION,
        "status": status.value,
        "calibrated_ski_probability": 1 - snowboard,
        "calibrated_snowboard_probability": snowboard,
        "fused_log_odds": fused,
        "available_calibration_channels": sorted(
            item["calibration_channel_id"] for item in provider_evidence
        ),
        "available_evidence_kinds": sorted(grouped),
        "provider_evidence": provider_evidence,
        "kind_evidence": kind_evidence,
        "agreement_state": agreement.value,
        "routing_eligible": False,
        "reason_codes": [status.value],
        "limitations": [
            "RESEARCH_DIAGNOSTIC_ONLY",
            "CALIBRATED_FUSION_DOES_NOT_CONTROL_ROUTING",
            "FUSION_PRIOR_IS_RESEARCH_NEUTRAL_NOT_PRODUCT_PREVALENCE",
        ],
        "CALIBRATED_FUSION_CONTROLS_ROUTING": CALIBRATED_FUSION_CONTROLS_ROUTING,
        "performance": {"calibrated_fusion_seconds": time.perf_counter() - started},
    }


def run_calibration_golden(path: str | Path) -> dict[str, object]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for case in fixture["cases"]:
        llrs = case["provider_llrs"]
        grouped = defaultdict(list)
        for item in llrs:
            grouped[item["evidence_kind"]].append(item["llr"])
        kind = {key: sum(values) / len(values) for key, values in grouped.items()}
        values = list(kind.values())
        conflict = any(stable_sigmoid(value) >= 0.8 for value in values) and any(
            stable_sigmoid(value) <= 0.2 for value in values
        )
        fused = sum(values)
        agreement = _agreement(values, conflict).value
        actual = stable_sigmoid(fused)
        passed = (
            math.isclose(fused, case["expected_fused_log_odds"], abs_tol=1e-12)
            and math.isclose(actual, case["expected_snowboard_probability"], abs_tol=1e-12)
            and agreement == case["expected_agreement_state"]
        )
        cases.append({"case_id": case["case_id"], "passed": passed, "fused_log_odds": fused})
    return {
        "contract_version": CALIBRATED_FUSION_VERSION,
        "golden_passed": all(item["passed"] for item in cases),
        "cases": cases,
    }


def _agreement(llrs: list[float], conflict: bool) -> AgreementState:
    if conflict:
        return AgreementState.CONFLICT
    if len(llrs) == 1:
        return (
            AgreementState.SINGLE_KIND_SNOWBOARD if llrs[0] > 0 else AgreementState.SINGLE_KIND_SKI
        )
    if all(item > 0 for item in llrs):
        return AgreementState.AGREE_SNOWBOARD
    if all(item < 0 for item in llrs):
        return AgreementState.AGREE_SKI
    return AgreementState.WEAK_OR_MIXED
