"""Cross-representation consistency checks for DiagnosisResult truth."""

from __future__ import annotations

from .registry import RULE_REGISTRY


def validate_diagnosis_truth_consistency(diagnosis_result) -> None:
    payload = (
        diagnosis_result.to_dict() if hasattr(diagnosis_result, "to_dict") else diagnosis_result
    )
    if not isinstance(payload, dict):
        raise ValueError("DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT")
    known_codes = {rule.diagnosis_code.value for rule in RULE_REGISTRY}
    evaluation_pairs: dict[tuple[str, str], str] = {}
    for evaluation in payload.get("rule_evaluations", ()):
        code = evaluation.get("diagnosis_code")
        turn_id = evaluation.get("turn_id")
        status = evaluation.get("status")
        if (
            not isinstance(code, str)
            or not code
            or code not in known_codes
            or not isinstance(turn_id, str)
            or not turn_id
            or status not in {"TRIGGERED", "NOT_TRIGGERED", "NOT_EVALUABLE"}
            or evaluation.get("triggered") is not (status == "TRIGGERED")
        ):
            raise ValueError("DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT")
        pair = (code, turn_id)
        if pair in evaluation_pairs:
            raise ValueError("DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT")
        evaluation_pairs[pair] = status
    diagnosis_pairs = set()
    for diagnosis in payload.get("diagnoses", ()):
        code = diagnosis.get("diagnosis_code")
        turns = diagnosis.get("affected_turn_ids")
        if (
            not isinstance(code, str)
            or not code
            or code not in known_codes
            or diagnosis.get("evaluation_status") != "TRIGGERED"
            or diagnosis.get("provisional") is not True
            or diagnosis.get("validation_status") != "UNVALIDATED_RESEARCH_RULE"
            or diagnosis.get("severity") is not None
            or diagnosis.get("confidence") is not None
            or not isinstance(turns, list | tuple)
            or not turns
        ):
            raise ValueError("DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT")
        for turn_id in turns:
            if not isinstance(turn_id, str) or not turn_id:
                raise ValueError("DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT")
            pair = (code, turn_id)
            if pair in diagnosis_pairs:
                raise ValueError("DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT")
            diagnosis_pairs.add(pair)
            if evaluation_pairs.get(pair) != "TRIGGERED":
                raise ValueError("DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT")
    triggered_pairs = {pair for pair, status in evaluation_pairs.items() if status == "TRIGGERED"}
    if triggered_pairs != diagnosis_pairs:
        raise ValueError("DIAGNOSIS_TRUTH_CONTRACT_INCONSISTENT")
