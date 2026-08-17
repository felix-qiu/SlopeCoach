"""Pure A7 sport-gated, turn-window, multi-frame research rule engine."""

from __future__ import annotations

import statistics

from .contracts import (
    DiagnosisEvaluationStatus,
    DiagnosisResult,
    DiagnosisResultStatus,
    DiagnosisRuleConfig,
)
from .evidence import collect_turn_feature_evidence
from .registry import RULE_REGISTRY


def diagnose_biomechanics(
    *, sport_type_result, biomechanics_result, turn_segments, config=None
) -> DiagnosisResult:
    settings = config or DiagnosisRuleConfig()
    sport = _value(sport_type_result, "effective_sport_type")
    source = _value(sport_type_result, "effective_source")
    limitations = [
        "IMAGE_SPACE_2D_ONLY",
        "UNVALIDATED_RESEARCH_RULES",
        "NO_DIAGNOSIS_GROUND_TRUTH",
        "PYTHON_RESEARCH_REFERENCE_ONLY",
    ]
    if source == "USER":
        limitations.append("SPORT_TYPE_USER_SELECTED_ROUTING_NOT_GT")
    elif source == "AUTO":
        limitations.append("AUTO_SPORT_TYPE_NOT_PRODUCT_VALIDATED")
    if sport not in {"SKI", "SNOWBOARD"}:
        return DiagnosisResult(
            DiagnosisResultStatus.NOT_ANALYZABLE_SPORT_TYPE_UNKNOWN,
            sport or "UNKNOWN",
            source or "AUTO",
            (),
            (),
            ("SPORT_TYPE_UNKNOWN",),
            settings,
            tuple(limitations),
        )
    turns = [_turn_dict(item) for item in turn_segments]
    _validate_turns(turns)
    frame_facts = [_fact_dict(item) for item in _items(biomechanics_result, "frame_facts")]
    turn_results = [_fact_group_dict(item) for item in _items(biomechanics_result, "turn_features")]
    _validate_frame_facts(frame_facts)
    qualified = [item for item in turns if item.get("status") in {"VALID", "PARTIAL"}]
    if not qualified:
        return DiagnosisResult(
            DiagnosisResultStatus.NOT_ANALYZABLE_NO_QUALIFIED_TURNS,
            sport,
            source,
            (),
            (),
            ("NO_QUALIFIED_TURNS",),
            settings,
            tuple(limitations),
        )
    evaluations = []
    diagnoses = []
    blockers = []
    registry_order = {item.diagnosis_code.value: index for index, item in enumerate(RULE_REGISTRY)}
    for turn in sorted(qualified, key=lambda item: (item["apex_timestamp_us"], item["turn_id"])):
        complete_reason = _complete_turn_reason(turn)
        matching_turn_features = next(
            (item for item in turn_results if item["turn_id"] == turn["turn_id"]), None
        )
        if matching_turn_features and (
            matching_turn_features["temporal_segment_id"] != turn["temporal_segment_id"]
            or matching_turn_features["signal_run_id"] != turn["signal_run_id"]
        ):
            raise ValueError("turn biomechanics segment/run mismatch")
        for definition in RULE_REGISTRY:
            code = definition.diagnosis_code.value
            if sport not in definition.applicable_sport_types:
                evaluation = _not_evaluable(
                    code,
                    turn,
                    "SPORT_TYPE_RULE_NOT_APPLICABLE",
                    phase=definition.phase.value,
                )
            elif complete_reason:
                evaluation = _not_evaluable(
                    code, turn, complete_reason, phase=definition.phase.value
                )
            else:
                evaluation = _evaluate_rule(
                    definition,
                    turn,
                    frame_facts,
                    matching_turn_features,
                    settings,
                )
            evaluations.append(evaluation)
            if evaluation["status"] == DiagnosisEvaluationStatus.NOT_EVALUABLE.value:
                blockers.extend(evaluation["reason_codes"])
            if evaluation["status"] == DiagnosisEvaluationStatus.TRIGGERED.value:
                diagnoses.append(
                    {
                        "diagnosis_code": code,
                        "sport_type": sport,
                        "evaluation_status": "TRIGGERED",
                        "validation_status": "UNVALIDATED_RESEARCH_RULE",
                        "provisional": True,
                        "severity": None,
                        "confidence": None,
                        "phase": evaluation["phase"],
                        "affected_turn_ids": [turn["turn_id"]],
                        "evidence_frames": evaluation["evidence_frames"],
                        "feature_evidence": evaluation["feature_evidence"],
                        "limitations": evaluation["limitations"],
                    }
                )
    evaluations.sort(
        key=lambda item: (item["turn_apex_timestamp_us"], registry_order[item["diagnosis_code"]])
    )
    diagnoses.sort(
        key=lambda item: (
            next(
                turn["apex_timestamp_us"]
                for turn in turns
                if turn["turn_id"] == item["affected_turn_ids"][0]
            ),
            registry_order[item["diagnosis_code"]],
        )
    )
    if diagnoses:
        status = DiagnosisResultStatus.EXECUTED_WITH_PROVISIONAL_DIAGNOSES
    elif evaluations and all(item["status"] == "NOT_EVALUABLE" for item in evaluations):
        status = DiagnosisResultStatus.NOT_ANALYZABLE_INSUFFICIENT_DIAGNOSIS_EVIDENCE
    else:
        status = DiagnosisResultStatus.EXECUTED_NO_PROVISIONAL_RULES_TRIGGERED
    return DiagnosisResult(
        status,
        sport,
        source,
        tuple(evaluations),
        tuple(diagnoses),
        tuple(dict.fromkeys(blockers)),
        settings,
        tuple(limitations),
    )


def _evaluate_rule(definition, turn, frame_facts, turn_result, config):
    code = definition.diagnosis_code.value
    primary_feature = (
        definition.required_features[-1]
        if len(definition.required_features) > 1
        else definition.required_features[0]
    )
    evidence = collect_turn_feature_evidence(
        turn=turn, frame_facts=frame_facts, feature_id=primary_feature
    )
    if evidence["sample_count"] < config.minimum_turn_feature_samples:
        return _not_evaluable(
            code,
            turn,
            "INSUFFICIENT_FEATURE_SAMPLES",
            evidence,
            phase=definition.phase.value,
        )
    if evidence["coverage"] < config.minimum_turn_feature_coverage:
        return _not_evaluable(
            code,
            turn,
            "INSUFFICIENT_FEATURE_COVERAGE",
            evidence,
            phase=definition.phase.value,
        )
    values = [float(item["value"]) for item in evidence["facts"]]
    if code == "LIMITED_KNEE_FLEXION_MODULATION_2D":
        value = max(values) - min(values)
        statistic, operator = "range", "<"
        threshold = config.limited_knee_flexion_range_deg
        triggered = value < threshold
        unit = "deg"
    elif code == "BILATERAL_KNEE_ASYMMETRY_2D":
        value = statistics.median(values)
        statistic, operator = "median", ">="
        threshold = config.knee_asymmetry_median_deg
        triggered = value >= threshold
        unit = "deg"
    else:
        fact = _turn_fact(turn_result, "minimum_mean_knee_angle_phase_offset")
        if fact is None or fact.get("status") != "AVAILABLE" or fact.get("value") is None:
            return _not_evaluable(
                code,
                turn,
                "TURN_FEATURE_UNAVAILABLE",
                evidence,
                phase=definition.phase.value,
            )
        value = abs(float(fact["value"]))
        statistic, operator = "absolute_value", ">"
        threshold = config.knee_flexion_phase_offset_abs
        triggered = value > threshold
        unit = "ratio"
    status = "TRIGGERED" if triggered else "NOT_TRIGGERED"
    feature_evidence = _feature_evidence(
        primary_feature
        if code != "KNEE_FLEXION_TIMING_OFFSET_2D"
        else "minimum_mean_knee_angle_phase_offset",
        unit,
        statistic,
        value,
        operator,
        threshold,
        evidence,
    )
    if code == "KNEE_FLEXION_TIMING_OFFSET_2D":
        minimum = _turn_fact(turn_result, "minimum_mean_knee_angle_timestamp_us")
        feature_evidence["selected_minimum_timestamp_us"] = (
            minimum.get("value") if minimum else None
        )
        feature_evidence["apex_timestamp_us"] = turn["apex_timestamp_us"]
    return {
        "diagnosis_code": code,
        "turn_id": turn["turn_id"],
        "turn_apex_timestamp_us": turn["apex_timestamp_us"],
        "status": status,
        "triggered": triggered,
        "phase": definition.phase.value,
        "reason_codes": [],
        "evidence_frames": evidence["evidence_timestamps_us"],
        "feature_evidence": [feature_evidence],
        "limitations": list(definition.limitations),
    }


def _feature_evidence(feature_id, unit, statistic, value, operator, threshold, evidence):
    return {
        "feature_id": feature_id,
        "unit": unit,
        "measurement_space": "IMAGE_SPACE_2D",
        "statistic": statistic,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "sample_count": evidence["sample_count"],
        "eligible_sample_count": evidence["eligible_sample_count"],
        "coverage": evidence["coverage"],
        "evidence_timestamps_us": evidence["evidence_timestamps_us"],
        "minimum_support_confidence": evidence["minimum_support_confidence"],
        "interpolated_sample_count": evidence["interpolated_sample_count"],
        "observed_sample_count": evidence["observed_sample_count"],
        "limitations": ["CAMERA_VIEW_DEPENDENT"],
    }


def _not_evaluable(code, turn, reason, evidence=None, *, phase):
    return {
        "diagnosis_code": code,
        "turn_id": turn["turn_id"],
        "turn_apex_timestamp_us": turn["apex_timestamp_us"],
        "status": "NOT_EVALUABLE",
        "triggered": False,
        "phase": phase,
        "reason_codes": [reason],
        "evidence_frames": (evidence or {}).get("evidence_timestamps_us", []),
        "feature_evidence": [],
        "limitations": ["INSUFFICIENT_DIAGNOSIS_EVIDENCE"],
    }


def _complete_turn_reason(turn):
    if (
        turn.get("status") != "VALID"
        or turn.get("start_timestamp_us") is None
        or turn.get("end_timestamp_us") is None
    ):
        return "TURN_BOUNDARY_UNAVAILABLE"
    if not turn["start_timestamp_us"] < turn["apex_timestamp_us"] < turn["end_timestamp_us"]:
        raise ValueError("complete turn must satisfy start < apex < end")
    if not _positive_int(turn.get("temporal_segment_id")):
        return "TEMPORAL_SEGMENT_MISMATCH"
    if not _positive_int(turn.get("signal_run_id")):
        return "SIGNAL_RUN_MISMATCH"
    return None


def _validate_turns(turns):
    ids = [item.get("turn_id") for item in turns]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate turn IDs")
    for item in turns:
        for field in ("apex_timestamp_us", "start_timestamp_us", "end_timestamp_us"):
            value = item.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"invalid {field}")


def _validate_frame_facts(facts):
    seen = set()
    for fact in facts:
        timestamp = fact.get("timestamp_us")
        if timestamp is not None and (
            isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0
        ):
            raise ValueError("frame fact timestamp must be a non-negative integer")
        key = (timestamp, fact.get("temporal_segment_id"), fact.get("feature_id"))
        if timestamp is not None and key in seen:
            raise ValueError(f"duplicate frame feature fact: {key}")
        seen.add(key)


def _turn_fact(turn_result, feature_id):
    if not turn_result:
        return None
    return next(
        (item for item in turn_result["facts"] if item.get("feature_id") == feature_id), None
    )


def _value(obj, name):
    value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name)
    return value.value if hasattr(value, "value") else value


def _items(obj, name):
    return obj.get(name, ()) if isinstance(obj, dict) else getattr(obj, name)


def _turn_dict(item):
    return item if isinstance(item, dict) else item.to_dict()


def _fact_dict(item):
    return item if isinstance(item, dict) else item.to_dict()


def _fact_group_dict(item):
    return item if isinstance(item, dict) else item.to_dict()


def _positive_int(value):
    return not isinstance(value, bool) and isinstance(value, int) and value > 0
