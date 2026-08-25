"""Single-pass B3 product orchestration over existing A3-A9 reference modules."""

from __future__ import annotations

import hashlib
from collections import Counter

from slopecoach_ml.analysis_result import build_analysis_result, build_product_report
from slopecoach_ml.biomechanics import (
    FEATURE_REGISTRY_SHA256,
    FEATURE_REGISTRY_V1,
    FRAME_FEATURE_REGISTRY_V1,
    TEMPORAL_FEATURE_REGISTRY_V1,
    TURN_FEATURE_REGISTRY_V1,
)
from slopecoach_ml.coach import build_coach_context, build_coach_report
from slopecoach_ml.diagnosis import (
    build_diagnosis_semantics_provenance,
    diagnose_biomechanics,
)

from .analysis import build_mvp_analysis_payload
from .sport_type import MvpSportTypeProvenance


def assemble_analyze_video_product(
    *,
    video: str,
    sport_type: MvpSportTypeProvenance,
    biomechanics_report: dict[str, object],
) -> dict[str, object]:
    """Consume one completed A5 pass and assemble deterministic A7-A9 output."""

    source = _source_summary(video, biomechanics_report)
    target = _target_summary(biomechanics_report)
    turns = _turn_summary(biomechanics_report)
    biomechanics = _biomechanics_summary(biomechanics_report)
    sport = {
        **sport_type.to_dict(),
        "CALIBRATED_FUSION_CONTROLS_ROUTING": False,
        "limitations": ["SPORT_TYPE_USER_SELECTED_ROUTING_NOT_GT"],
    }

    diagnosis = provenance = scorecard = coach = None
    unavailable = {}
    if not target["safe_for_analysis"]:
        unavailable = _downstream_unavailable("TARGET_IDENTITY_UNCERTAIN")
    elif turns["qualified_turn_count"] == 0:
        unavailable = _downstream_unavailable("NO_QUALIFIED_TURNS")
    else:
        diagnosis_result = diagnose_biomechanics(
            sport_type_result=sport,
            biomechanics_result=biomechanics_report["biomechanics_result"],
            turn_segments=biomechanics_report["turn_segments"],
        )
        diagnosis = diagnosis_result.to_dict()
        provenance = build_diagnosis_semantics_provenance(
            diagnosis_result.config,
            diagnosis_contract_version=diagnosis_result.contract_version,
        ).to_dict()
        coach_context = build_coach_context(diagnosis_result)
        scorecard = coach_context.scorecard
        coach = build_coach_report(coach_context).to_dict()

    analysis_result = build_analysis_result(
        source=source,
        target_identity=target,
        sport_type=sport,
        turns=turns,
        biomechanics=biomechanics,
        diagnosis=diagnosis,
        diagnosis_semantics_provenance=provenance,
        scorecard=scorecard,
        coach=coach,
        unavailable_reasons=unavailable,
    )
    product_report = build_product_report(analysis_result)
    return build_mvp_analysis_payload(
        video=video,
        sport_type=sport_type,
        analysis_result=analysis_result,
        product_report=product_report,
        pipeline_provenance=_pipeline_provenance(biomechanics_report),
    )


def _source_summary(video: str, report: dict[str, object]) -> dict[str, object]:
    metadata = _mapping(report, "video")
    duration_seconds = metadata.get("duration_seconds")
    return {
        "source_video_id": "request-path-sha256:"
        + hashlib.sha256(video.encode("utf-8")).hexdigest(),
        "duration_us": (
            round(float(duration_seconds) * 1_000_000)
            if isinstance(duration_seconds, int | float) and not isinstance(duration_seconds, bool)
            else None
        ),
        "width_px": metadata.get("width_px"),
        "height_px": metadata.get("height_px"),
        "limitations": ["SOURCE_VIDEO_ID_DERIVED_FROM_EXPLICIT_REQUEST_PATH_NOT_CONTENT_HASH"],
    }


def _target_summary(report: dict[str, object]) -> dict[str, object]:
    identity = _mapping(report, "identity_input")
    frame = _mapping(report, "frame_biomechanics")
    locked = _nonnegative_count(identity.get("identity_locked_frame_count"))
    unsafe = _nonnegative_count(identity.get("identity_unsafe_frame_count"))
    trusted = _nonnegative_count(frame.get("trusted_frame_count"))
    safe = locked > 0 and trusted > 0
    total = locked + unsafe
    limitations = ["TARGET_IDENTITY_GT_NOT_AVAILABLE"]
    if not safe:
        limitations.append("TARGET_IDENTITY_UNCERTAIN")
    return {
        "state": "LOCKED" if safe else "UNSAFE_NO_TRUSTED_TARGET_POSE",
        "safe_for_analysis": safe,
        "selection_mode": "MANUAL_SEED" if "manual_target_seed" in report else "AUTO_INITIAL",
        "coverage": locked / total if total else None,
        "TARGET_IDENTITY_GT_ANNOTATION_STATUS": "DEFERRED",
        "limitations": limitations,
    }


def _turn_summary(report: dict[str, object]) -> dict[str, object]:
    segments = report.get("turn_segments")
    if not isinstance(segments, list) or any(not isinstance(item, dict) for item in segments):
        raise ValueError("PRODUCT_TURN_SEGMENTS_MISSING")
    counts = Counter(str(item.get("status")) for item in segments)
    valid = counts["VALID"]
    partial = counts["PARTIAL"]
    rejected = len(segments) - valid - partial
    rejection_reasons = Counter(
        str(item.get("status") or "UNKNOWN_REJECTION_REASON")
        for item in segments
        if item.get("status") not in {"VALID", "PARTIAL"}
    )
    complete = sum(
        item.get("status") == "VALID"
        and item.get("start_timestamp_us") is not None
        and item.get("end_timestamp_us") is not None
        for item in segments
    )
    return {
        "turn_candidate_count": len(segments),
        "qualified_turn_count": valid + partial,
        "valid_turn_count": valid,
        "partial_turn_count": partial,
        "rejected_turn_count": rejected,
        "complete_diagnosis_eligible_turn_count": complete,
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "TURN_SEGMENTATION_GT_STATUS": "NOT_AVAILABLE",
        "limitations": ["NO_TURN_GT"],
    }


def _biomechanics_summary(report: dict[str, object]) -> dict[str, object]:
    result = _mapping(report, "biomechanics_result")
    frame_facts = _list(result, "frame_facts")
    _list(result, "temporal_segment_features")
    turn_features = _list(result, "turn_features")
    return {
        "feature_registry_count": len(FEATURE_REGISTRY_V1),
        "feature_registry_sha256": str(
            report.get("feature_registry_sha256") or FEATURE_REGISTRY_SHA256
        ),
        "frame_feature_count": len(FRAME_FEATURE_REGISTRY_V1),
        "temporal_feature_count": len(TEMPORAL_FEATURE_REGISTRY_V1),
        "turn_feature_count": len(TURN_FEATURE_REGISTRY_V1),
        "frame_fact_count": len(frame_facts),
        "turn_feature_group_count": len(turn_features),
        "availability_summary": result.get("feature_coverage"),
        "coverage_summary": {
            "trusted_frame_count": _mapping(report, "frame_biomechanics").get("trusted_frame_count")
        },
        "limitations": list(result.get("limitations", ())),
    }


def _pipeline_provenance(report: dict[str, object]) -> dict[str, object]:
    return {
        "execution_mode": "SINGLE_VIDEO_ANALYSIS_PASS",
        "video_read_count": 1,
        "automatic_sport_type_executed": False,
        "upstream_benchmark_contract_version": report.get("benchmark_contract_version"),
        "biomechanics_contract_version": _mapping(report, "biomechanics_result").get(
            "contract_version"
        ),
        "feature_schema_version": report.get("feature_schema_version"),
        "models": report.get("models"),
        "runtime": report.get("runtime"),
        "performance": report.get("performance"),
    }


def _downstream_unavailable(reason: str) -> dict[str, str]:
    return {name: reason for name in ("DIAGNOSIS", "SCORECARD", "COACH")}


def _mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"PRODUCT_{key.upper()}_MISSING")
    return value


def _list(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"PRODUCT_{key.upper()}_MISSING")
    return value


def _nonnegative_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("PRODUCT_COUNT_INVALID")
    return value
