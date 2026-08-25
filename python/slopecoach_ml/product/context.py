"""Typed, deterministic boundary between A5 evidence and product orchestration."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass

from slopecoach_ml.biomechanics import (
    FEATURE_REGISTRY_SHA256,
    FEATURE_REGISTRY_V1,
    FRAME_FEATURE_REGISTRY_V1,
    TEMPORAL_FEATURE_REGISTRY_V1,
    TURN_FEATURE_REGISTRY_V1,
)

from .sport_type import MvpSportTypeProvenance

ANALYSIS_CONTEXT_VERSION = "mvp-analysis-context-v1"


@dataclass(frozen=True)
class AnalysisContext:
    """Immutable-at-the-boundary snapshot used by the B3 product pipeline.

    The context is an internal Python product-orchestration model. It does not replace any
    A7-A9 contract and is not the future Rust production source of truth.
    """

    video: str
    sport_type: MvpSportTypeProvenance
    source: dict[str, object]
    target_identity: dict[str, object]
    turns: dict[str, object]
    biomechanics: dict[str, object]
    biomechanics_result: dict[str, object]
    turn_segments: tuple[dict[str, object], ...]
    pipeline_provenance: dict[str, object]
    contract_version: str = ANALYSIS_CONTEXT_VERSION

    @classmethod
    def from_biomechanics_report(
        cls,
        *,
        video: str,
        sport_type: MvpSportTypeProvenance,
        biomechanics_report: dict[str, object],
    ) -> AnalysisContext:
        """Validate and snapshot the only raw A5 dictionary accepted by product code."""

        if not isinstance(video, str) or not video:
            raise ValueError("PRODUCT_VIDEO_REQUIRED")
        if not isinstance(biomechanics_report, dict):
            raise ValueError("PRODUCT_BIOMECHANICS_REPORT_INVALID")
        result = _mapping(biomechanics_report, "biomechanics_result")
        segments = _dict_list(biomechanics_report, "turn_segments")
        context = cls(
            video=video,
            sport_type=sport_type,
            source=_source_summary(video, biomechanics_report),
            target_identity=_target_summary(biomechanics_report),
            turns=_turn_summary(segments),
            biomechanics=_biomechanics_summary(biomechanics_report, result, segments),
            biomechanics_result=_snapshot(result),
            turn_segments=tuple(_snapshot(item) for item in segments),
            pipeline_provenance=_pipeline_provenance(biomechanics_report, result),
        )
        # Fail at the boundary instead of allowing non-JSON or NaN evidence downstream.
        context.to_dict()
        return context

    @property
    def target_safe_for_analysis(self) -> bool:
        return self.target_identity["safe_for_analysis"] is True

    @property
    def qualified_turn_count(self) -> int:
        return int(self.turns["qualified_turn_count"])

    @property
    def biomechanics_evidence_available(self) -> bool:
        availability = self.biomechanics["availability_summary"]
        return isinstance(availability, dict) and availability.get("status") == "AVAILABLE"

    @property
    def analysis_context_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._semantic_dict()).encode("utf-8")).hexdigest()

    def _semantic_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "video": self.video,
            "sport_type": self.sport_type.to_dict(),
            "source": self.source,
            "target_identity": self.target_identity,
            "turns": self.turns,
            "biomechanics": self.biomechanics,
            "biomechanics_result": self.biomechanics_result,
            "turn_segments": list(self.turn_segments),
            "pipeline_provenance": self.pipeline_provenance,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._semantic_dict()
        payload["analysis_context_sha256"] = self.analysis_context_sha256
        return _snapshot(payload)


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


def _turn_summary(segments: list[dict[str, object]]) -> dict[str, object]:
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


def _biomechanics_summary(
    report: dict[str, object],
    result: dict[str, object],
    segments: list[dict[str, object]],
) -> dict[str, object]:
    frame_facts = _dict_list(result, "frame_facts")
    _dict_list(result, "temporal_segment_features")
    turn_features = _dict_list(result, "turn_features")
    available_frame_facts = sum(
        item.get("status") == "AVAILABLE" and item.get("value") is not None for item in frame_facts
    )
    complete_turn_ids = {
        item.get("turn_id")
        for item in segments
        if item.get("status") == "VALID"
        and item.get("start_timestamp_us") is not None
        and item.get("end_timestamp_us") is not None
    }
    matched_turn_groups = sum(item.get("turn_id") in complete_turn_ids for item in turn_features)
    evidence_available = bool(complete_turn_ids and available_frame_facts)
    reason_codes = [] if evidence_available else ["INSUFFICIENT_BIOMECHANICS_EVIDENCE"]
    limitations = list(result.get("limitations", ()))
    limitations.extend(reason for reason in reason_codes if reason not in limitations)
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
        "availability_summary": {
            "status": "AVAILABLE" if evidence_available else "INSUFFICIENT",
            "available_frame_fact_count": available_frame_facts,
            "complete_diagnosis_eligible_turn_count": len(complete_turn_ids),
            "matched_turn_feature_group_count": matched_turn_groups,
            "reason_codes": reason_codes,
            "upstream_feature_coverage": result.get("feature_coverage"),
        },
        "coverage_summary": {
            "trusted_frame_count": _mapping(report, "frame_biomechanics").get("trusted_frame_count")
        },
        "limitations": limitations,
    }


def _pipeline_provenance(report: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    return {
        "execution_mode": "SINGLE_VIDEO_ANALYSIS_PASS",
        "video_read_count": 1,
        "automatic_sport_type_executed": False,
        "upstream_benchmark_contract_version": report.get("benchmark_contract_version"),
        "biomechanics_contract_version": result.get("contract_version"),
        "feature_schema_version": report.get("feature_schema_version"),
        "models": report.get("models"),
        "runtime": report.get("runtime"),
        "performance": report.get("performance"),
    }


def _mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"PRODUCT_{key.upper()}_MISSING")
    return value


def _dict_list(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"PRODUCT_{key.upper()}_MISSING")
    return value


def _nonnegative_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("PRODUCT_COUNT_INVALID")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _snapshot(value):
    # deepcopy preserves integer dictionary keys while the JSON pass validates determinism.
    _canonical_json(value)
    return copy.deepcopy(value)
