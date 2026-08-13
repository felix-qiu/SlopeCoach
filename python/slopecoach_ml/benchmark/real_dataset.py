"""A5.2 local real-video manifest and cross-clip robustness aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median

from slopecoach_ml.biomechanics import (
    BIOMECHANICS_FEATURE_SCHEMA_VERSION,
    FEATURE_REGISTRY_SHA256,
    FIXED_ML_FEATURE_VECTOR_STATUS,
    FRAME_FEATURE_REGISTRY_V1,
    TEMPORAL_FEATURE_REGISTRY_V1,
    TURN_FEATURE_REGISTRY_V1,
)
from slopecoach_ml.video import inspect_video

DATASET_CONTRACT_VERSION = "biomechanics-real-dataset-v1"
DATASET_BENCHMARK_CONTRACT_VERSION = "ski-bench-biomechanics-dataset-v1"
SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v"})
MIRROR_POLICIES = frozenset({"NON_MIRRORED", "MIRRORED"})


@dataclass(frozen=True)
class BiomechanicsDatasetValidationConfig:
    minimum_multiclip_source_videos: int = 5
    minimum_clip_feature_coverage_for_available: float = 0.50
    robust_clip_support_ratio: float = 0.80

    def validate(self) -> None:
        if (
            isinstance(self.minimum_multiclip_source_videos, bool)
            or not isinstance(self.minimum_multiclip_source_videos, int)
            or self.minimum_multiclip_source_videos < 2
        ):
            raise ValueError("minimum_multiclip_source_videos must be an integer >= 2")
        for name in (
            "minimum_clip_feature_coverage_for_available",
            "robust_clip_support_ratio",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class RealDatasetClip:
    clip_id: str
    source_video_id: str
    path: str
    sample_fps: float = 5.0
    mirror_policy: str = "NON_MIRRORED"
    enabled: bool = False
    expected_sha256: str | None = None
    manual_tags: dict[str, object] = field(default_factory=dict)
    observed_engineering_metrics: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        for name in ("clip_id", "source_video_id", "path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (
            isinstance(self.sample_fps, bool)
            or not isinstance(self.sample_fps, int | float)
            or not math.isfinite(self.sample_fps)
            or self.sample_fps <= 0
        ):
            raise ValueError("sample_fps must be finite and positive")
        if self.mirror_policy not in MIRROR_POLICIES:
            raise ValueError("mirror_policy must be explicit and supported")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be bool")
        if self.expected_sha256 is not None and (
            not isinstance(self.expected_sha256, str)
            or len(self.expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be null or lowercase SHA256 hex")
        if not isinstance(self.manual_tags, dict):
            raise ValueError("manual_tags must be an object")
        if not isinstance(self.observed_engineering_metrics, dict):
            raise ValueError("observed_engineering_metrics must be an object")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RealDatasetClip:
        if not isinstance(data, dict):
            raise ValueError("manifest clip must be an object")
        allowed = {
            "clip_id",
            "source_video_id",
            "path",
            "sample_fps",
            "mirror_policy",
            "enabled",
            "expected_sha256",
            "manual_tags",
            "observed_engineering_metrics",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown manifest clip fields: {sorted(unknown)}")
        try:
            result = cls(
                clip_id=data["clip_id"],
                source_video_id=data["source_video_id"],
                path=data["path"],
                sample_fps=data.get("sample_fps", 5.0),
                mirror_policy=data.get("mirror_policy", ""),
                enabled=data.get("enabled", False),
                expected_sha256=data.get("expected_sha256"),
                manual_tags=data.get("manual_tags", {}),
                observed_engineering_metrics=data.get("observed_engineering_metrics", {}),
            )
        except KeyError as error:
            raise ValueError(f"manifest clip missing field: {error.args[0]}") from error
        result.validate()
        return result

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RealDatasetManifest:
    dataset_id: str
    description: str
    clips: tuple[RealDatasetClip, ...]
    contract_version: str = DATASET_CONTRACT_VERSION

    def validate(self) -> None:
        if self.contract_version != DATASET_CONTRACT_VERSION:
            raise ValueError(f"contract_version must be {DATASET_CONTRACT_VERSION}")
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        for clip in self.clips:
            clip.validate()
        clip_ids = [clip.clip_id for clip in self.clips]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("clip_id values must be unique")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RealDatasetManifest:
        if not isinstance(data, dict):
            raise ValueError("dataset manifest must be an object")
        unknown = set(data) - {"contract_version", "dataset_id", "description", "clips"}
        if unknown:
            raise ValueError(f"unknown dataset manifest fields: {sorted(unknown)}")
        if data.get("contract_version") != DATASET_CONTRACT_VERSION:
            raise ValueError(f"unsupported dataset contract: {data.get('contract_version')}")
        clips = data.get("clips")
        if not isinstance(clips, list):
            raise ValueError("clips must be an array")
        result = cls(
            dataset_id=data.get("dataset_id"),
            description=data.get("description", ""),
            clips=tuple(RealDatasetClip.from_dict(item) for item in clips),
            contract_version=data["contract_version"],
        )
        result.validate()
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "dataset_id": self.dataset_id,
            "description": self.description,
            "clips": [clip.to_dict() for clip in self.clips],
        }


def load_real_dataset_manifest(path: str | Path) -> RealDatasetManifest:
    return RealDatasetManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_real_dataset_manifest(
    video_dir: str | Path,
    output_path: str | Path,
    *,
    known_enabled_clip_ids: frozenset[str] = frozenset({"ski_test_001"}),
) -> dict[str, object]:
    directory = Path(video_dir)
    if not directory.is_dir():
        raise ValueError(f"VIDEO_DIRECTORY_NOT_FOUND: {directory}")
    clips = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        metadata = inspect_video(path)
        clip_id = path.stem
        clips.append(
            RealDatasetClip(
                clip_id=clip_id,
                source_video_id=clip_id,
                path=path.as_posix(),
                sample_fps=5.0,
                mirror_policy="NON_MIRRORED",
                enabled=clip_id in known_enabled_clip_ids,
                expected_sha256=file_sha256(path),
                manual_tags={
                    "camera_view": "UNKNOWN",
                    "camera_motion": "UNKNOWN",
                    "subject_distance": "UNKNOWN",
                    "crowding": "UNKNOWN",
                    "occlusion": "UNKNOWN",
                    "motion_note": "",
                },
                observed_engineering_metrics={
                    "metadata_probe_status": "READABLE" if metadata.readable else "UNREADABLE",
                    "duration_seconds": metadata.duration_seconds,
                    "width_px": metadata.width_px,
                    "height_px": metadata.height_px,
                    "frame_count": metadata.frame_count,
                },
            )
        )
    manifest = RealDatasetManifest(
        dataset_id="local-biomechanics-a5-2",
        description="Local-only A5.2 real-video engineering robustness manifest.",
        clips=tuple(clips),
    )
    manifest.validate()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest.to_dict()


def real_data_validation_status(
    independent_source_video_count: int,
    config: BiomechanicsDatasetValidationConfig | None = None,
) -> str:
    settings = config or BiomechanicsDatasetValidationConfig()
    settings.validate()
    if independent_source_video_count == 0:
        return "NOT_ANALYZABLE_NO_REAL_CLIPS"
    if independent_source_video_count == 1:
        return "INSUFFICIENT_DATASET_SINGLE_CLIP"
    if independent_source_video_count < settings.minimum_multiclip_source_videos:
        return "LIMITED_MULTICLIP_EVIDENCE"
    return "MULTICLIP_ENGINEERING_EVIDENCE"


def execute_biomechanics_dataset(
    manifest: RealDatasetManifest,
    benchmark_clip: Callable[[RealDatasetClip, Path, Path | None], dict[str, object]],
    *,
    per_clip_output_dir: str | Path | None = None,
    debug_dir: str | Path | None = None,
    config: BiomechanicsDatasetValidationConfig | None = None,
) -> dict[str, object]:
    """Run enabled clips sequentially with per-clip failure isolation."""
    manifest.validate()
    output_root = Path(per_clip_output_dir) if per_clip_output_dir else None
    debug_root = Path(debug_dir) if debug_dir else None
    records = []
    for clip in (item for item in manifest.clips if item.enabled):
        path = Path(clip.path)
        record = {"clip_id": clip.clip_id, "source_video_id": clip.source_video_id}
        if clip.mirror_policy != "NON_MIRRORED":
            records.append(
                {
                    **record,
                    "execution_status": "BENCHMARK_FAILED",
                    "error": "MIRRORED_INPUT_CANONICALIZATION_NOT_CONFIGURED",
                }
            )
            continue
        if not path.is_file():
            records.append({**record, "execution_status": "VIDEO_NOT_FOUND", "error": str(path)})
            continue
        actual_sha = file_sha256(path)
        if clip.expected_sha256 is not None and clip.expected_sha256 != actual_sha:
            records.append(
                {
                    **record,
                    "sha256": actual_sha,
                    "execution_status": "SHA_MISMATCH",
                    "error": "actual SHA256 does not match manifest",
                }
            )
            continue
        artifact_path = output_root / f"{clip.clip_id}.json" if output_root else None
        clip_debug = debug_root / clip.clip_id if debug_root else None
        try:
            report = benchmark_clip(clip, path, clip_debug)
            if report.get("benchmark_contract_version") != "ski-bench-biomechanics-v2":
                raise ValueError("per-clip benchmark contract must be ski-bench-biomechanics-v2")
            if report.get("feature_registry_sha256") != FEATURE_REGISTRY_SHA256:
                raise ValueError("per-clip feature registry SHA256 mismatch")
            if artifact_path:
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            records.append(
                {
                    **record,
                    "sha256": actual_sha,
                    "execution_status": "SUCCESS",
                    "result_source": "FRESH_EXECUTION",
                    "per_clip_artifact": str(artifact_path) if artifact_path else None,
                    "report": report,
                }
            )
        except Exception as error:  # clip isolation is an explicit dataset policy
            message = f"{type(error).__name__}: {error}"
            if "DECODE" in message or "VIDEO" in message and "read" in message.lower():
                status = "VIDEO_DECODE_FAILED"
            elif any(token in message for token in ("MODEL", "OpenMMLab", "mmdet", "mmpose")):
                status = "MODEL_RUNTIME_FAILED"
            else:
                status = "BENCHMARK_FAILED"
            records.append(
                {
                    **record,
                    "sha256": actual_sha,
                    "execution_status": status,
                    "error": message,
                }
            )
    report = aggregate_biomechanics_dataset(manifest, records, config)
    report["artifacts"] = (
        write_coverage_csvs(report, output_root.parent if output_root else Path("artifacts"))
        if output_root
        else {}
    )
    if debug_root:
        report["artifacts"]["dataset_contact_sheet"] = write_dataset_contact_sheet(
            report, debug_root
        )
    json.dumps(report, sort_keys=True, allow_nan=False)
    return report


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "median": median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "range": max(values) - min(values) if values else None,
    }


def _domain_violation(feature_id: str, value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return "NONFINITE_OR_NONNUMERIC"
    if (
        feature_id
        in {
            "left_knee_angle_2d_deg",
            "right_knee_angle_2d_deg",
            "bilateral_knee_mean_angle_2d_deg",
            "bilateral_knee_abs_difference_2d_deg",
        }
        and not 0 <= value <= 180
    ):
        return "KNEE_ANGLE_DOMAIN"
    if feature_id == "shoulder_hip_axis_difference_2d_deg" and not 0 <= value <= 90:
        return "AXIS_DIFFERENCE_DOMAIN"
    if feature_id == "signed_lateral_body_proxy" and not -1.000000001 <= value <= 1.000000001:
        return "SIGNED_LATERAL_PROXY_DOMAIN"
    if (
        feature_id.endswith("_separation_body_scale")
        or feature_id == "ankle_to_shoulder_separation_ratio_2d"
    ) and value < 0:
        return "NORMALIZED_DISTANCE_DOMAIN"
    return None


def _feature_status(
    *,
    violations: int,
    independent_sources: int,
    clips_available: int,
    evaluable_clips: int,
    macro_mean: float | None,
    config: BiomechanicsDatasetValidationConfig,
) -> tuple[str, str]:
    if violations:
        return "CONTRACT_FAILURE", "REVIEW_IMPLEMENTATION"
    if clips_available == 0:
        return "NO_EVIDENCE", "NEED_MORE_DATA"
    if independent_sources < config.minimum_multiclip_source_videos or clips_available < 2:
        return "LIMITED_EVIDENCE", "NEED_MORE_DATA"
    support = clips_available / evaluable_clips if evaluable_clips else 0.0
    if (
        support >= config.robust_clip_support_ratio
        and macro_mean is not None
        and macro_mean >= config.minimum_clip_feature_coverage_for_available
    ):
        return "ROBUST_COVERAGE_EVIDENCE", "KEEP"
    return "USABLE_WITH_LIMITATIONS", "KEEP_WITH_LIMITATIONS"


def _successful(record: dict[str, object]) -> bool:
    return record.get("execution_status") == "SUCCESS" and isinstance(record.get("report"), dict)


def aggregate_biomechanics_dataset(
    manifest: RealDatasetManifest,
    execution_records: list[dict[str, object]],
    config: BiomechanicsDatasetValidationConfig | None = None,
) -> dict[str, object]:
    settings = config or BiomechanicsDatasetValidationConfig()
    settings.validate()
    manifest.validate()
    started = time.perf_counter()
    enabled = [clip for clip in manifest.clips if clip.enabled]
    record_by_id = {record["clip_id"]: record for record in execution_records}
    if len(record_by_id) != len(execution_records):
        raise ValueError("execution record clip IDs must be unique")
    successful_records = [record for record in execution_records if _successful(record)]
    actual_source_ids = {
        record["source_video_id"]
        for record in execution_records
        if record.get("execution_status") not in {"VIDEO_NOT_FOUND", "SHA_MISMATCH"}
    }
    independent_sources = len(actual_source_ids)
    evaluable_records = [
        record
        for record in successful_records
        if record["report"]["frame_biomechanics"]["trusted_frame_count"] > 0
    ]
    failure_matrix: dict[str, dict[str, dict[str, int]]] = {}
    contract_totals = Counter()
    frame_results = []
    for definition in FRAME_FEATURE_REGISTRY_V1:
        coverages, available_total, trusted_total, clips_any, clips_aggregate = [], 0, 0, 0, 0
        failure_totals = Counter()
        clip_values = []
        violations = 0
        for record in evaluable_records:
            report = record["report"]
            coverage = report["frame_biomechanics"]["feature_coverage"][definition.feature_id]
            ratio = coverage["coverage_ratio"]
            coverages.append(ratio)
            available_total += coverage["available_frame_count"]
            trusted_total += coverage["total_trusted_frames"]
            clips_any += coverage["available_frame_count"] > 0
            reasons = Counter(coverage["status_reason_counts"])
            available_count = coverage["available_frame_count"]
            failure_matrix.setdefault(record["clip_id"], {})[definition.feature_id] = {
                "AVAILABLE": available_count,
                **dict(reasons),
            }
            failure_totals.update(reasons)
            aggregates = [
                item
                for item in report["temporal_segment_features"]
                if item["feature_id"] == definition.feature_id and item["status"] == "AVAILABLE"
            ]
            clips_aggregate += bool(aggregates)
            if aggregates:
                clip_values.append(median(item["median"] for item in aggregates))
            for fact in report["biomechanics_result"]["frame_facts"]:
                if fact["feature_id"] == definition.feature_id and fact["status"] == "AVAILABLE":
                    violation = _domain_violation(definition.feature_id, fact["value"])
                    if violation:
                        violations += 1
                        contract_totals[violation] += 1
        macro_mean = mean(coverages) if coverages else None
        status, recommendation = _feature_status(
            violations=violations,
            independent_sources=independent_sources,
            clips_available=clips_any,
            evaluable_clips=len(evaluable_records),
            macro_mean=macro_mean,
            config=settings,
        )
        frame_results.append(
            {
                "feature_id": definition.feature_id,
                "clip_evaluable_count": len(evaluable_records),
                "clip_with_any_available_count": clips_any,
                "clip_with_aggregate_available_count": clips_aggregate,
                "macro_coverage_mean": macro_mean,
                "macro_coverage_median": median(coverages) if coverages else None,
                "macro_coverage_min": min(coverages) if coverages else None,
                "macro_coverage_max": max(coverages) if coverages else None,
                "micro_available_frame_count": available_total,
                "micro_trusted_frame_count": trusted_total,
                "micro_coverage_ratio": available_total / trusted_total if trusted_total else None,
                "top_failure_reason": min(
                    failure_totals,
                    key=lambda key: (-failure_totals[key], key),
                    default=None,
                ),
                "failure_reason_counts": dict(sorted(failure_totals.items())),
                "contract_violation_count": violations,
                "clip_value_distribution": _summary(clip_values),
                "robustness_status": status,
                "retention_recommendation": recommendation,
            }
        )
    temporal_results = []
    for definition in TEMPORAL_FEATURE_REGISTRY_V1:
        items = [
            item
            for record in evaluable_records
            for item in record["report"]["temporal_segment_features"]
            if item["feature_id"] == definition.feature_id
        ]
        available = [item for item in items if item["status"] == "AVAILABLE"]
        values = [item["median"] for item in available]
        temporal_results.append(
            {
                "feature_id": definition.feature_id,
                "eligible_temporal_segment_count": len(items),
                "available_segment_count": len(available),
                "coverage_ratio": len(available) / len(items) if items else None,
                "value_distribution": _summary(values),
                "robustness_status": "LIMITED_EVIDENCE" if available else "NO_EVIDENCE",
            }
        )
    turn_results = []
    all_turns = [
        turn
        for record in successful_records
        for turn in record["report"].get("turn_biomechanics", [])
    ]
    for definition in TURN_FEATURE_REGISTRY_V1:
        facts = [
            fact
            for turn in all_turns
            for fact in turn["facts"]
            if fact["feature_id"] == definition.feature_id
        ]
        available = sum(fact["status"] == "AVAILABLE" for fact in facts)
        turn_results.append(
            {
                "feature_id": definition.feature_id,
                "eligible_turn_count": len(facts),
                "available_turn_fact_count": available,
                "coverage_ratio": available / len(facts) if facts else None,
                "robustness_status": (
                    "LIMITED_EVIDENCE" if facts else "NOT_EVALUATED_NO_QUALIFIED_TURNS"
                ),
            }
        )
    per_clip = [_compact_clip(record_by_id.get(clip.clip_id), clip) for clip in enabled]
    performance = _dataset_performance(successful_records)
    complete_turn_count = sum(
        segment.get("start_timestamp_us") is not None
        and segment.get("end_timestamp_us") is not None
        and segment.get("duration_us") is not None
        and segment["duration_us"] > 0
        for record in successful_records
        for segment in record["report"].get("turn_segments", [])
        if segment["status"] in {"VALID", "PARTIAL"}
    )
    partial_turn_count = sum(
        segment["status"] == "PARTIAL"
        for record in successful_records
        for segment in record["report"].get("turn_segments", [])
    )
    report = {
        "benchmark_contract_version": DATASET_BENCHMARK_CONTRACT_VERSION,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "dataset_id": manifest.dataset_id,
        "feature_schema_version": BIOMECHANICS_FEATURE_SCHEMA_VERSION,
        "feature_registry_sha256": FEATURE_REGISTRY_SHA256,
        "FIXED_ML_FEATURE_VECTOR_STATUS": FIXED_ML_FEATURE_VECTOR_STATUS,
        "config": {"profile": "RESEARCH_DEFAULTS_A5_2", **asdict(settings)},
        "dataset": {
            "manifest_clip_count": len(manifest.clips),
            "enabled_clip_count": len(enabled),
            "independent_source_video_count": independent_sources,
            "successful_clip_count": len(successful_records),
            "feature_evaluable_clip_count": len(evaluable_records),
            "upstream_non_evaluable_clip_count": len(successful_records) - len(evaluable_records),
        },
        "per_clip": per_clip,
        "frame_feature_robustness": frame_results,
        "temporal_feature_robustness": temporal_results,
        "turn_feature_robustness": turn_results,
        "turn_evidence": {
            "clips_with_valid_turn_signal": sum(
                record["report"].get("turn_signal_summary", {}).get("valid_sample_count", 0) > 0
                for record in successful_records
            ),
            "clips_with_qualified_turns": sum(
                record["report"]["turn_input"]["qualified_turn_count"] > 0
                for record in successful_records
            ),
            "clips_with_complete_turns": sum(
                any(
                    segment.get("start_timestamp_us") is not None
                    and segment.get("end_timestamp_us") is not None
                    and segment.get("duration_us", 0) > 0
                    and segment["status"] in {"VALID", "PARTIAL"}
                    for segment in record["report"].get("turn_segments", [])
                )
                for record in successful_records
            ),
            "clips_with_partial_turns": sum(
                any(
                    segment["status"] == "PARTIAL"
                    for segment in record["report"].get("turn_segments", [])
                )
                for record in successful_records
            ),
            "total_qualified_turns": len(all_turns),
            "complete_turn_count": complete_turn_count,
            "partial_turn_count": partial_turn_count,
        },
        "upstream_conditions": _upstream_summary(successful_records),
        "failure_reason_matrix": {
            "per_clip_feature": failure_matrix,
            "dataset_status_totals": _failure_totals(frame_results),
        },
        "contract_checks": {
            "knee_angle_domain_violations": contract_totals["KNEE_ANGLE_DOMAIN"],
            "axis_difference_domain_violations": contract_totals["AXIS_DIFFERENCE_DOMAIN"],
            "signed_lateral_proxy_domain_violations": contract_totals[
                "SIGNED_LATERAL_PROXY_DOMAIN"
            ],
            "normalized_distance_domain_violations": contract_totals["NORMALIZED_DISTANCE_DOMAIN"],
            "other_contract_violations": contract_totals["NONFINITE_OR_NONNUMERIC"],
            "total_contract_violation_count": sum(contract_totals.values()),
        },
        "performance": {**performance, "aggregation_seconds": time.perf_counter() - started},
        "ground_truth": {
            "TARGET_IDENTITY_GT_ANNOTATION_STATUS": "DEFERRED",
            "TARGET_IDENTITY_ACCURACY_STATUS": "UNKNOWN",
            "TURN_SEGMENTATION_GT_STATUS": "NOT_AVAILABLE",
            "BIOMECHANICS_GT_STATUS": "NOT_AVAILABLE",
            "feature_accuracy": None,
            "biomechanics_mae": None,
        },
        "validation": {
            "A5_2_ENGINEERING_VALIDATION": "PASS",
            "A5_2_REAL_DATA_VALIDATION": real_data_validation_status(independent_sources, settings),
            "A5_PRODUCT_VALIDATION": "BLOCKED_BY_GT",
        },
        "limitations": [
            "HIGH_FEATURE_COVERAGE_DOES_NOT_IMPLY_ACCURACY",
            "ENGINEERING_EVIDENCE_AVAILABILITY_NOT_BIOMECHANICS_ACCURACY",
            "NO_IDENTITY_TURN_OR_BIOMECHANICS_GROUND_TRUTH",
            "IMAGE_SPACE_2D_CAMERA_DEPENDENT",
            "MINIMUM_SOURCE_VIDEO_THRESHOLD_IS_PROJECT_GOVERNANCE_NOT_STATISTICAL_PROOF",
            "PYTHON_RESEARCH_REFERENCE_ONLY",
            "NO_DIAGNOSIS_OR_SCORE",
        ],
    }
    json.dumps(report, sort_keys=True, allow_nan=False)
    return report


def _compact_clip(record: dict[str, object] | None, clip: RealDatasetClip) -> dict[str, object]:
    base = {
        "clip_id": clip.clip_id,
        "source_video_id": clip.source_video_id,
        "path": clip.path,
        "sample_fps": clip.sample_fps,
        "manual_tags": clip.manual_tags,
    }
    if record is None:
        return {**base, "execution_status": "BENCHMARK_FAILED", "error": "NO_EXECUTION_RECORD"}
    if not _successful(record):
        return {**base, **record, "report": None}
    report = record["report"]
    sampled = report["sampling"]["sampled_frame_count"]
    locked = report["identity_input"]["identity_locked_frame_count"]
    signal = report.get("turn_signal_summary", {})
    knee = next(
        (
            item
            for item in report["temporal_segment_features"]
            if item["feature_id"] == "bilateral_knee_mean_angle_2d_deg"
            and item["status"] == "AVAILABLE"
        ),
        None,
    )
    velocity = next(
        (
            item
            for item in report["temporal_segment_features"]
            if item["feature_id"] == "bilateral_knee_mean_angle_abs_velocity_median_deg_per_s"
            and item["status"] == "AVAILABLE"
        ),
        None,
    )
    return {
        **base,
        "sha256": record.get("sha256"),
        "execution_status": "SUCCESS",
        "result_source": record.get("result_source", "FRESH_EXECUTION"),
        "video_duration_seconds": report["video"]["duration_seconds"],
        "sampled_frames": sampled,
        "raw_detection_count": report.get("upstream_conditions", {}).get("raw_detection_count"),
        "identity_locked_frames": locked,
        "identity_unsafe_frames": report["identity_input"]["identity_unsafe_frame_count"],
        "identity_locked_ratio": locked / sampled if sampled else None,
        "temporal_segment_count": report["temporal_input"]["temporal_segment_count"],
        "valid_turn_signal_samples": signal.get("valid_sample_count"),
        "valid_signal_run_count": signal.get("valid_signal_run_count"),
        "longest_signal_run_duration_us": signal.get("longest_valid_signal_run_duration_us"),
        "qualified_turn_count": report["turn_input"]["qualified_turn_count"],
        "real_turn_status": report["turn_input"]["turn_status"],
        "trusted_biomechanics_frames": report["frame_biomechanics"]["trusted_frame_count"],
        "real_biomechanics_status": report["validation"]["REAL_BIOMECHANICS_STATUS"],
        "feature_coverage": report["frame_biomechanics"]["feature_coverage"],
        "observed_engineering_metrics": report.get("upstream_conditions", {}),
        "motion_evidence": {
            "signal_value_span": signal.get("signal_value_span"),
            "median_absolute_signal_delta": signal.get("median_absolute_signal_delta"),
            "qualified_turn_count": report["turn_input"]["qualified_turn_count"],
            "bilateral_knee_mean_angle_range": knee["range"] if knee else None,
            "bilateral_knee_mean_angle_velocity_median": velocity["median"] if velocity else None,
        },
        "performance": report["performance"],
        "per_clip_artifact": record.get("per_clip_artifact"),
    }


def _failure_totals(frame_results: list[dict[str, object]]) -> dict[str, int]:
    totals = Counter(
        {
            status: 0
            for status in (
                "LOW_CONFIDENCE",
                "REQUIRED_JOINT_MISSING",
                "REQUIRED_JOINT_OUT_OF_FRAME",
                "UNSUPPORTED_PIXEL_ASPECT_RATIO",
                "DEGENERATE_GEOMETRY",
                "INSUFFICIENT_EVIDENCE",
            )
        }
    )
    for result in frame_results:
        totals.update(result["failure_reason_counts"])
    return dict(sorted(totals.items()))


def _upstream_summary(records: list[dict[str, object]]) -> dict[str, object]:
    fields = (
        "raw_candidate_density",
        "median_target_bbox_height_ratio",
        "median_target_bbox_area_ratio",
        "required_joint_visibility_ratio",
    )
    result = {}
    for name in fields:
        values = [record["report"].get("upstream_conditions", {}).get(name) for record in records]
        result[name] = _summary([value for value in values if value is not None])
    locked_ratios = []
    for record in records:
        report = record["report"]
        count = report["sampling"]["sampled_frame_count"]
        if count:
            locked_ratios.append(report["identity_input"]["identity_locked_frame_count"] / count)
    result["identity_locked_ratio"] = _summary(locked_ratios)
    return result


def _dataset_performance(records: list[dict[str, object]]) -> dict[str, float]:
    result = {
        "total_detector_seconds": 0.0,
        "total_pose_seconds": 0.0,
        "total_tracking_identity_seconds": 0.0,
        "total_temporal_seconds": 0.0,
        "total_turn_seconds": 0.0,
        "total_biomechanics_seconds": 0.0,
        "dataset_total_seconds": 0.0,
    }
    for record in records:
        perf = record["report"]["performance"]
        result["total_detector_seconds"] += perf["detector_total_seconds"]
        result["total_pose_seconds"] += perf["pose_total_seconds"]
        result["total_tracking_identity_seconds"] += perf["tracking_identity_total_seconds"]
        result["total_temporal_seconds"] += perf["temporal_total_seconds"]
        result["total_turn_seconds"] += perf["turn_total_seconds"]
        result["total_biomechanics_seconds"] += (
            perf["biomechanics_frame_total_seconds"]
            + perf["biomechanics_aggregation_total_seconds"]
            + perf["biomechanics_turn_total_seconds"]
        )
        result["dataset_total_seconds"] += perf["total_seconds"]
    return result


def write_coverage_csvs(report: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frame_path = destination / "feature_coverage_matrix.csv"
    temporal_path = destination / "temporal_feature_coverage_matrix.csv"
    clips = report["per_clip"]
    with frame_path.open("w", newline="", encoding="utf-8") as target:
        fieldnames = ["clip_id", *(item.feature_id for item in FRAME_FEATURE_REGISTRY_V1)]
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for clip in clips:
            row = {"clip_id": clip["clip_id"]}
            if clip.get("trusted_biomechanics_frames", 0) > 0:
                row.update(
                    {
                        feature_id: values["coverage_ratio"]
                        for feature_id, values in clip["feature_coverage"].items()
                    }
                )
            writer.writerow(row)
    with temporal_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "feature_id",
                "eligible_segment_count",
                "available_segment_count",
                "coverage_ratio",
            ],
        )
        writer.writeheader()
        for item in report["temporal_feature_robustness"]:
            writer.writerow(
                {
                    "feature_id": item["feature_id"],
                    "eligible_segment_count": item["eligible_temporal_segment_count"],
                    "available_segment_count": item["available_segment_count"],
                    "coverage_ratio": item["coverage_ratio"]
                    if item["coverage_ratio"] is not None
                    else "",
                }
            )
    return {
        "frame_feature_coverage_csv": str(frame_path),
        "temporal_feature_coverage_csv": str(temporal_path),
    }


def write_dataset_contact_sheet(
    report: dict[str, object], debug_dir: str | Path, *, frames_per_clip: int = 2
) -> str | None:
    """Compose a capped engineering sheet from already-generated per-clip frames."""
    if frames_per_clip < 1 or frames_per_clip > 2:
        raise ValueError("frames_per_clip must be in [1, 2]")
    try:
        import cv2
    except ImportError:
        return None
    root = Path(debug_dir)
    rows = []
    for clip in report["per_clip"]:
        if clip.get("execution_status") != "SUCCESS":
            continue
        images = sorted((root / clip["clip_id"]).glob("frame_*.jpg"))[:frames_per_clip]
        tiles = []
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                continue
            width = 360
            height = max(1, round(image.shape[0] * width / image.shape[1]))
            image = cv2.resize(image, (width, height))
            cv2.putText(
                image,
                f"{clip['clip_id']} {path.stem}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            tiles.append(image)
        if tiles:
            while len(tiles) < frames_per_clip:
                tiles.append(tiles[-1].copy())
            rows.append(cv2.hconcat(tiles))
    if not rows:
        return None
    width = max(row.shape[1] for row in rows)
    padded = [
        cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT) for row in rows
    ]
    destination = root / "dataset_contact_sheet.jpg"
    if not cv2.imwrite(str(destination), cv2.vconcat(padded)):
        raise RuntimeError("DATASET_CONTACT_SHEET_WRITE_FAILED")
    return str(destination)
