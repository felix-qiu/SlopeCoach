"""A6.3 local GT preparation and artifact-only calibration dataset extraction."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..contracts import SportEvidenceKind
from .aggregation import aggregate_subclips_by_source, summarize_observations
from .contracts import (
    SPORT_TYPE_CALIBRATION_DATASET_VERSION,
    SPORT_TYPE_GT_CONTRACT_VERSION,
    GroundTruthSportType,
    SportCalibrationFitConfig,
    SportTypeGroundTruth,
    strict_json,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_sport_type_gt(manifest_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    clips = manifest.get("clips")
    if not isinstance(clips, list):
        raise ValueError("manifest clips must be a list")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for clip in clips:
        clip_id = str(clip["clip_id"])
        source_id = str(clip.get("source_video_id", clip_id))
        video_path = Path(str(clip.get("path", clip.get("video_path", ""))))
        if not video_path.is_absolute():
            workspace_candidate = Path.cwd() / video_path
            video_path = (
                workspace_candidate
                if workspace_candidate.is_file()
                else Path(manifest_path).resolve().parent / video_path
            )
        if not video_path.is_file():
            raise ValueError(f"video missing for GT template: {video_path}")
        video_sha = sha256_file(video_path)
        expected_sha = clip.get("expected_sha256")
        if expected_sha is not None and expected_sha != video_sha:
            raise ValueError(f"manifest video SHA mismatch for {clip_id}")
        gt = SportTypeGroundTruth(clip_id, source_id, video_sha)
        output = destination / f"{clip_id}.sport_type.json"
        output.write_text(strict_json(gt.to_dict(), indent=2) + "\n", encoding="utf-8")
        outputs.append(str(output))
    return {
        "contract_version": SPORT_TYPE_GT_CONTRACT_VERSION,
        "status": "TEMPLATES_CREATED_REQUIRES_HUMAN_LABELING",
        "template_count": len(outputs),
        "outputs": outputs,
        "AUTO_LABEL_FROM_FILENAME": False,
        "AUTO_LABEL_FROM_MODEL": False,
    }


def build_calibration_dataset(
    artifact_paths: list[str | Path],
    annotations_dir: str | Path | None = None,
    config: SportCalibrationFitConfig | None = None,
) -> dict[str, object]:
    settings = config or SportCalibrationFitConfig()
    started = time.perf_counter()
    summaries = []
    clip_records = []
    raw_auto_decisions = []
    for artifact_path in artifact_paths:
        path = Path(artifact_path)
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("benchmark_contract_version") not in {
            "ski-bench-sport-type-v3",
            "ski-bench-sport-type-v4",
            "ski-bench-sport-type-v5",
        }:
            raise ValueError("incompatible SportType benchmark artifact")
        video = artifact.get("video", {})
        video_path = Path(str(video.get("path", "")))
        if not video_path.is_absolute():
            candidate = Path.cwd() / video_path
            video_path = candidate if candidate.is_file() else path.parent / video_path
        artifact_sha = video.get("sha256") or video.get("video_sha256")
        if artifact_sha:
            video_sha = str(artifact_sha)
            if video_path.is_file() and sha256_file(video_path) != video_sha:
                raise ValueError("artifact video SHA mismatch")
        elif video_path.is_file():
            video_sha = sha256_file(video_path)
        else:
            raise ValueError("artifact lacks video SHA and source video is unavailable")
        clip_id = Path(str(video.get("path", path.stem))).stem
        explicit_source_id = artifact.get("source_video_id")
        source_id = str(explicit_source_id or clip_id)
        source_id_origin = str(
            artifact.get("source_video_id_origin")
            or ("EXPLICIT" if explicit_source_id else "LEGACY_INFERRED")
        )
        model_maps = {
            SportEvidenceKind.EQUIPMENT: {
                item["provider_name"]: item for item in artifact.get("equipment_models", [])
            },
            SportEvidenceKind.VISUAL_CLASSIFIER: {
                item["provider_name"]: item for item in artifact.get("visual_models", [])
            },
        }
        provider_results = artifact.get("sport_type", {}).get("provider_results", [])
        for provider in provider_results:
            kind = SportEvidenceKind(provider["evidence_kind"])
            if not kind.is_primary:
                continue
            name = str(provider["provider_name"])
            summaries.append(
                summarize_observations(
                    provider_name=name,
                    evidence_kind=kind,
                    source_video_id=source_id,
                    video_sha256=video_sha,
                    observations=provider.get("observations", []),
                    provenance=model_maps[kind].get(name, {}),
                )
            )
        clip_records.append(
            {
                "clip_id": clip_id,
                "source_video_id": source_id,
                "source_video_id_origin": source_id_origin,
                "video_sha256": video_sha,
                "source_artifact": str(path),
            }
        )
        raw_auto_decisions.append(artifact.get("sport_type", {}).get("auto_decision"))
    source_summaries = aggregate_subclips_by_source(summaries)
    annotations = _load_annotations(annotations_dir)
    source_ids = sorted({item["source_video_id"] for item in clip_records})
    selected_annotations = {}
    for source_id in source_ids:
        values = [item for item in annotations.values() if item.source_video_id == source_id]
        if values:
            sha_values = {item.video_sha256 for item in values}
            expected = {
                item["video_sha256"]
                for item in clip_records
                if item["source_video_id"] == source_id
            }
            if sha_values != expected:
                raise ValueError(f"SportType GT video SHA mismatch for {source_id}")
            if (
                len(
                    {(item.target_sport_type, item.intended_target_confirmation) for item in values}
                )
                != 1
            ):
                raise ValueError(f"conflicting annotations for source {source_id}")
            selected_annotations[source_id] = values[0]
    counts = _annotation_counts(source_ids, selected_annotations)
    payload = {
        "contract_version": SPORT_TYPE_CALIBRATION_DATASET_VERSION,
        "dataset_id": "",
        "manifest_clip_count": len(clip_records),
        "enabled_clip_count": len(clip_records),
        "independent_source_video_count": len(source_ids),
        **counts,
        "clips": clip_records,
        "source_samples": [item.to_dict() for item in source_summaries],
        "annotations": {
            source_id: selected_annotations[source_id].to_dict()
            for source_id in sorted(selected_annotations)
        },
        "per_channel_usable_source_counts": {
            channel: sum(
                item.calibration_channel_id == channel and item.raw_direction is not None
                for item in source_summaries
            )
            for channel in sorted({item.calibration_channel_id for item in source_summaries})
        },
        "raw_auto_decisions": raw_auto_decisions,
        "performance": {"dataset_extraction_seconds": time.perf_counter() - started},
        "status": (
            "READY_FOR_CALIBRATION_FIT"
            if counts["ski_labeled_source_count"] >= settings.minimum_labeled_sources_per_class
            and counts["snowboard_labeled_source_count"]
            >= settings.minimum_labeled_sources_per_class
            else "INSUFFICIENT_LABELED_SPORT_TYPE_GT"
        ),
        "limitations": [
            "INDEPENDENT_SOURCE_VIDEO_IS_THE_CALIBRATION_UNIT",
            "FRAME_OBSERVATIONS_ARE_CORRELATED",
            "PYTHON_RESEARCH_REFERENCE_ONLY",
        ],
    }
    canonical = strict_json(semantic_dataset_payload(payload))
    payload["dataset_id"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def semantic_dataset_payload(payload: dict[str, object]) -> dict[str, object]:
    """Return deterministic dataset semantics without runtime measurements or self-ID."""
    return {
        key: value for key, value in payload.items() if key not in {"dataset_id", "performance"}
    }


def _load_annotations(path: str | Path | None) -> dict[str, SportTypeGroundTruth]:
    if path is None:
        return {}
    result = {}
    for item in sorted(Path(path).glob("*.json")):
        gt = SportTypeGroundTruth.from_dict(json.loads(item.read_text(encoding="utf-8")))
        result[gt.clip_id] = gt
    return result


def _annotation_counts(source_ids, annotations):
    labeled = [item for item in annotations.values() if item.eligible_for_fitting]
    uncertain = [
        item
        for item in annotations.values()
        if item.target_sport_type is GroundTruthSportType.UNCERTAIN
    ]
    unlabeled = len(source_ids) - len(labeled) - len(uncertain)
    return {
        "labeled_source_count": len(labeled),
        "ski_labeled_source_count": sum(
            item.target_sport_type is GroundTruthSportType.SKI for item in labeled
        ),
        "snowboard_labeled_source_count": sum(
            item.target_sport_type is GroundTruthSportType.SNOWBOARD for item in labeled
        ),
        "unlabeled_source_count": unlabeled,
        "uncertain_source_count": len(uncertain),
        "target_confirmed_labeled_source_count": len(labeled),
    }
